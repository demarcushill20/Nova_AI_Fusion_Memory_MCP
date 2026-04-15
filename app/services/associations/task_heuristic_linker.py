"""Task-status heuristic linker (Phase 7b, PLAN-0759).

Overview
--------

This module lands :class:`TaskHeuristicLinker` — a write-time heuristic that
creates ``CAUSED_BY`` edges when a ``bug_fix`` memory references a
``task_failed`` memory via the ``fixes_task_id`` metadata key.

The heuristic is pattern-matching, NOT causal inference:

    memory_a.category == "bug_fix"
    AND memory_a.fixes_task_id == memory_b.memory_id
    AND memory_b.category == "task_failed"
    AND memory_a.project == memory_b.project
    AND both event_time fields are present and parseable
    AND memory_a.event_time > memory_b.event_time
    AND (memory_a.event_time - memory_b.event_time) < 30 days

Edge spec
---------

- Edge type: ``CAUSED_BY`` (directed: effect -> cause, i.e. bug_fix -> task_failed)
- Weight: 1.0 (binary heuristic)
- ``created_by = "task_heuristic_linker"``
- ``metadata = None`` (Neo4j 5 rejects Map-valued relationship props)

Concurrency model
-----------------

Mirrors the semaphore-bounded task-per-call pattern from Sprints 6/9/10.
A global semaphore (``BACKGROUND_MAX_IN_FLIGHT = 32``) bounds total
in-flight linker tasks. When saturated, new requests are dropped with a
logged warning rather than queuing unboundedly.

Memory fetcher injection
------------------------

The linker needs to look up the target memory's metadata (category, project,
event_time) by ID. It takes an async callable ``memory_fetcher`` with
signature ``(memory_id: str) -> dict | None`` that returns the memory's
metadata dict or ``None`` if the memory does not exist. Production callers
wire this to ``PineconeClient.fetch_vectors``; tests supply a controllable
async mock.

Exception containment
---------------------

:meth:`_link_one_safe` wraps :meth:`_link_one` in ``asyncio.wait_for``
with a ``BACKGROUND_TIMEOUT`` deadline. Every failure path is caught,
logged as ``task_heuristic_link.failed``, and swallowed. This is the
contract that protects ``perform_upsert()``.

Module invariants
-----------------

- Does **not** import ``app.config.settings``. The feature flag check
  lives in the ``perform_upsert()`` hook, not in the linker itself.
- Does **not** create a Neo4j driver at import or construct time.
- Does **not** pass Map-valued metadata onto the relationship.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from .memory_edges import MemoryEdge

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from .edge_service import MemoryEdgeService

logger = logging.getLogger(__name__)


#: Type alias for the memory-fetcher callable contract. Returns a metadata
#: dict for the given memory_id, or ``None`` if the memory does not exist.
MemoryFetcher = Callable[[str], Awaitable[Optional[dict[str, Any]]]]


class TaskHeuristicLinker:
    """Fire-and-forget write-time task-status heuristic linker.

    See module docstring for the full design rationale. The hot path is
    :meth:`enqueue_link`; every other method is internal plumbing that
    tests call directly.
    """

    #: Maximum time delta between bug_fix and task_failed memories.
    MAX_TIME_DELTA_DAYS: int = 30

    #: Per-call upper bound on total linker work.
    BACKGROUND_TIMEOUT: float = 30.0

    #: Maximum number of simultaneously in-flight linker tasks.
    BACKGROUND_MAX_IN_FLIGHT: int = 32

    def __init__(
        self,
        edge_service: "MemoryEdgeService",
        memory_fetcher: Optional[MemoryFetcher] = None,
    ) -> None:
        """Store injected dependencies.

        Parameters
        ----------
        edge_service:
            A fully-constructed :class:`MemoryEdgeService` bound to an
            async Neo4j driver.
        memory_fetcher:
            An async callable ``(memory_id: str) -> dict | None`` that
            returns memory metadata by ID. If ``None``, the linker is
            effectively disabled — :meth:`enqueue_link` will accept calls
            but :meth:`_link_one` will return 0 on every invocation.
        """
        self._edge_service = edge_service
        self._memory_fetcher = memory_fetcher
        self._semaphore = asyncio.Semaphore(self.BACKGROUND_MAX_IN_FLIGHT)

        # Strong references to in-flight tasks so the asyncio loop does
        # not GC them mid-flight.
        self._inflight: set[asyncio.Task[int]] = set()

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #

    async def enqueue_link(
        self,
        memory_id: str,
        metadata: dict[str, Any],
        project: Optional[str],
        *,
        run_id: Optional[str] = None,
    ) -> None:
        """Dispatch a background task-heuristic linking task and return.

        Only dispatches if the memory is a ``bug_fix`` with a
        ``fixes_task_id`` metadata key. Otherwise returns immediately.

        Parameters
        ----------
        memory_id:
            ``entity_id`` of the freshly-persisted ``:base`` memory.
        metadata:
            Full metadata dict of the freshly-persisted memory.
        project:
            Project scoping tag from the metadata.
        run_id:
            Optional per-call run_id for rollback. If ``None``, a fresh
            one is generated.
        """
        if not isinstance(project, str) or not project.strip() or not isinstance(metadata, dict) or not metadata:
            return

        category = metadata.get("category", "")
        fixes_task_id = metadata.get("fixes_task_id")
        if category != "bug_fix" or not isinstance(fixes_task_id, str) or not fixes_task_id.strip():
            return

        if memory_id == fixes_task_id.strip():
            return

        # Saturation guard: drop if in-flight tasks are at capacity.
        if len(self._inflight) >= self.BACKGROUND_MAX_IN_FLIGHT:
            logger.warning(
                "task_heuristic_link.dropped_at_capacity "
                "memory_id=%s inflight=%d max=%d",
                memory_id,
                len(self._inflight),
                self.BACKGROUND_MAX_IN_FLIGHT,
            )
            return

        effective_run_id = run_id if run_id is not None else f"wt-taskheur-{uuid.uuid4().hex[:8]}"

        logger.info(
            "task_heuristic_link.queued memory_id=%s run_id=%s "
            "fixes_task_id=%s project=%s",
            memory_id,
            effective_run_id,
            fixes_task_id,
            project,
        )

        task = asyncio.create_task(
            self._link_one_safe(
                memory_id=memory_id,
                target_id=str(fixes_task_id),
                metadata=metadata,
                project=project,
                run_id=effective_run_id,
            )
        )
        # Strong reference so the task isn't GC'd mid-flight.
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    # ------------------------------------------------------------------ #
    # Internal plumbing                                                  #
    # ------------------------------------------------------------------ #

    async def _link_one_safe(
        self,
        memory_id: str,
        target_id: str,
        metadata: dict[str, Any],
        project: str,
        run_id: str,
    ) -> int:
        """Exception-containment wrapper around :meth:`_link_one`."""
        try:
            async with self._semaphore:
                return await asyncio.wait_for(
                    self._link_one(
                        memory_id=memory_id,
                        target_id=target_id,
                        metadata=metadata,
                        project=project,
                        run_id=run_id,
                    ),
                    timeout=self.BACKGROUND_TIMEOUT,
                )
        except asyncio.CancelledError:
            logger.debug(
                "task_heuristic_link.cancelled memory_id=%s run_id=%s",
                memory_id,
                run_id,
            )
            return 0
        except asyncio.TimeoutError:
            logger.warning(
                "task_heuristic_link.failed memory_id=%s run_id=%s "
                "reason=timeout timeout_s=%s",
                memory_id,
                run_id,
                self.BACKGROUND_TIMEOUT,
            )
            return 0
        except Exception as exc:  # noqa: BLE001 — fail-open envelope
            logger.warning(
                "task_heuristic_link.failed memory_id=%s run_id=%s "
                "reason=%s error=%s",
                memory_id,
                run_id,
                type(exc).__name__,
                exc,
            )
            return 0

    async def _link_one(
        self,
        memory_id: str,
        target_id: str,
        metadata: dict[str, Any],
        project: str,
        run_id: str,
    ) -> int:
        """Create CAUSED_BY edge if target is a task_failed memory within 30 days.

        Returns the number of edges created (0 or 1).
        """
        if not self._memory_fetcher:
            return 0

        target_meta = await self._memory_fetcher(target_id)
        if not isinstance(target_meta, dict) or not target_meta:
            logger.info(
                "task_heuristic_link.target_not_found memory_id=%s "
                "target_id=%s run_id=%s",
                memory_id,
                target_id,
                run_id,
            )
            return 0

        # Validate: target must be task_failed
        if target_meta.get("category") != "task_failed":
            logger.info(
                "task_heuristic_link.target_not_task_failed memory_id=%s "
                "target_id=%s target_category=%s run_id=%s",
                memory_id,
                target_id,
                target_meta.get("category"),
                run_id,
            )
            return 0

        # Validate: same project
        if target_meta.get("project") != project:
            logger.info(
                "task_heuristic_link.cross_project memory_id=%s "
                "target_id=%s source_project=%s target_project=%s run_id=%s",
                memory_id,
                target_id,
                project,
                target_meta.get("project"),
                run_id,
            )
            return 0

        # Validate: time delta < 30 days (fail-closed: missing or
        # unparseable timestamps reject the edge).
        source_time_str = metadata.get("event_time", "")
        target_time_str = target_meta.get("event_time", "")
        if (
            not isinstance(source_time_str, str) or not source_time_str
            or not isinstance(target_time_str, str) or not target_time_str
        ):
            logger.info(
                "task_heuristic_link.missing_timestamp memory_id=%s "
                "target_id=%s run_id=%s",
                memory_id,
                target_id,
                run_id,
            )
            return 0
        try:
            # Normalize "Z" suffix for Python 3.10 compatibility
            if source_time_str.endswith("Z"):
                source_time_str = source_time_str[:-1] + "+00:00"
            if target_time_str.endswith("Z"):
                target_time_str = target_time_str[:-1] + "+00:00"
            source_time = datetime.fromisoformat(source_time_str)
            target_time = datetime.fromisoformat(target_time_str)
            # Normalize naive datetimes to UTC-aware to prevent TypeError
            if source_time.tzinfo is None:
                source_time = source_time.replace(tzinfo=timezone.utc)
            if target_time.tzinfo is None:
                target_time = target_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError) as e:
            logger.warning(
                "task_heuristic_link.time_parse_failed memory_id=%s "
                "target_id=%s error=%s run_id=%s",
                memory_id,
                target_id,
                e,
                run_id,
            )
            return 0
        if source_time <= target_time:
            logger.info(
                "task_heuristic_link.wrong_time_order memory_id=%s "
                "target_id=%s run_id=%s",
                memory_id,
                target_id,
                run_id,
            )
            return 0
        delta = source_time - target_time
        if delta >= timedelta(days=self.MAX_TIME_DELTA_DAYS):
            logger.info(
                "task_heuristic_link.too_old memory_id=%s "
                "target_id=%s delta_days=%d run_id=%s",
                memory_id,
                target_id,
                delta.days,
                run_id,
            )
            return 0

        # Create the CAUSED_BY edge: bug_fix -> task_failed
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        edge = MemoryEdge(
            source_id=memory_id,
            target_id=target_id,
            edge_type="CAUSED_BY",
            weight=1.0,
            created_at=now_iso,
            last_seen_at=now_iso,
            created_by="task_heuristic_linker",
            run_id=run_id,
            # metadata is None — Neo4j 5 rejects Map-valued relationship
            # props. Attribution is captured by created_by + run_id.
            metadata=None,
        )

        created = await self._edge_service.create_edge(edge)
        edges_count = 1 if created else 0

        logger.info(
            "task_heuristic_link.completed memory_id=%s target_id=%s "
            "run_id=%s edges_created=%d",
            memory_id,
            target_id,
            run_id,
            edges_count,
        )
        return edges_count


__all__ = ["TaskHeuristicLinker", "MemoryFetcher"]
