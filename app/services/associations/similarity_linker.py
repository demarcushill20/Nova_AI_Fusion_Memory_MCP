"""Write-time similarity linker for PLAN-0759 Phase 2 (Sprint 6).

Overview
--------

This module lands :class:`SimilarityLinker` — the first write-time associative
linker component. When ``ASSOC_SIMILARITY_WRITE_ENABLED`` is True, the
``MemoryService.perform_upsert()`` hook calls
:meth:`SimilarityLinker.enqueue_link` after the memory is durably persisted to
both Pinecone and Neo4j. The linker fires off a fire-and-forget background
task that:

1. Queries Pinecone for the K nearest neighbors of the new memory's embedding
   (excluding self) with optional project scoping,
2. Filters candidates by :data:`SIMILARITY_THRESHOLD`,
3. Constructs a ``MemoryEdge`` per surviving candidate with ``edge_type =
   "SIMILAR_TO"``, and
4. Persists the batch via an injected :class:`MemoryEdgeService`
   (which handles MERGE + bidirectional canonicalization).

The component is fully injected: it does not construct a Pinecone client or
an edge service of its own, it does not reach into ``app.config.settings``,
and it holds no module-level state. All lifecycle concerns (driver, cache,
error handling) are externalized to the caller that constructs it.

Concurrency pattern — deviation from v2 plan
--------------------------------------------

The v2 PLAN-0759 Phase 2 Step 2.1 sketch describes a classic queue + worker
loop. Sprint 6 deliberately deviates to a **semaphore-bounded task-per-call**
pattern: every call to :meth:`enqueue_link` dispatches one new
``asyncio.create_task(_link_one_safe(...))`` and the number of in-flight
tasks is bounded by :data:`BACKGROUND_MAX_IN_FLIGHT` via an
``asyncio.Semaphore``. Rationale: a queue + worker loop would require
lifecycle management on :class:`MemoryService` (start/stop on init/shutdown),
which is a net new surface. A semaphore-bounded task is equivalently bounded,
carries no lifecycle state, and fails closed (drops requests + logs) if the
system becomes saturated. If profiling later shows the task-per-call
overhead is measurable, a bounded queue can be introduced behind the same
:meth:`enqueue_link` contract without touching any caller.

Exception containment
---------------------

:meth:`enqueue_link` is an ``async def`` that returns immediately after
scheduling the background task — the caller's ``perform_upsert()`` stack is
never exposed to a linker exception. The background task runs
:meth:`_link_one_safe`, which wraps :meth:`_link_one` in a ``try/except
Exception`` envelope. Every failure path logs a structured
``similarity_link.failed`` event and returns zero; none propagates. This is
the "fail-open" contract PLAN-0759 ADR §4 makes load-bearing: similarity
linking is best-effort and **must** not regress the durability of memory
writes.

Schema anchor
-------------

Edges are always ``(:base {entity_id})-[:SIMILAR_TO]->(:base {entity_id})``
per ADR-0759 §7. ``SIMILAR_TO`` is bidirectional (see
:data:`memory_edges.BIDIRECTIONAL_EDGE_TYPES`), so the edge service
canonicalizes the endpoint pair before writing. This linker does **not**
pre-canonicalize — it constructs the edge in ``(new_memory, neighbor)``
order and lets :class:`MemoryEdgeService` do the canonicalization.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from .memory_edges import MemoryEdge

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from .edge_service import MemoryEdgeService

logger = logging.getLogger(__name__)


class SimilarityLinker:
    """Fire-and-forget write-time similarity linker.

    See module docstring for the full design rationale. The hot path is
    :meth:`enqueue_link`; every other method is internal plumbing that
    tests call directly.
    """

    #: Minimum cosine score for an edge to be created. Operator decision
    #: A on 2026-04-13: ship 0.82 as the conservative default. No
    #: calibration work (v2 plan Step 2.5) is done in Sprint 6. Pinning
    #: tests in ``test_similarity_linker.py`` assert this exact value so a
    #: future refactor cannot silently tighten or loosen the threshold.
    SIMILARITY_THRESHOLD: float = 0.82

    #: Maximum edges created per new memory. Fan-out bound on the graph.
    MAX_NEIGHBORS: int = 10

    #: Number of Pinecone candidates to request. Slightly over-fetch so
    #: threshold filtering + self-exclusion still leave headroom to reach
    #: :data:`MAX_NEIGHBORS`.
    CANDIDATE_POOL: int = 30

    #: Per-call upper bound on total linker work, including Pinecone I/O
    #: and all Neo4j writes. If the work takes longer, the task is
    #: cancelled and a ``similarity_link.failed`` event is logged.
    BACKGROUND_TIMEOUT: float = 30.0

    #: Maximum number of simultaneously in-flight linker tasks. Past this
    #: we drop new requests (with a logged warning) rather than block the
    #: caller or unbounded-queue them. Bounded concurrency is enforced via
    #: an internal :class:`asyncio.Semaphore`.
    BACKGROUND_MAX_IN_FLIGHT: int = 32

    def __init__(
        self,
        pinecone_client: Any,
        edge_service: "MemoryEdgeService",
        *,
        cross_project_enabled: bool = False,
    ) -> None:
        """Store injected dependencies.

        Parameters
        ----------
        pinecone_client:
            An initialized Pinecone client with a synchronous
            ``query_vector(query_vector, top_k, filter, include_values)``
            method. The linker wraps calls in :func:`asyncio.to_thread`
            since Pinecone's client is sync.
        edge_service:
            A fully-constructed :class:`MemoryEdgeService` bound to an
            async Neo4j driver. The linker never creates its own driver —
            the caller (i.e. ``MemoryService``) is the single owner.
        cross_project_enabled:
            If True, the Pinecone query is not filtered by project (cross-
            project candidates are allowed). If False (default), candidate
            pool is scoped via a ``{"project": project}`` metadata filter;
            if ``project`` itself is ``None`` at call time, linking is
            skipped with a ``similarity_link.no_project`` warning — the
            linker never silently falls back to cross-project.
        """
        self._pinecone = pinecone_client
        self._edge_service = edge_service
        self._cross_project_enabled = cross_project_enabled
        self._semaphore = asyncio.Semaphore(self.BACKGROUND_MAX_IN_FLIGHT)
        # Keep strong references to in-flight tasks so the asyncio loop
        # does not garbage-collect them mid-flight (a classic
        # create_task footgun). The set is pruned in the done callback.
        self._inflight: set[asyncio.Task[int]] = set()

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #

    async def enqueue_link(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict,
        project: Optional[str],
    ) -> None:
        """Dispatch a background similarity-linking task and return.

        This method does **not** block the caller on any real work. Its
        only job is to:

        1. Verify the linker is not saturated (semaphore has capacity —
           otherwise log ``similarity_link.dropped_semaphore_full`` and
           return).
        2. If project scoping is required and ``project is None``, log
           ``similarity_link.no_project`` and return without scheduling.
        3. Otherwise ``asyncio.create_task(self._link_one_safe(...))``
           and return immediately.

        The caller's ``perform_upsert()`` return value is therefore never
        affected by anything downstream of this method. All structured
        logging happens inside the background task.
        """
        # Scoping guard: if the operator has disabled cross-project linking
        # and the memory has no project tag, we refuse to schedule. We do
        # NOT silently fall back to cross-project, because doing so would
        # surprise operators who turned the flag off precisely to avoid
        # cross-tenant leakage.
        if not self._cross_project_enabled and project is None:
            logger.warning(
                "similarity_link.no_project memory_id=%s "
                "cross_project_enabled=False project=None skipping=True",
                memory_id,
            )
            return

        # Saturation guard: a non-blocking check on the semaphore. If
        # BACKGROUND_MAX_IN_FLIGHT tasks are already running we drop the
        # new request rather than queue unboundedly. The caller still
        # returns immediately.
        if self._semaphore.locked():
            # ``locked()`` is True when the internal value is 0. The
            # below re-check under the ``_value`` internals guarantees
            # correctness against the public contract: if the semaphore
            # would block on acquire, we drop the work and log.
            if getattr(self._semaphore, "_value", 0) <= 0:
                logger.warning(
                    "similarity_link.dropped_semaphore_full "
                    "memory_id=%s max_in_flight=%d",
                    memory_id,
                    self.BACKGROUND_MAX_IN_FLIGHT,
                )
                return

        # Per-call run_id — used by MemoryEdgeService.delete_edges_by_run
        # for surgical rollback if this particular linker invocation
        # creates bad edges.
        run_id = f"wt-link-{uuid.uuid4().hex[:8]}"

        logger.info(
            "similarity_link.queued memory_id=%s run_id=%s "
            "cross_project_enabled=%s project=%s",
            memory_id,
            run_id,
            self._cross_project_enabled,
            project,
        )

        task = asyncio.create_task(
            self._link_one_safe(
                memory_id=memory_id,
                embedding=embedding,
                metadata=metadata,
                project=project,
                run_id=run_id,
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
        embedding: list[float],
        metadata: dict,
        project: Optional[str],
        run_id: str,
    ) -> int:
        """Exception-containment wrapper around :meth:`_link_one`.

        This is the function actually scheduled onto the event loop. Any
        exception thrown by ``_link_one`` (Pinecone down, Neo4j down,
        semaphore wait cancelled, timeout, etc.) is caught here, logged
        as ``similarity_link.failed``, and swallowed — the task returns
        ``0`` instead of raising. This is the contract that protects
        ``perform_upsert()``.
        """
        try:
            return await asyncio.wait_for(
                self._link_one(
                    memory_id=memory_id,
                    embedding=embedding,
                    metadata=metadata,
                    project=project,
                    run_id=run_id,
                ),
                timeout=self.BACKGROUND_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "similarity_link.failed memory_id=%s run_id=%s "
                "reason=timeout timeout_s=%s",
                memory_id,
                run_id,
                self.BACKGROUND_TIMEOUT,
            )
            return 0
        except Exception as exc:  # noqa: BLE001 — fail-open envelope
            logger.warning(
                "similarity_link.failed memory_id=%s run_id=%s "
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
        embedding: list[float],
        metadata: dict,
        project: Optional[str],
        run_id: str,
    ) -> int:
        """Do the actual similarity-linking work for one memory.

        Returns the number of ``SIMILAR_TO`` edges that were handed to
        :meth:`MemoryEdgeService.create_edges_batch` (i.e. the batch size,
        not the number the underlying MERGE reports as "created" — that
        distinction is deliberate: this method reports linker intent, the
        edge service reports database outcome).
        """
        t0 = time.perf_counter()

        async with self._semaphore:
            # Step 1: query Pinecone. The client is sync, so we wrap in
            # asyncio.to_thread. Over-fetch by 1 so that after
            # self-exclusion we still have CANDIDATE_POOL candidates to
            # threshold-filter.
            filter_dict: Optional[dict[str, Any]] = None
            if not self._cross_project_enabled:
                # project is guaranteed non-None by the enqueue_link
                # guard; assertion here is defence-in-depth.
                assert project is not None
                filter_dict = {"project": project}

            matches = await asyncio.to_thread(
                self._pinecone.query_vector,
                query_vector=embedding,
                top_k=self.CANDIDATE_POOL + 1,
                filter=filter_dict,
                include_values=False,
            )

            # Step 2: self-exclude. Pinecone may or may not return the
            # freshly-upserted vector in this query window (depends on
            # indexing lag), but if it does we must drop it.
            non_self: list[Any] = []
            for m in matches or []:
                mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
                if mid == memory_id:
                    continue
                non_self.append(m)

            # Step 3: threshold filter + sort by score DESC for
            # deterministic top-K selection.
            qualified: list[tuple[str, float, dict]] = []
            for m in non_self:
                if isinstance(m, dict):
                    score = float(m.get("score", 0.0))
                    mid = m.get("id", "")
                    mmeta = m.get("metadata") or {}
                else:
                    score = float(getattr(m, "score", 0.0))
                    mid = getattr(m, "id", "")
                    mmeta = getattr(m, "metadata", None) or {}
                if not mid:
                    continue
                if score < self.SIMILARITY_THRESHOLD:
                    continue
                qualified.append((mid, score, dict(mmeta)))

            if not qualified:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    "similarity_link.no_candidates memory_id=%s run_id=%s "
                    "pool=%d above_threshold=0 elapsed_ms=%.3f",
                    memory_id,
                    run_id,
                    len(non_self),
                    elapsed_ms,
                )
                return 0

            # Stable, deterministic ordering: score DESC then id ASC for
            # tie-break so two runs over the same candidate set always
            # pick the same top-K.
            qualified.sort(key=lambda t: (-t[1], t[0]))
            top = qualified[: self.MAX_NEIGHBORS]

            # Step 4: build MemoryEdge objects. Canonicalization for
            # bidirectional types (SIMILAR_TO is bidirectional) happens
            # inside MemoryEdgeService.create_edges_batch — we do NOT
            # pre-canonicalize here.
            now_iso = datetime.now(tz=timezone.utc).isoformat()
            edges: list[MemoryEdge] = []
            for neighbor_id, score, _neighbor_meta in top:
                edges.append(
                    MemoryEdge(
                        source_id=memory_id,
                        target_id=neighbor_id,
                        edge_type="SIMILAR_TO",
                        # Pinecone returns cosine similarity already in
                        # [0, 1] for normalized vectors. Clip defensively
                        # in case a floating-point drift pushes it
                        # fractionally above 1.0; MemoryEdge enforces
                        # [0.0, 1.0] in its __post_init__.
                        weight=min(1.0, max(0.0, float(score))),
                        created_at=now_iso,
                        last_seen_at=now_iso,
                        created_by="similarity_linker",
                        run_id=run_id,
                        # metadata=None: Sprint 4's edge_cypher template
                        # writes ``r.metadata = $metadata`` directly, and
                        # Neo4j refuses Map-valued properties. The linker's
                        # attribution is already fully captured in
                        # created_by + run_id, so the extra "source" tag
                        # the v2 plan sketch proposed is redundant and is
                        # dropped until Sprint 4's template is extended
                        # (future work) to either JSON-serialize the map
                        # or flatten keys onto the relationship.
                        metadata=None,
                    )
                )

            # Step 5: batch create via edge service. The executor does
            # the per-edge MERGE (no bulk UNWIND in Sprint 5); failures
            # of individual edges do not abort the batch — they just
            # reduce the ``created`` count returned.
            await self._edge_service.create_edges_batch(edges)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "similarity_link.completed memory_id=%s run_id=%s "
                "edges_created=%d pool=%d above_threshold=%d "
                "top_k=%d elapsed_ms=%.3f",
                memory_id,
                run_id,
                len(edges),
                len(non_self),
                len(qualified),
                len(top),
                elapsed_ms,
            )
            return len(edges)


__all__ = ["SimilarityLinker"]
