"""Write-time co-occurrence edge linker for PLAN-0759 Phase 7a.

Overview
--------

This module lands :class:`CooccurrenceLinker` — a write-time linker that
creates ``CO_OCCURS`` edges between memories sharing >=2 non-hub entities.
It runs after entity linking has created ``MENTIONS`` edges.

When ``ASSOC_COOCCURRENCE_WRITE_ENABLED`` is True, the hook inside
:meth:`MemoryService.perform_upsert` calls
:meth:`CooccurrenceLinker.enqueue_link` after the memory is durably
persisted. The linker fires off a fire-and-forget background task that:

1. Fetches entities for the newly stored memory via
   :meth:`EntityLinker.get_entities_for_memory`.
2. For each entity, fetches co-mentioning memories via
   :meth:`EntityLinker.get_memories_for_entity` — skipping hub entities
   (those mentioned by more than :data:`HUB_THRESHOLD` memories).
3. Counts shared entities between this memory and each co-mentioner.
4. Creates ``CO_OCCURS`` edges for pairs sharing >=
   :data:`MIN_SHARED_ENTITIES` entities, weighted by an IDF-style score.

Hub suppression
---------------

Entities mentioned by more than :data:`HUB_THRESHOLD` (50) memories are
skipped entirely. These ultra-common entities (e.g. a project-level tag)
add noise rather than signal to co-occurrence linking.

IDF-style weighting
-------------------

For each shared entity, we compute ``idf = log(total_memories /
entity_degree)`` where ``entity_degree`` is the number of memories that
mention that entity (approximated by ``mention_count`` from the entity
query or by the length of the memories-for-entity result). The edge
weight is the sum of IDF scores across all shared entities, normalized
to [0, 1] via ``min(1.0, idf_sum / 10.0)``.

Concurrency model
-----------------

Mirrors the semaphore-bounded task-per-call pattern established by
SimilarityLinker (Sprint 6) and EntityLinker (Sprint 9):

- Global semaphore (``BACKGROUND_MAX_IN_FLIGHT = 32``) bounds total
  in-flight tasks.
- ``asyncio.wait_for`` enforces a ``BACKGROUND_TIMEOUT`` of 30 seconds.
- Saturation drops with a logged warning, fail-open.

Per-entity degree cap
---------------------

To prevent a single prolific entity from generating excessive edges,
the linker limits evaluation to at most :data:`MAX_EDGES_PER_ENTITY`
(30) co-mentioning memories per entity.

Exception containment
---------------------

:meth:`_link_one_safe` wraps :meth:`_link_one` in
``asyncio.wait_for`` + catch-all. Per-edge writes use individual
try/except blocks so one failed edge does not abort the rest. This
is the fail-open contract that protects ``perform_upsert()``.

Module invariants
-----------------

- Does **not** import ``app.config.settings``. The feature flag check
  lives in the ``perform_upsert()`` hook.
- Does **not** create a Neo4j driver at import or construct time.
- Accepts injected ``EntityLinker`` and ``MemoryEdgeService`` instances.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from .memory_edges import EDGE_VERSION, MemoryEdge

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from .edge_service import MemoryEdgeService
    from .entity_linker import EntityLinker

logger = logging.getLogger(__name__)


class CooccurrenceLinker:
    """Fire-and-forget write-time co-occurrence linker.

    Creates ``CO_OCCURS`` edges between memories that share >=2 non-hub
    entities. See module docstring for the full design rationale.
    """

    #: Entities mentioned by more than this many memories are skipped.
    HUB_THRESHOLD: int = 50

    #: Maximum co-mentioning memories evaluated per entity.
    MAX_EDGES_PER_ENTITY: int = 30

    #: Minimum shared entities required to create an edge.
    MIN_SHARED_ENTITIES: int = 2

    #: Per-call timeout for the background task.
    BACKGROUND_TIMEOUT: float = 30.0

    #: Maximum number of simultaneously in-flight linker tasks.
    BACKGROUND_MAX_IN_FLIGHT: int = 32

    def __init__(
        self,
        edge_service: "MemoryEdgeService",
        entity_linker: "EntityLinker",
    ) -> None:
        """Store injected dependencies.

        Parameters
        ----------
        edge_service:
            A fully-constructed ``MemoryEdgeService`` for writing
            ``CO_OCCURS`` edges. The linker does not own it.
        entity_linker:
            A fully-constructed ``EntityLinker`` whose read-only lookup
            methods (``get_entities_for_memory``,
            ``get_memories_for_entity``) are used to discover
            co-occurring memory pairs.
        """
        self._edge_service = edge_service
        self._entity_linker = entity_linker
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
        project: str | None,
        *,
        run_id: str | None = None,
    ) -> None:
        """Dispatch a background co-occurrence linking task and return.

        This method does **not** block the caller on any real work.

        Parameters
        ----------
        memory_id:
            The ``entity_id`` of the newly stored memory.
        project:
            The project namespace. Required (``None`` ⇒ skip).
        run_id:
            Optional run identifier for rollback. Auto-generated if
            not provided.
        """
        if not project:
            logger.warning(
                "cooccurrence_link.no_project memory_id=%s skipping=True",
                memory_id,
            )
            return

        # Saturation guard
        if self._semaphore.locked():
            if getattr(self._semaphore, "_value", 0) <= 0:
                logger.warning(
                    "cooccurrence_link.dropped_semaphore_full "
                    "memory_id=%s max_in_flight=%d",
                    memory_id,
                    self.BACKGROUND_MAX_IN_FLIGHT,
                )
                return

        run_id = run_id or f"wt-cooccur-{uuid.uuid4().hex[:8]}"

        logger.info(
            "cooccurrence_link.queued memory_id=%s run_id=%s project=%s",
            memory_id,
            run_id,
            project,
        )

        task = asyncio.create_task(
            self._link_one_safe(
                memory_id=memory_id,
                project=project,
                run_id=run_id,
            )
        )
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    # ------------------------------------------------------------------ #
    # Internal plumbing                                                  #
    # ------------------------------------------------------------------ #

    async def _link_one_safe(
        self,
        memory_id: str,
        project: str,
        run_id: str,
    ) -> int:
        """Exception-containment wrapper around :meth:`_link_one`."""
        try:
            async with self._semaphore:
                return await asyncio.wait_for(
                    self._link_one(
                        memory_id=memory_id,
                        project=project,
                        run_id=run_id,
                    ),
                    timeout=self.BACKGROUND_TIMEOUT,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "cooccurrence_link.failed memory_id=%s run_id=%s "
                "reason=timeout timeout_s=%s",
                memory_id,
                run_id,
                self.BACKGROUND_TIMEOUT,
            )
            return 0
        except Exception as exc:  # noqa: BLE001 — fail-open envelope
            logger.warning(
                "cooccurrence_link.failed memory_id=%s run_id=%s "
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
        project: str,
        run_id: str,
    ) -> int:
        """Core co-occurrence logic.

        1. Get entities for this memory.
        2. For each non-hub entity, get co-mentioning memories.
        3. Count shared entities between this memory and each co-mentioner.
        4. Create CO_OCCURS edges for pairs sharing >= MIN_SHARED_ENTITIES,
           weighted by IDF-style score.

        Returns the number of edges created.
        """
        t0 = time.perf_counter()

        # Step 1: Get entities for this memory
        entities = await self._entity_linker.get_entities_for_memory(
            memory_id
        )
        if not entities:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "cooccurrence_link.no_entities memory_id=%s run_id=%s "
                "elapsed_ms=%.3f",
                memory_id,
                run_id,
                elapsed_ms,
            )
            return 0

        # Step 2: For each entity, get co-mentioning memories.
        # Track: {other_memory_id: [list of shared entity names]}
        # Also track entity degree for IDF computation.
        comentions: dict[str, list[str]] = {}
        entity_degrees: dict[str, int] = {}

        for entity in entities:
            if not isinstance(entity, dict):
                logger.debug("cooccurrence_link.skip_non_dict_entity entity=%r", entity)
                continue

            entity_name = entity.get("name")
            if entity_name is None or entity_name == "":
                continue

            mention_count = int(entity.get("mention_count") or 0)

            # Hub suppression: skip if entity is too popular.
            if mention_count > self.HUB_THRESHOLD:
                logger.debug(
                    "cooccurrence_link.hub_skip entity=%s "
                    "mention_count=%d threshold=%d",
                    entity_name,
                    mention_count,
                    self.HUB_THRESHOLD,
                )
                continue

            # Get memories mentioning this entity (capped)
            try:
                memories = await self._entity_linker.get_memories_for_entity(
                    project=project,
                    entity_name=entity_name,
                    limit=self.HUB_THRESHOLD + 1,
                )
            except Exception as exc:  # noqa: BLE001 — per-entity fail-open
                logger.warning(
                    "cooccurrence_link.entity_fetch_failed entity=%s: %s",
                    entity_name,
                    exc,
                )
                continue

            # Secondary hub suppression: if the actual degree exceeds
            # HUB_THRESHOLD, skip this entity entirely.
            if len(memories) > self.HUB_THRESHOLD:
                logger.debug(
                    "cooccurrence_link.hub_skip_secondary entity=%s "
                    "actual_degree=%d threshold=%d",
                    entity_name,
                    len(memories),
                    self.HUB_THRESHOLD,
                )
                continue

            # Cap to MAX_EDGES_PER_ENTITY for evaluation
            memories = memories[: self.MAX_EDGES_PER_ENTITY]

            # Record entity degree for IDF
            entity_degrees[entity_name] = max(len(memories), 1)

            for mem in memories:
                mid = mem.get("memory_id") if isinstance(mem, dict) else None
                if mid is None or mid == "" or mid == memory_id:
                    continue
                if mid not in comentions:
                    comentions[mid] = []
                comentions[mid].append(entity_name)

        # Step 3+4: Filter to pairs sharing >= MIN_SHARED_ENTITIES,
        # compute IDF weight, create edges.
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        edges_created = 0

        # max_degree is the largest entity degree observed. Used as a
        # local IDF denominator — not a true corpus count, but a
        # conservative approximation that underestimates IDF for rare
        # entities (safe: errs toward lower weights, not higher).
        max_degree = max(
            (d for d in entity_degrees.values()), default=1
        )
        max_degree = max(max_degree, 1)

        for other_id, shared_entities in comentions.items():
            if len(shared_entities) < self.MIN_SHARED_ENTITIES:
                continue

            # IDF-style weight: sum of log(max_degree / degree) for each
            # shared entity, normalized to [0, 1].
            idf_sum = 0.0
            for ent_name in shared_entities:
                degree = entity_degrees.get(ent_name, 1)
                idf_sum += math.log(max(max_degree / degree, 1.0))

            # Normalize to [0, 1]
            weight = min(1.0, idf_sum / 10.0)
            # Ensure minimum positive weight for any qualifying pair
            weight = max(weight, 0.01)

            if not math.isfinite(weight):
                weight = 0.01

            try:
                edge = MemoryEdge(
                    source_id=memory_id,
                    target_id=other_id,
                    edge_type="CO_OCCURS",
                    weight=round(weight, 4),
                    created_at=now_iso,
                    last_seen_at=now_iso,
                    created_by="cooccurrence_linker",
                    run_id=run_id,
                    edge_version=EDGE_VERSION,
                    metadata=None,
                )
                created = await self._edge_service.create_edge(edge)
                if created:
                    edges_created += 1
            except Exception as exc:  # noqa: BLE001 — per-edge fail-open
                logger.warning(
                    "cooccurrence_link.edge_failed %s->%s: %s",
                    memory_id,
                    other_id,
                    exc,
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "cooccurrence_link.completed memory_id=%s run_id=%s "
            "project=%s entities=%d candidates=%d edges_created=%d "
            "elapsed_ms=%.3f",
            memory_id,
            run_id,
            project,
            len(entities),
            len(comentions),
            edges_created,
            elapsed_ms,
        )
        return edges_created


__all__ = ["CooccurrenceLinker"]
