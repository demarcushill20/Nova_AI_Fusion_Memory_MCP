"""Write-time entity linker for PLAN-0759 Phase 3 (Sprint 9).

Overview
--------

This module lands :class:`EntityLinker` — the second write-time associative
linker in PLAN-0759. When ``ASSOC_ENTITY_WRITE_ENABLED`` is True, the hook
inside :meth:`MemoryService.perform_upsert` calls
:meth:`EntityLinker.enqueue_link` after the memory is durably persisted. The
linker fires off a fire-and-forget background task that:

1. Resolves entity candidates via a two-tier contract:

   - **Tier A** — if the caller supplied ``metadata["entities"]`` as a
     non-empty ``list[str]``, use that verbatim (still canonicalized
     before write).
   - **Tier B** — otherwise run the pure-Python heuristic extractor from
     :mod:`app.services.associations.entity_extractor` on the memory's
     ``content`` string. No LLM-based NER is involved in Phase 3.

2. Canonicalizes every candidate via
   :func:`entity_extractor.canon_entity` and de-duplicates by canonical
   form — raw ``["Neo4j", "neo4j", "NEO4J"]`` collapses to one entry.

3. Within a single Neo4j session, for each surviving canonical name,
   ``MERGE``s a ``(:Entity {project, name})`` node and a directed
   ``(:base {entity_id: memory_id})-[:MENTIONS]->(:Entity {project, name})``
   edge. The node sets ``created_at`` on create; the edge sets
   ``weight``, ``created_at``, ``last_seen_at``, ``created_by``,
   ``run_id``, ``edge_version``, and ``metadata=None`` on create and
   refreshes only ``last_seen_at`` on match.

Dispatch is fire-and-forget via ``asyncio.create_task``; exception
containment uses a two-layer envelope (``_link_one_safe`` wraps
``_link_one`` in ``asyncio.wait_for`` + catch-all, and the caller's
:meth:`perform_upsert` hook adds an outer ``try/except`` for lazy-import
and construction failures).

Concurrency pattern
-------------------

Mirrors Sprint 6's :class:`SimilarityLinker`: a semaphore-bounded
task-per-call (``BACKGROUND_MAX_IN_FLIGHT = 32``) rather than a
queue+worker loop. Rationale is identical — no lifecycle state on
:class:`MemoryService`, fails closed (drops + logs) when saturated, and
can be swapped for a bounded queue later without touching any caller.

Cypher — deviation from Sprint 4 template
-----------------------------------------

``build_merge_edge_cypher`` in :mod:`edge_cypher` pins **both** endpoints
to ``:base``. For ``MENTIONS`` edges one endpoint is ``:base`` (the
memory) and the other is ``:Entity`` (the named entity), so the Sprint 4
template cannot be reused as-is. Sprint 9 is forbidden from modifying
Sprint 4 files (per guardrail).

Therefore this module writes **its own inline Cypher** for the two-step
entity upsert / edge upsert flow, following the same discipline as
Sprint 5's inline admin queries in :class:`MemoryEdgeService`:

- Both endpoints labeled explicitly (``:base`` on the memory side,
  ``:Entity`` on the entity side).
- All values parameterized via ``$param`` placeholders — no f-string
  interpolation of any user-controlled value.
- The ``(project, name)`` pair is the :class:`Entity` node's primary key
  and is enforced via ``MERGE``. Per ADR-0759 §7 every ``:Entity`` node
  **must** carry a ``project`` property; nodes without one are not
  created (the ``project is None`` path is a skip+log).
- Edge ``metadata=None`` is enforced inline because Neo4j 5 rejects
  Map-valued relationship properties (Sprint 6 finding, still valid).
- The entity-node upsert, the memory-match, and the edge upsert are
  issued as a **single combined** Cypher per canonical name so that
  there is exactly one Neo4j round-trip per entity within the session's
  transaction window.

Schema anchors
--------------

- Memory node identity: ``(:base {entity_id: <str>})`` — verified in
  ``graph_client.py:20`` (``NEO4J_NODE_LABEL = "base"``) and the unique
  constraint at line 97, per ADR-0759 §7.
- Entity node identity: ``(:Entity {project: <str>, name: <str>})``
  where ``name = canon_entity(raw)``. ``created_at`` is set on create
  only. No ``mention_count`` property — the reverse-lookup helpers
  derive mention count from the ``:MENTIONS`` degree at query time.
- Edge type: ``MENTIONS`` is directed and is listed in
  :data:`memory_edges.VALID_EDGE_TYPES`. It is **not** bidirectional —
  no canonicalization is applied to the endpoint order.
- ``MAX_ENTITIES_PER_MEMORY = 20``, inherited from
  :data:`entity_extractor.MAX_ENTITIES_PER_MEMORY` and re-enforced by
  the Tier A branch so caller-provided lists are also truncated.

Skip conditions
---------------

- ``project is None`` → log ``entity_link.no_project`` and return 0.
  Sprint 9 does not support a cross-project entity namespace because the
  ``(project, name)`` primary key is load-bearing for multi-tenant
  safety.
- No entities resolved → log ``entity_link.no_entities`` and return 0
  without opening any Neo4j session.

Exception containment
---------------------

:meth:`_link_one_safe` wraps :meth:`_link_one` in ``asyncio.wait_for``
with a :data:`BACKGROUND_TIMEOUT` deadline. Any exception (timeout,
Neo4j down, bad canonicalization, bad Cypher) is caught, logged as a
structured ``entity_link.failed`` event, and swallowed — the task
returns ``0`` instead of raising. This is the contract that protects
``perform_upsert()``.

Module invariants (verified by tests)
-------------------------------------

- Does **not** import ``app.config.settings``. The feature flag check
  lives in the :meth:`MemoryService.perform_upsert` hook, not in the
  linker itself — the linker is flag-agnostic.
- Does **not** create a Neo4j driver at import or construct time. The
  ``AsyncDriver`` is injected through the constructor.
- Does **not** construct a :class:`MemoryEdgeService` because the
  ``MENTIONS`` edge has mixed ``:base``/``:Entity`` endpoints which
  Sprint 4's template cannot express; all Cypher is inline here.
- Does **not** pass Map-valued metadata onto the relationship. Every
  write sets ``r.metadata = $metadata`` to the literal value ``None``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from app.observability.metrics import record_entity_mention

from .entity_extractor import MAX_ENTITIES_PER_MEMORY, canon_entity, extract_entities
from .memory_edges import EDGE_VERSION

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


# Default Neo4j database name — matches Sprint 5's ``MemoryEdgeService``
# default. Tests may override via the constructor keyword.
_DEFAULT_DATABASE: str = "neo4j"


# Combined Cypher: upsert the Entity node AND upsert the MENTIONS edge in
# one transaction-local statement. The memory-side MATCH is an actual
# MATCH (not a MERGE) because the memory ``:base`` node was persisted by
# _persist_memory_item() before the hook ran; if it is somehow missing
# (race, deletion), the MATCH fails silently and the MERGE body never
# runs, which is exactly the fail-open behavior we want.
#
# Both endpoints are labeled explicitly. Every value is a ``$param``.
_MENTIONS_UPSERT_CYPHER: str = (
    "MATCH (m:base {entity_id: $memory_id})\n"
    "MERGE (e:Entity {project: $project, name: $canon_name})\n"
    "  ON CREATE SET e.created_at = $now\n"
    "MERGE (m)-[r:MENTIONS]->(e)\n"
    "  ON CREATE SET\n"
    "    r.weight = $weight,\n"
    "    r.created_at = $now,\n"
    "    r.last_seen_at = $now,\n"
    "    r.created_by = $created_by,\n"
    "    r.run_id = $run_id,\n"
    "    r.edge_version = $edge_version,\n"
    "    r.metadata = $metadata\n"
    "  ON MATCH SET\n"
    "    r.last_seen_at = $now\n"
    "RETURN e.name AS name"
)


# Read-only query: memories that mention a specific (project, name)
# Entity. Pinned to ``:base`` on the memory side and ``:Entity`` on the
# entity side so it cannot contaminate with ``:Session`` or any other
# pre-existing labels.
_MEMORIES_FOR_ENTITY_CYPHER: str = (
    "MATCH (m:base)-[r:MENTIONS]->(e:Entity {project: $project, name: $canon_name})\n"
    "RETURN m.entity_id AS memory_id, "
    "r.created_at AS created_at, r.last_seen_at AS last_seen_at\n"
    "ORDER BY r.last_seen_at DESC\n"
    "LIMIT $limit"
)


# Read-only query: entities mentioned by a given memory, with mention
# count derived from :MENTIONS degree. The degree subquery counts all
# incoming :MENTIONS edges to the Entity regardless of source memory,
# which is the right semantics for "how often has this entity been
# referenced across the whole graph".
_ENTITIES_FOR_MEMORY_CYPHER: str = (
    "MATCH (m:base {entity_id: $memory_id})-[:MENTIONS]->(e:Entity)\n"
    "WITH e, size([(other:base)-[:MENTIONS]->(e) | other]) AS mention_count\n"
    "RETURN e.project AS project, e.name AS name, "
    "mention_count AS mention_count\n"
    "ORDER BY mention_count DESC, e.name ASC"
)


class EntityLinker:
    """Fire-and-forget write-time entity linker.

    See module docstring for the full design rationale. The hot path is
    :meth:`enqueue_link`; every other public method is either an
    internal plumbing hook (``_link_one_safe``, ``_link_one``) or a
    read-only lookup (``get_memories_for_entity``,
    ``get_entities_for_memory``).
    """

    #: Per-call upper bound on total linker work including all Neo4j
    #: writes. If exceeded, the task is cancelled and a
    #: ``entity_link.failed`` event is logged.
    BACKGROUND_TIMEOUT: float = 30.0

    #: Maximum number of simultaneously in-flight linker tasks. Past
    #: this point new requests are dropped (with a logged warning)
    #: rather than unbounded-queued. Matches Sprint 6's value.
    BACKGROUND_MAX_IN_FLIGHT: int = 32

    #: Hard cap on entities surfaced per memory. Re-exported from
    #: :data:`entity_extractor.MAX_ENTITIES_PER_MEMORY` so that a future
    #: plan change only has to touch one module. The Tier A branch
    #: (caller-provided ``metadata["entities"]``) also enforces this.
    MAX_ENTITIES_PER_MEMORY: int = MAX_ENTITIES_PER_MEMORY

    def __init__(
        self,
        driver: "AsyncDriver",
        *,
        database: str = _DEFAULT_DATABASE,
    ) -> None:
        """Store the injected Neo4j driver.

        Parameters
        ----------
        driver:
            A fully-constructed ``neo4j.AsyncDriver``. The linker never
            creates a driver — the caller (i.e. ``MemoryService``) is
            the single owner and is responsible for shutdown.
        database:
            The Neo4j database name. Defaults to ``"neo4j"``; tests may
            override to target a non-default db.
        """
        self._driver = driver
        self._database = database
        self._semaphore = asyncio.Semaphore(self.BACKGROUND_MAX_IN_FLIGHT)
        # Strong references to in-flight tasks so the asyncio loop does
        # not GC them mid-flight (classic create_task footgun). The set
        # is pruned in the done-callback.
        self._inflight: set[asyncio.Task[int]] = set()

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #

    async def enqueue_link(
        self,
        memory_id: str,
        content: str,
        metadata: dict,
        project: Optional[str],
    ) -> None:
        """Dispatch a background entity-linking task and return.

        This method does **not** block the caller on any real work. It:

        1. Resolves entities via Tier A (caller-provided) or Tier B
           (heuristic extractor on ``content``).
        2. If ``project is None`` → log ``entity_link.no_project`` and
           return without scheduling.
        3. If the resolved entity list is empty → log
           ``entity_link.no_entities`` and return without scheduling.
        4. Otherwise ``asyncio.create_task(self._link_one_safe(...))``
           and return immediately.

        The caller's ``perform_upsert()`` return value is never affected
        by anything downstream of this method — all structured logging
        happens either at this dispatch point or inside the background
        task.
        """
        # Scoping guard: Sprint 9 does not support a cross-project
        # entity namespace because ``(project, name)`` is the Entity
        # node primary key. Falling back to "global" would silently
        # collide entities across tenants. Skip + log instead.
        if project is None:
            logger.warning(
                "entity_link.no_project memory_id=%s project=None skipping=True",
                memory_id,
            )
            return

        # Resolve entities BEFORE scheduling so the background task has
        # a simple, already-validated list to operate on. This also
        # means the no_entities and too_many checks run synchronously
        # on the caller's stack (microseconds on mocks, still << 1 ms
        # on real content because the extractor is pure-Python regex).
        entities = self._resolve_entities(content=content, metadata=metadata)

        if not entities:
            logger.info(
                "entity_link.no_entities memory_id=%s project=%s skipping=True",
                memory_id,
                project,
            )
            return

        # Saturation guard: a non-blocking check on the semaphore. If
        # BACKGROUND_MAX_IN_FLIGHT tasks are already running we drop
        # the new request rather than queue unboundedly. The caller
        # still returns immediately.
        if self._semaphore.locked():
            if getattr(self._semaphore, "_value", 0) <= 0:
                logger.warning(
                    "entity_link.dropped_semaphore_full "
                    "memory_id=%s max_in_flight=%d",
                    memory_id,
                    self.BACKGROUND_MAX_IN_FLIGHT,
                )
                return

        # Per-call run_id — used by ``scripts/assoc_rollback`` (which
        # deletes by ``r.run_id`` regardless of type) for surgical
        # rollback if this particular linker invocation creates bad
        # edges. The ``wt-entity-`` prefix distinguishes entity-linker
        # runs from Sprint 6's ``wt-link-`` similarity runs.
        run_id = f"wt-entity-{uuid.uuid4().hex[:8]}"

        logger.info(
            "entity_link.queued memory_id=%s run_id=%s project=%s entities=%d",
            memory_id,
            run_id,
            project,
            len(entities),
        )

        task = asyncio.create_task(
            self._link_one_safe(
                memory_id=memory_id,
                entities=entities,
                project=project,
                run_id=run_id,
            )
        )
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    # ------------------------------------------------------------------ #
    # Entity resolution (Tier A / Tier B)                                #
    # ------------------------------------------------------------------ #

    def _resolve_entities(self, *, content: str, metadata: dict) -> list[str]:
        """Return the raw entity list for this memory.

        Tier A takes precedence: if ``metadata["entities"]`` is a
        non-empty ``list``, every non-empty string in it is returned
        verbatim (truncated to :data:`MAX_ENTITIES_PER_MEMORY`).
        Non-string values are skipped defensively.

        Otherwise Tier B runs :func:`extract_entities` on ``content``,
        which is already bounded to :data:`MAX_ENTITIES_PER_MEMORY`
        internally and enforces the 100 KB content cap.

        The returned list is the **raw** surface forms; canonicalization
        runs later in :meth:`_link_one` as a final dedup pass.
        """
        raw_meta = metadata.get("entities") if isinstance(metadata, dict) else None
        if isinstance(raw_meta, list) and raw_meta:
            out: list[str] = []
            for item in raw_meta:
                if not isinstance(item, str):
                    continue
                stripped = item.strip()
                if not stripped:
                    continue
                out.append(stripped)
                if len(out) >= self.MAX_ENTITIES_PER_MEMORY:
                    break
            if out:
                return out
            # If the Tier A list was pathological (all non-strings,
            # all empty), fall through to Tier B rather than silently
            # returning [] — the caller presumably wanted SOME entities.

        if not isinstance(content, str) or not content:
            return []

        try:
            return extract_entities(content, max_entities=self.MAX_ENTITIES_PER_MEMORY)
        except Exception as exc:  # noqa: BLE001 — fail-open envelope
            logger.warning(
                "entity_link.extractor_failed error=%s content_len=%d",
                exc,
                len(content),
            )
            return []

    # ------------------------------------------------------------------ #
    # Internal plumbing                                                  #
    # ------------------------------------------------------------------ #

    async def _link_one_safe(
        self,
        memory_id: str,
        entities: list[str],
        project: str,
        run_id: str,
    ) -> int:
        """Exception-containment wrapper around :meth:`_link_one`.

        This is the function actually scheduled onto the event loop.
        Any exception thrown by ``_link_one`` (Neo4j down, cancelled,
        timeout, bad canonicalization, ...) is caught here, logged as
        ``entity_link.failed``, and swallowed — the task returns ``0``
        instead of raising. This is the contract that protects
        ``perform_upsert()``.
        """
        try:
            return await asyncio.wait_for(
                self._link_one(
                    memory_id=memory_id,
                    entities=entities,
                    project=project,
                    run_id=run_id,
                ),
                timeout=self.BACKGROUND_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "entity_link.failed memory_id=%s run_id=%s "
                "reason=timeout timeout_s=%s",
                memory_id,
                run_id,
                self.BACKGROUND_TIMEOUT,
            )
            return 0
        except Exception as exc:  # noqa: BLE001 — fail-open envelope
            logger.warning(
                "entity_link.failed memory_id=%s run_id=%s reason=%s error=%s",
                memory_id,
                run_id,
                type(exc).__name__,
                exc,
            )
            return 0

    async def _link_one(
        self,
        memory_id: str,
        entities: list[str],
        project: str,
        run_id: str,
    ) -> int:
        """Do the actual entity-linking work for one memory.

        Canonicalizes every raw entity, dedups by canonical form, and
        issues one combined Entity-node + MENTIONS-edge upsert per
        survivor inside a single Neo4j session. Returns the number of
        ``MENTIONS`` upserts **issued** (not the number the underlying
        MERGE reports as "created" — this is the same reporting
        discipline Sprint 6 uses).
        """
        t0 = time.perf_counter()

        async with self._semaphore:
            # Step 1: canonicalize and dedup. We preserve the FIRST
            # occurrence of each canonical name so repeated runs over
            # the same list produce a deterministic write order.
            canon_seen: dict[str, str] = {}
            for raw in entities:
                try:
                    canon = canon_entity(raw)
                except (ValueError, TypeError):
                    # Empty / whitespace-only / non-string input. Skip
                    # silently — upstream Tier A/B has already filtered
                    # most bad cases, but canon_entity is the canonical
                    # validator and we trust it as the last word.
                    continue
                if canon not in canon_seen:
                    canon_seen[canon] = raw

            canonical_names = list(canon_seen.keys())

            if not canonical_names:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    "entity_link.no_entities memory_id=%s run_id=%s "
                    "reason=all_candidates_invalid elapsed_ms=%.3f",
                    memory_id,
                    run_id,
                    elapsed_ms,
                )
                return 0

            # Truncate defensively in case Tier A delivered more than
            # MAX_ENTITIES_PER_MEMORY distinct canonicals. The Tier A
            # branch in _resolve_entities already caps raw input but
            # canonicalization can collapse or expand the count
            # slightly; re-enforce the ceiling here.
            if len(canonical_names) > self.MAX_ENTITIES_PER_MEMORY:
                canonical_names = canonical_names[: self.MAX_ENTITIES_PER_MEMORY]

            now_iso = datetime.now(tz=timezone.utc).isoformat()

            # Step 2: open a single session and issue one combined
            # upsert per canonical name. We do NOT use explicit
            # transactions — Sprint 5 established that per-edge
            # auto-commit is fine at Phase 1 bounded fanout and the
            # per-memory entity count is bounded to 20. The edge count
            # returned is the number of MERGE statements that bound
            # the ``MATCH (m:base ...)`` — if the memory node is
            # missing for some reason, the RETURN list is empty and we
            # record zero for that entity (fail-open).
            edges_created = 0
            async with self._driver.session(database=self._database) as session:
                for canon_name in canonical_names:
                    params: dict[str, Any] = {
                        "memory_id": memory_id,
                        "project": project,
                        "canon_name": canon_name,
                        "now": now_iso,
                        "weight": 1.0,
                        "created_by": "entity_linker",
                        "run_id": run_id,
                        "edge_version": EDGE_VERSION,
                        # metadata is LITERALLY None, not an empty dict.
                        # Neo4j 5 refuses Map-valued relationship props.
                        "metadata": None,
                    }
                    result = await session.run(_MENTIONS_UPSERT_CYPHER, params)
                    records = [rec async for rec in result]
                    await result.consume()
                    if records:
                        edges_created += 1
                        record_entity_mention()

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "entity_link.completed memory_id=%s run_id=%s project=%s "
                "canonical=%d edges_created=%d elapsed_ms=%.3f",
                memory_id,
                run_id,
                project,
                len(canonical_names),
                edges_created,
                elapsed_ms,
            )
            return edges_created

    # ------------------------------------------------------------------ #
    # Read-only lookups                                                  #
    # ------------------------------------------------------------------ #

    async def get_memories_for_entity(
        self,
        project: str,
        entity_name: str,
        limit: int = 20,
    ) -> list[dict]:
        """Return memories that ``MENTIONS`` a given entity.

        Read-only — issues no writes. The ``entity_name`` is passed
        through :func:`canon_entity` before the lookup so callers can
        use any surface form.

        Parameters
        ----------
        project:
            The project namespace of the entity. Non-empty string.
        entity_name:
            Raw or canonical entity name. Will be canonicalized.
        limit:
            Maximum rows to return. Non-negative; capped at 1000 as a
            safety guard against pathological callers.

        Returns
        -------
        list[dict]
            List of dicts ordered by ``last_seen_at`` DESC, with keys
            ``{memory_id, created_at, last_seen_at}``. ``created_at``
            and ``last_seen_at`` are taken from the MENTIONS edge (not
            from the memory node).
        """
        if not isinstance(project, str) or not project:
            raise ValueError(f"project must be a non-empty string, got {project!r}")
        if not isinstance(entity_name, str) or not entity_name:
            raise ValueError(
                f"entity_name must be a non-empty string, got {entity_name!r}"
            )
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError(f"limit must be a non-negative int, got {limit!r}")
        if limit == 0:
            return []
        if limit > 1000:
            limit = 1000

        canon_name = canon_entity(entity_name)

        rows: list[dict] = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _MEMORIES_FOR_ENTITY_CYPHER,
                {
                    "project": project,
                    "canon_name": canon_name,
                    "limit": limit,
                },
            )
            async for record in result:
                rows.append(
                    {
                        "memory_id": record["memory_id"],
                        "created_at": record["created_at"],
                        "last_seen_at": record["last_seen_at"],
                    }
                )
            await result.consume()
        return rows

    async def get_entities_for_memory(self, memory_id: str) -> list[dict]:
        """Return entities mentioned by a given memory.

        Read-only. ``mention_count`` is derived from the ``:MENTIONS``
        degree of each Entity node at query time — there is no stored
        ``mention_count`` property on the node.

        Parameters
        ----------
        memory_id:
            Non-empty ``entity_id`` of the source ``:base`` memory.

        Returns
        -------
        list[dict]
            List of dicts ordered by ``mention_count`` DESC then
            ``name`` ASC, with keys ``{project, name, mention_count}``.
            Empty list if the memory has no outgoing MENTIONS edges.
        """
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError(
                f"memory_id must be a non-empty string, got {memory_id!r}"
            )

        rows: list[dict] = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _ENTITIES_FOR_MEMORY_CYPHER, {"memory_id": memory_id}
            )
            async for record in result:
                rows.append(
                    {
                        "project": record["project"],
                        "name": record["name"],
                        "mention_count": int(record["mention_count"]),
                    }
                )
            await result.consume()
        return rows


__all__ = ["EntityLinker"]
