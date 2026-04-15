"""Async Neo4j executor for PLAN-0759 associative edges (Phase 1 / Sprint 5).

Overview
--------

This module lands the :class:`MemoryEdgeService` — the only component in
PLAN-0759's Phase 1 edge infrastructure that actually executes Cypher against
Neo4j. Sprint 4 put the dataclass (``memory_edges.MemoryEdge``) and the query
template builders (``edge_cypher``) in place; Sprint 5 wires a thin executor
on top so that Phase 2+ linker components (similarity, entity, temporal,
provenance, co-occurrence, task-heuristic) can call a single uniform API
instead of re-inventing session bookkeeping.

Design principles
~~~~~~~~~~~~~~~~~

1. **Injected driver, no module-level state.** The service takes an
   ``AsyncDriver`` in its constructor. It never creates a driver at import
   time, never reads ``app.config.settings``, and never touches any global.
   Production callers (Phase 2+) will pass in Fusion Memory's existing
   ``GraphClient.driver`` or a freshly constructed ``AsyncGraphDatabase``
   driver. Tests pass in their own driver pointed at the live Neo4j with
   ``auth=None``.

2. **All Cypher goes through ``edge_cypher`` where possible.** The
   :func:`edge_cypher.build_merge_edge_cypher`,
   :func:`edge_cypher.build_delete_edges_by_run_cypher`,
   :func:`edge_cypher.build_neighbors_cypher`, and
   :func:`edge_cypher.build_path_cypher` templates are the single source of
   truth for ``:base``-to-``:base`` scoping and edge-type whitelisting. The
   four admin methods (``delete_edges_by_tag``, ``count_edges_per_node``,
   ``get_edge_stats``, plus the single-edge ``delete_edges``) write their own
   inline Cypher because no template existed in Sprint 4 — **each of those
   inline strings still pins both endpoints to ``:base``** per ADR-0759 §7
   and the Phase 0 schema audit finding about ``INCLUDES`` (a pre-existing
   ``(:Session)-[:INCLUDES]->(:base)`` relationship type that would contaminate
   any blind ``MATCH ()-[r]-()`` traversal).

3. **Bidirectional edge canonicalization.** For
   ``SIMILAR_TO`` and ``CO_OCCURS`` (see
   :data:`memory_edges.BIDIRECTIONAL_EDGE_TYPES`), the service canonicalizes
   the ``(source_id, target_id)`` tuple via
   :meth:`memory_edges.MemoryEdge.canonicalize_for_bidirectional` before
   issuing the ``MERGE``. This prevents two linker runs that observe the
   same pair in opposite order from creating two parallel edges for the same
   logical fact and double-counting in traversal queries.

4. **Feature-flag agnosticism.** The service does **not** read
   ``app.config.settings.ASSOC_*`` flags. Phase 2's ``perform_upsert`` hook
   is responsible for the flag check. This keeps the service testable in
   isolation and keeps the ASSOC hot path in the hands of the orchestration
   layer.

5. **Fail-open contract.** Errors in service methods *do* raise — the
   service surfaces ``ValueError`` for invalid inputs and lets driver/session
   errors propagate. It is the **caller's** responsibility (Phase 2's
   ``perform_upsert`` hook) to wrap these calls in the fail-open envelope
   that never blocks ``store()`` on edge creation failure.

6. **No hook into production.** This module does not import
   ``memory_service``, ``memory_router``, ``graph_client``, or any other
   production path. It is a self-contained executor that Phase 2 will call.
   Sprint 5 lands the executor only; no wiring happens here.

Schema anchor
~~~~~~~~~~~~~

The canonical memory-node shape is ``(:base {entity_id: <str>})``. Verified
against ``graph_client.py:20`` (``NEO4J_NODE_LABEL = "base"``), the live
Neo4j 5.19 unique constraint, and the Phase 0 schema audit (825 ``:base``
nodes, 339 ``:Session`` nodes, 1 ``FOLLOWS`` edge, 517 pre-existing
``INCLUDES`` edges at audit time). The ``INCLUDES`` relationship is a
``(:Session)-[:INCLUDES]->(:base)`` edge used by the Phase 5 session system
— every query in this module must therefore structurally exclude ``:Session``
and must filter by :data:`memory_edges.VALID_EDGE_TYPES`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from neo4j import AsyncDriver

from .edge_cypher import (
    build_delete_edges_by_run_cypher,
    build_merge_edge_cypher,
    build_neighbors_cypher,
    build_path_cypher,
)
from .memory_edges import (
    BIDIRECTIONAL_EDGE_TYPES,
    VALID_EDGE_TYPES,
    MemoryEdge,
)

logger = logging.getLogger(__name__)


# Default database name for session binding. The production Fusion Memory
# container uses the default ``neo4j`` database; tests may override via
# the constructor.
_DEFAULT_DATABASE: str = "neo4j"


def _validate_edge_type(edge_type: str) -> None:
    """Reject any ``edge_type`` that is not one of the nine whitelisted names.

    Duplicates ``edge_cypher._validate_edge_type`` rather than importing a
    private symbol. Kept local so that every Cypher-interpolating path in
    this module routes through a visible choke-point.
    """
    if edge_type not in VALID_EDGE_TYPES:
        raise ValueError(
            f"Invalid edge type: {edge_type!r} (not in VALID_EDGE_TYPES)"
        )


class MemoryEdgeService:
    """Executor for PLAN-0759 associative edges against a ``:base`` graph.

    The service is a thin async wrapper over an injected ``AsyncDriver``.
    All Cypher templates are supplied either by :mod:`edge_cypher` (for
    CRUD + rollback + traversal) or inline in this file (for admin methods
    that did not have a Sprint 4 template). Inline templates are pinned to
    ``:base`` on both endpoints and parameterize every value.

    Public surface
    --------------

    Core CRUD:
        - :meth:`create_edge`
        - :meth:`create_edges_batch`
        - :meth:`get_neighbors`
        - :meth:`get_path`
        - :meth:`delete_edges`

    Admin / rollback:
        - :meth:`delete_edges_by_run`
        - :meth:`delete_edges_by_tag`
        - :meth:`count_edges_per_node`
        - :meth:`get_edge_stats`

    Lifecycle hooks:
        - :meth:`on_memory_delete`
        - :meth:`on_memory_supersede`
        - :meth:`on_memory_promote`
        - :meth:`on_memory_compact`

    Constructor
    -----------

    Parameters
    ----------
    driver:
        A neo4j ``AsyncDriver``. The service does not own it — it does not
        call ``close()`` — the caller that constructed the driver is
        responsible for shutting it down.
    database:
        The Neo4j database name. Defaults to ``"neo4j"``. Tests may override
        to target a non-default db.
    """

    def __init__(self, driver: AsyncDriver, database: str = _DEFAULT_DATABASE) -> None:
        self._driver = driver
        self._database = database

    # ------------------------------------------------------------------ #
    # Core CRUD                                                          #
    # ------------------------------------------------------------------ #

    async def create_edge(self, edge: MemoryEdge) -> bool:
        """``MERGE`` a single edge. Returns ``True`` on success, ``False`` if
        either endpoint does not exist (silent MATCH failure).

        For bidirectional edge types, the ``(source_id, target_id)`` pair is
        canonicalized via
        :meth:`memory_edges.MemoryEdge.canonicalize_for_bidirectional` before
        the ``MERGE`` so repeated linker runs over the same pair in opposite
        order produce a single stored relationship.
        """
        effective_edge = edge
        if edge.edge_type in BIDIRECTIONAL_EDGE_TYPES:
            canon_src, canon_dst = MemoryEdge.canonicalize_for_bidirectional(
                edge.source_id, edge.target_id
            )
            if (canon_src, canon_dst) != (edge.source_id, edge.target_id):
                effective_edge = MemoryEdge(
                    source_id=canon_src,
                    target_id=canon_dst,
                    edge_type=edge.edge_type,
                    weight=edge.weight,
                    created_at=edge.created_at,
                    last_seen_at=edge.last_seen_at,
                    created_by=edge.created_by,
                    run_id=edge.run_id,
                    edge_version=edge.edge_version,
                    metadata=edge.metadata,
                )

        query = build_merge_edge_cypher(effective_edge.edge_type)
        params = effective_edge.as_cypher_params()

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params)
            records = [rec async for rec in result]
            await result.consume()

        # MERGE returns r on the RETURN clause; if the initial MATCH failed
        # (either endpoint missing) the result set is empty and nothing was
        # created. Caller interprets False as "one or both nodes missing".
        return len(records) > 0

    async def create_edges_batch(self, edges: list[MemoryEdge]) -> int:
        """Create edges one-by-one in a single session.

        Sprint 5 prefers per-edge transactions (one ``session.run`` per edge)
        over a single bulk transaction: simpler error isolation, and the
        performance gap is negligible at Phase 1's bounded fanout (K=10
        neighbors per node). Phase 2+ can introduce a batched path if
        profiling demands it. Returns the number of edges that successfully
        landed (MERGE bound both endpoints).
        """
        if not edges:
            return 0

        created = 0
        async with self._driver.session(database=self._database) as session:
            for edge in edges:
                effective_edge = edge
                if edge.edge_type in BIDIRECTIONAL_EDGE_TYPES:
                    canon_src, canon_dst = MemoryEdge.canonicalize_for_bidirectional(
                        edge.source_id, edge.target_id
                    )
                    if (canon_src, canon_dst) != (edge.source_id, edge.target_id):
                        effective_edge = MemoryEdge(
                            source_id=canon_src,
                            target_id=canon_dst,
                            edge_type=edge.edge_type,
                            weight=edge.weight,
                            created_at=edge.created_at,
                            last_seen_at=edge.last_seen_at,
                            created_by=edge.created_by,
                            run_id=edge.run_id,
                            edge_version=edge.edge_version,
                            metadata=edge.metadata,
                        )

                query = build_merge_edge_cypher(effective_edge.edge_type)
                params = effective_edge.as_cypher_params()
                result = await session.run(query, params)
                records = [rec async for rec in result]
                await result.consume()
                if records:
                    created += 1

        return created

    async def get_neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
        min_weight: float = 0.0,
        limit: int = 20,
    ) -> list[dict]:
        """Return neighbors of ``node_id`` across the whitelisted edge types.

        Returns a list of dicts with keys ``{node_id, edge_type, weight,
        created_at, last_seen_at}``. Ordered by ``weight DESC`` then
        ``created_at DESC`` for deterministic ranking. ``limit`` is enforced
        client-side after the query so the Sprint 4 template — which does not
        bake in a LIMIT — remains unchanged.

        Known deferred gap: ``run_id`` and ``metadata`` are NOT projected by
        Sprint 4's ``build_neighbors_cypher`` template, and Sprint 5 was
        forbidden from modifying Sprint 4 files. If Phase 6's graph-traversal
        recall needs those fields (e.g. for rollback-aware filtering or
        intent-specific metadata), extend the template in a follow-up sprint
        and add the two keys here. Phase 2 similarity linking uses Pinecone,
        not this method, so this gap does not block Phase 2.
        """
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"node_id must be a non-empty string, got {node_id!r}")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError(f"limit must be a non-negative int, got {limit!r}")

        query = build_neighbors_cypher(
            edge_types=edge_types, direction=direction, min_weight=min_weight
        )

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                {"node_id": node_id, "min_weight": float(min_weight)},
            )
            rows: list[dict] = []
            async for record in result:
                rows.append(
                    {
                        "node_id": record["neighbor_id"],
                        "edge_type": record["edge_type"],
                        "weight": record["weight"],
                        "created_at": record["created_at"],
                        "last_seen_at": record["last_seen_at"],
                    }
                )
            await result.consume()

        # Deterministic in-memory ordering: primary weight DESC, secondary
        # created_at DESC (string compare on ISO-8601 is lexicographically
        # equivalent to chronological). The Sprint 4 template already sorts
        # by weight, but client-side resort guarantees the tie-break.
        # Negative weight inverts the primary key; ``_neg_iso`` inverts the
        # secondary key so a single ascending sort yields the DESC-then-DESC
        # ordering the contract requires.
        rows.sort(
            key=lambda r: (-float(r["weight"]), _neg_iso(r["created_at"])),
        )
        return rows[:limit] if limit > 0 else rows

    async def get_memory_neighbors_via_mentions(
        self,
        node_id: str,
        *,
        hub_threshold: int = 50,
        limit: int = 20,
    ) -> list[dict]:
        """Return memories sharing :Entity nodes with ``node_id`` via MENTIONS.

        Walks the 2-hop bridge ``(:base)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(:base)``.
        Required because :func:`build_neighbors_cypher` pins both endpoints
        to ``:base`` (Sprint 4 template constraint), so MENTIONS edges to
        ``:Entity`` nodes are invisible to :meth:`get_neighbors`. Without
        this method, intent="entity_recall" and "general" — both of which
        list MENTIONS in their edge filter — would silently return 0 graph
        neighbors via MENTIONS even when the graph is densely populated.

        Hub suppression mirrors Phase 7a's co-occurrence design: any
        ``:Entity`` mentioned by more than ``hub_threshold`` distinct
        memories is skipped during the walk. This prevents project-name
        entities (e.g. ``"nova-core"``, ``"novatrade"``) from collapsing
        the neighborhood to "every memory in the project". Tuned at 50 to
        match the v2 plan's Phase 7a hub cap.

        Returns dicts in the same shape as :meth:`get_neighbors` rows
        (``{node_id, edge_type, weight, created_at, last_seen_at}``) so
        the caller can merge results without branching on shape. The
        ``weight`` is derived from the count of shared non-hub entities,
        normalized to ``[0, 1]`` via ``min(1.0, shared / 5.0)`` — 5+
        shared entities saturates at 1.0. ``created_at`` and
        ``last_seen_at`` are returned as ``None`` because the relationship
        is synthetic (the underlying MENTIONS edges may have arbitrary
        timestamps; the synthetic edge has no single canonical time).
        """
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"node_id must be a non-empty string, got {node_id!r}")
        if not isinstance(hub_threshold, int) or hub_threshold < 1:
            raise ValueError(
                f"hub_threshold must be a positive int, got {hub_threshold!r}"
            )
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError(f"limit must be a non-negative int, got {limit!r}")
        if limit == 0:
            return []

        # Two-step: first collect non-hub entities reachable from the seed,
        # then walk back to other memories. Inlined as a single Cypher so
        # we make exactly one round-trip and the planner can choose its
        # own join order.
        query = (
            "MATCH (m:base {entity_id: $node_id})-[:MENTIONS]->(e:Entity)\n"
            "WITH m, e, "
            "  size([(other:base)-[:MENTIONS]->(e) | other]) AS deg\n"
            "WHERE deg <= $hub_threshold\n"
            "MATCH (e)<-[:MENTIONS]-(other:base)\n"
            "WHERE other.entity_id <> $node_id\n"
            "WITH other.entity_id AS neighbor_id, count(DISTINCT e) AS shared\n"
            "RETURN neighbor_id, shared\n"
            "ORDER BY shared DESC, neighbor_id ASC\n"
            "LIMIT $limit"
        )

        rows: list[dict] = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                {
                    "node_id": node_id,
                    "hub_threshold": int(hub_threshold),
                    "limit": int(limit),
                },
            )
            async for record in result:
                shared = int(record["shared"])
                weight = min(1.0, shared / 5.0)
                rows.append(
                    {
                        "node_id": record["neighbor_id"],
                        "edge_type": "MENTIONS",
                        "weight": weight,
                        "created_at": None,
                        "last_seen_at": None,
                        "shared_entities": shared,
                    }
                )
            await result.consume()
        return rows

    async def get_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 4,
        edge_types: list[str] | None = None,
    ) -> list[dict] | None:
        """Return the shortest path between two ``:base`` nodes.

        Returns ``None`` if no path exists within ``max_depth`` hops over the
        selected edge types. Otherwise returns a list of hop dicts of the
        same shape as :meth:`get_neighbors` rows, one per relationship on
        the path (N-1 dicts for an N-node path).
        """
        if not isinstance(start_id, str) or not start_id:
            raise ValueError(f"start_id must be a non-empty string, got {start_id!r}")
        if not isinstance(end_id, str) or not end_id:
            raise ValueError(f"end_id must be a non-empty string, got {end_id!r}")

        query = build_path_cypher(edge_types=edge_types, max_depth=max_depth)

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                {"start_id": start_id, "end_id": end_id},
            )
            records = [rec async for rec in result]
            await result.consume()

        if not records:
            return None

        path = records[0]["p"]
        if path is None:
            return None

        hops: list[dict] = []
        # neo4j Path iterates relationships in traversal order.
        for rel in path.relationships:
            start_node = rel.start_node
            end_node = rel.end_node
            hops.append(
                {
                    "source_id": start_node.get("entity_id"),
                    "target_id": end_node.get("entity_id"),
                    "edge_type": rel.type,
                    "weight": rel.get("weight"),
                    "created_at": rel.get("created_at"),
                    "last_seen_at": rel.get("last_seen_at"),
                }
            )
        return hops

    async def delete_edges(
        self,
        source_id: str,
        target_id: str | None = None,
        edge_type: str | None = None,
    ) -> int:
        """Delete edges leaving ``source_id``, optionally filtered by target
        and/or edge_type. Returns the number deleted.

        Both endpoints are scoped to ``:base`` so that a bad filter cannot
        collaterally touch ``:Session`` chains or any pre-existing
        ``INCLUDES`` edges. When ``edge_type`` is given it is validated
        against the whitelist *before* being interpolated into the Cypher
        string.
        """
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(
                f"source_id must be a non-empty string, got {source_id!r}"
            )
        if target_id is not None and (
            not isinstance(target_id, str) or not target_id
        ):
            raise ValueError(
                f"target_id must be None or a non-empty string, got {target_id!r}"
            )

        if edge_type is not None:
            _validate_edge_type(edge_type)
            rel_clause = f"-[r:{edge_type}]->"
        else:
            types_pattern = "|".join(sorted(VALID_EDGE_TYPES))
            rel_clause = f"-[r:{types_pattern}]->"

        where_clauses: list[str] = []
        if target_id is not None:
            where_clauses.append("b.entity_id = $target_id")
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses) + "\n"

        query = (
            f"MATCH (a:base {{entity_id: $source_id}})"
            f"{rel_clause}"
            f"(b:base)\n"
            f"{where_sql}"
            f"DELETE r\n"
            f"RETURN count(r) AS deleted"
        )

        params: dict[str, Any] = {"source_id": source_id}
        if target_id is not None:
            params["target_id"] = target_id

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params)
            record = await result.single()
            await result.consume()
            return int(record["deleted"]) if record else 0

    # ------------------------------------------------------------------ #
    # Admin / rollback                                                   #
    # ------------------------------------------------------------------ #

    async def delete_edges_by_run(self, run_id: str) -> int:
        """Thin wrapper over :func:`edge_cypher.build_delete_edges_by_run_cypher`.

        Scoped to ``:base``-to-``:base`` edges by the Sprint 4 template.
        Returns the number deleted.
        """
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"run_id must be a non-empty string, got {run_id!r}")

        query = build_delete_edges_by_run_cypher()
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, {"run_id": run_id})
            record = await result.single()
            await result.consume()
            return int(record["deleted"]) if record else 0

    async def delete_edges_by_tag(
        self,
        created_by: str,
        edge_type: str | None = None,
        created_after: str | None = None,
    ) -> int:
        """Delete edges by ``created_by`` attribution, optionally filtered.

        Inline Cypher — no Sprint 4 template exists. Both endpoints are
        pinned to ``:base`` and ``edge_type`` is whitelist-validated before
        interpolation. ``created_after`` is compared as a string (callers
        pass ISO-8601 timestamps, which sort lexicographically).
        """
        if not isinstance(created_by, str) or not created_by:
            raise ValueError(
                f"created_by must be a non-empty string, got {created_by!r}"
            )
        if edge_type is not None:
            _validate_edge_type(edge_type)
            rel_clause = f"-[r:{edge_type}]->"
        else:
            types_pattern = "|".join(sorted(VALID_EDGE_TYPES))
            rel_clause = f"-[r:{types_pattern}]->"

        where_clauses = ["r.created_by = $created_by"]
        params: dict[str, Any] = {"created_by": created_by}
        if created_after is not None:
            if not isinstance(created_after, str) or not created_after:
                raise ValueError(
                    "created_after must be None or a non-empty ISO-8601 string, "
                    f"got {created_after!r}"
                )
            where_clauses.append("r.created_at > $created_after")
            params["created_after"] = created_after

        where_sql = "WHERE " + " AND ".join(where_clauses) + "\n"
        query = (
            f"MATCH (a:base)"
            f"{rel_clause}"
            f"(b:base)\n"
            f"{where_sql}"
            f"DELETE r\n"
            f"RETURN count(r) AS deleted"
        )

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params)
            record = await result.single()
            await result.consume()
            return int(record["deleted"]) if record else 0

    async def count_edges_per_node(self, node_id: str) -> dict[str, int]:
        """Return ``{edge_type: count}`` for edges incident on ``node_id``.

        Counts edges in both directions. Only edge types in
        :data:`memory_edges.VALID_EDGE_TYPES` are counted — any pre-existing
        ``INCLUDES`` or ``FOLLOWS`` relationships are structurally excluded
        by the ``:base``-on-both-ends constraint and the relationship-type
        filter.
        """
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"node_id must be a non-empty string, got {node_id!r}")

        types_pattern = "|".join(sorted(VALID_EDGE_TYPES))
        query = (
            f"MATCH (a:base {{entity_id: $node_id}})"
            f"-[r:{types_pattern}]-"
            f"(b:base)\n"
            f"RETURN type(r) AS edge_type, count(r) AS c"
        )

        counts: dict[str, int] = {}
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, {"node_id": node_id})
            async for record in result:
                counts[record["edge_type"]] = int(record["c"])
            await result.consume()
        return counts

    async def get_edge_stats(self) -> dict[str, dict[str, Any]]:
        """Return aggregate stats for all nine whitelisted edge types.

        Shape:

        .. code-block:: python

           {
               "SIMILAR_TO": {"count": N, "avg_weight": f, "min_weight": f, "max_weight": f},
               "SUPERSEDES": {"count": 0, "avg_weight": None, "min_weight": None, "max_weight": None},
               ...
           }

        A type with zero matching edges returns ``count=0`` and ``None`` for
        the three weight aggregates. Both endpoints pinned to ``:base``.
        """
        stats: dict[str, dict[str, Any]] = {
            et: {
                "count": 0,
                "avg_weight": None,
                "min_weight": None,
                "max_weight": None,
            }
            for et in VALID_EDGE_TYPES
        }

        types_pattern = "|".join(sorted(VALID_EDGE_TYPES))
        query = (
            f"MATCH (a:base)-[r:{types_pattern}]->(b:base)\n"
            f"RETURN type(r) AS edge_type, "
            f"count(r) AS c, "
            f"avg(r.weight) AS avg_w, "
            f"min(r.weight) AS min_w, "
            f"max(r.weight) AS max_w"
        )

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query)
            async for record in result:
                et = record["edge_type"]
                stats[et] = {
                    "count": int(record["c"]),
                    "avg_weight": record["avg_w"],
                    "min_weight": record["min_w"],
                    "max_weight": record["max_w"],
                }
            await result.consume()
        return stats

    # ------------------------------------------------------------------ #
    # Lifecycle hooks                                                    #
    # ------------------------------------------------------------------ #

    async def on_memory_delete(self, node_id: str) -> dict[str, Any]:
        """Remove all edges touching ``node_id`` in both directions.

        Returns ``{"edges_removed": N, "orphaned_entities": []}``. The
        orphaned-entities list is always empty in Sprint 5 — entity linking
        is Phase 3 work, not Phase 1. The key behavior here is that deleting
        a memory must not leave dangling edges in the graph.

        The parameter is named ``node_id`` rather than the plan's illustrative
        placeholder name, to align with the ``(:base {entity_id})`` schema
        anchor and the naming that ADR-0759 §7 deliberately adopted — see
        the module docstring for full rationale.
        """
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"node_id must be a non-empty string, got {node_id!r}")

        types_pattern = "|".join(sorted(VALID_EDGE_TYPES))
        query = (
            f"MATCH (a:base {{entity_id: $node_id}})"
            f"-[r:{types_pattern}]-"
            f"(b:base)\n"
            f"DELETE r\n"
            f"RETURN count(r) AS deleted"
        )

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, {"node_id": node_id})
            record = await result.single()
            await result.consume()
            removed = int(record["deleted"]) if record else 0
        return {"edges_removed": removed, "orphaned_entities": []}

    async def on_memory_supersede(
        self,
        new_id: str,
        old_id: str,
        *,
        reason: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Record a ``SUPERSEDES`` edge from ``new_id`` → ``old_id``.

        Supersession is a **soft** operation per the v2 plan's Phase 5a: the
        old node is not deleted, its other edges are not removed, and its
        content is not rewritten. The only thing that happens is a directed
        ``SUPERSEDES`` edge from the new memory to the old memory, stamped
        with attribution metadata so rollback can unwind individual
        supersession events later.

        Both ``new_id`` and ``old_id`` must refer to existing ``:base`` nodes;
        if either is missing the underlying ``MERGE`` silently fails (no edge
        created) and the method returns ``None`` without raising. Callers
        that need a hard failure guarantee should follow up with
        :meth:`count_edges_per_node` or their own existence check.

        Parameters
        ----------
        new_id:
            ``entity_id`` of the newer memory that supersedes ``old_id``.
        old_id:
            ``entity_id`` of the older memory being superseded.
        reason:
            Optional human-readable reason for the supersession. When
            provided, stored in ``edge.metadata["reason"]``.
        run_id:
            Caller-supplied run identifier. Defaults to
            ``"supersession_hook"`` when not provided.
        """
        if not isinstance(new_id, str) or not new_id:
            raise ValueError(f"new_id must be a non-empty string, got {new_id!r}")
        if not isinstance(old_id, str) or not old_id:
            raise ValueError(f"old_id must be a non-empty string, got {old_id!r}")
        if new_id == old_id:
            return None

        from datetime import datetime, timezone

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        edge_metadata: dict | None = {"reason": reason} if reason else None
        edge = MemoryEdge(
            source_id=new_id,
            target_id=old_id,
            edge_type="SUPERSEDES",
            weight=1.0,
            created_at=now_iso,
            last_seen_at=now_iso,
            created_by="edge_service.on_memory_supersede",
            run_id="supersession_hook" if run_id is None else run_id,
            metadata=edge_metadata,
        )
        await self.create_edge(edge)
        return None

    async def on_memory_promote(
        self,
        new_id: str,
        old_id: str,
        *,
        from_layer: Optional[str] = None,
        to_layer: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Record a ``PROMOTED_FROM`` edge from ``new_id`` → ``old_id``.

        Promotion is a **soft** operation per the v2 plan's Phase 5b: when a
        memory is consolidated from a lower layer (e.g. ``scratch``) into a
        higher layer (e.g. ``core``), the higher-layer memory gets a directed
        ``PROMOTED_FROM`` edge pointing back at its lower-layer source. The
        old node is not deleted and its other edges are preserved.

        Both ``new_id`` and ``old_id`` must refer to existing ``:base`` nodes;
        if either is missing the underlying ``MERGE`` silently fails (no edge
        created) and the method returns ``None`` without raising.

        If ``new_id == old_id``, returns ``None`` without creating an edge.

        Parameters
        ----------
        new_id:
            ``entity_id`` of the promoted (higher-layer) memory.
        old_id:
            ``entity_id`` of the original (lower-layer) source memory.
        from_layer:
            Optional name of the source layer (e.g. ``"scratch"``). When
            provided together with ``to_layer``, stored in
            ``edge.metadata["from_layer"]`` and ``edge.metadata["to_layer"]``.
        to_layer:
            Optional name of the destination layer (e.g. ``"core"``).
        run_id:
            Caller-supplied run identifier. Defaults to
            ``"promotion_hook"`` when not provided.
        """
        if not isinstance(new_id, str) or not new_id:
            raise ValueError(f"new_id must be a non-empty string, got {new_id!r}")
        if not isinstance(old_id, str) or not old_id:
            raise ValueError(f"old_id must be a non-empty string, got {old_id!r}")
        if new_id == old_id:
            return None

        from datetime import datetime, timezone

        now_iso = datetime.now(tz=timezone.utc).isoformat()

        edge_metadata: dict | None = None
        if from_layer is not None or to_layer is not None:
            if from_layer is None or to_layer is None:
                raise ValueError(
                    f"from_layer and to_layer must both be provided or both omitted, "
                    f"got from_layer={from_layer!r}, to_layer={to_layer!r}"
                )
            edge_metadata = {
                "from_layer": from_layer,
                "to_layer": to_layer,
            }

        edge = MemoryEdge(
            source_id=new_id,
            target_id=old_id,
            edge_type="PROMOTED_FROM",
            weight=1.0,
            created_at=now_iso,
            last_seen_at=now_iso,
            created_by="edge_service.on_memory_promote",
            run_id="promotion_hook" if run_id is None else run_id,
            metadata=edge_metadata,
        )
        await self.create_edge(edge)
        return None

    async def on_memory_compact(
        self,
        summary_id: str,
        source_ids: list[str],
        *,
        run_id: str | None = None,
    ) -> int:
        """Record ``COMPACTED_FROM`` edges from ``summary_id`` to each source.

        When N memories are compacted (rolled up) into a single summary
        memory, this method records the provenance chain so that recall
        and audit code can trace back to the original source memories.

        The method uses **per-item** ``try/except`` so that a single
        failing edge (e.g. a missing source node) does not prevent the
        remaining edges from being written. This is the same fail-open
        pattern established by Phase 5a's supersession hook.

        Parameters
        ----------
        summary_id:
            ``entity_id`` of the compacted summary ``:base`` node.
        source_ids:
            ``entity_id`` values of the original source ``:base`` nodes
            that were rolled into the summary.
        run_id:
            Optional caller-supplied run identifier. When ``None``,
            defaults to ``"compaction_hook"``.

        Returns
        -------
        int
            The number of ``COMPACTED_FROM`` edges successfully created.
        """
        if not isinstance(summary_id, str) or not summary_id:
            raise ValueError(
                f"summary_id must be a non-empty string, got {summary_id!r}"
            )
        if isinstance(source_ids, str):
            raise TypeError(
                "source_ids must be a list of strings, not a bare string"
            )

        from datetime import datetime, timezone

        effective_run_id = run_id if run_id is not None else "compaction_hook"
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        created = 0

        for source_id in source_ids:
            try:
                # Self-loop guard: skip if the summary is also in the
                # source list (degenerate edge case).
                if source_id == summary_id:
                    logger.debug(
                        "Skipping self-loop COMPACTED_FROM edge: "
                        "summary_id == source_id == %s",
                        summary_id,
                    )
                    continue

                edge = MemoryEdge(
                    source_id=summary_id,
                    target_id=source_id,
                    edge_type="COMPACTED_FROM",
                    weight=1.0,
                    created_at=now_iso,
                    last_seen_at=now_iso,
                    created_by="edge_service.on_memory_compact",
                    run_id=effective_run_id,
                )
                if await self.create_edge(edge):
                    created += 1
            except Exception as e:
                logger.warning(
                    "COMPACTED_FROM edge %s->%s failed (non-fatal): %s",
                    summary_id,
                    source_id,
                    e,
                )

        return created

    # ------------------------------------------------------------------ #
    # Provenance read API                                                #
    # ------------------------------------------------------------------ #

    #: The three edge types that encode memory provenance (new → old).
    _PROVENANCE_EDGE_TYPES: tuple[str, ...] = (
        "SUPERSEDES",
        "PROMOTED_FROM",
        "COMPACTED_FROM",
    )
    assert all(
        t in VALID_EDGE_TYPES for t in _PROVENANCE_EDGE_TYPES
    ), "_PROVENANCE_EDGE_TYPES must be a subset of VALID_EDGE_TYPES"

    async def get_provenance(
        self,
        memory_id: str,
        max_depth: int = 10,
    ) -> dict:
        """Walk ``SUPERSEDES`` / ``PROMOTED_FROM`` / ``COMPACTED_FROM`` edges
        **outward** from ``memory_id`` to find its original episodic sources.

        These three edge types point from newer memory → older memory, so
        following them in the outgoing direction walks backward through the
        provenance chain to the original sources.

        Parameters
        ----------
        memory_id:
            ``entity_id`` of the starting ``:base`` node.
        max_depth:
            Maximum number of hops to traverse. Clamped to ``[1, 10]``.

        Returns
        -------
        dict
            ``{
                "memory_id": str,
                "provenance_chain": [
                    {"memory_id": str, "edge_type": str, "depth": int,
                     "metadata": dict | None},
                    ...
                ],
                "original_sources": [str],
                "depth": int,
                "max_depth": int,
                "depth_limited": bool,
            }``

            ``provenance_chain`` lists every ancestor discovered, ordered by
            depth (shallowest first). ``original_sources`` are the leaf nodes
            that have no further outgoing provenance edges **within the
            traversal** — when ``depth_limited`` is ``True``, frontier nodes
            may appear as leaves even if they have further provenance beyond
            ``max_depth``; in cycles, Neo4j's single-path no-revisit
            guarantee may also produce spurious leaves.
            ``depth`` is the maximum depth actually reached.
        """
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError(
                f"memory_id must be a non-empty string, got {memory_id!r}"
            )
        memory_id = memory_id.strip()
        if not isinstance(max_depth, int) or isinstance(max_depth, bool):
            raise ValueError(f"max_depth must be an int, got {max_depth!r}")
        if max_depth < 1:
            max_depth = 1
        if max_depth > 10:
            max_depth = 10

        # Variable-length path query over the three provenance edge types.
        # Neo4j's variable-length patterns do not revisit nodes within a
        # single path, providing inherent cycle safety.  We collect ALL
        # paths so fan-out (compaction) is captured.
        prov_pattern = "|".join(self._PROVENANCE_EDGE_TYPES)
        query = (
            f"MATCH path = (start:base {{entity_id: $memory_id}})"
            f"-[:{prov_pattern}*1..{max_depth}]->"
            f"(ancestor:base)\n"
            f"RETURN [node IN nodes(path) | node.entity_id] AS node_ids,\n"
            f"       [rel IN relationships(path) | type(rel)] AS edge_types,\n"
            f"       [rel IN relationships(path) | rel.metadata] AS metadatas,\n"
            f"       length(path) AS depth\n"
            f"ORDER BY depth"
        )

        # Execute and collect all paths.
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, {"memory_id": memory_id})
            rows: list[dict] = []
            async for record in result:
                rows.append(
                    {
                        "node_ids": list(record["node_ids"]),
                        "edge_types": list(record["edge_types"]),
                        "metadatas": list(record["metadatas"]),
                        "depth": int(record["depth"]),
                    }
                )
            await result.consume()

        # Build deduplicated provenance chain and identify leaf nodes.
        # Each row is a path from start to some ancestor; the paths share
        # prefixes (Neo4j returns all distinct paths).
        seen: set[str] = set()
        chain: list[dict] = []
        all_ancestors: set[str] = set()
        non_leaf: set[str] = set()
        max_depth_reached = 0

        for row in rows:
            node_ids = row["node_ids"]
            edge_types = row["edge_types"]
            metadatas = row["metadatas"]
            depth = row["depth"]

            if depth > max_depth_reached:
                max_depth_reached = depth

            if len(node_ids) != len(edge_types) + 1 or len(metadatas) != len(edge_types):
                continue

            # Rows are ORDER BY depth (shortest paths first), so the first
            # time we encounter an ancestor_id gives its minimum distance
            # from the start node.  i+1 == hop count in this path.
            for i, edge_type in enumerate(edge_types):
                ancestor_id = node_ids[i + 1]
                all_ancestors.add(ancestor_id)

                # Mark the source of this edge as non-leaf (it has an
                # outgoing provenance edge).
                non_leaf.add(node_ids[i])

                if ancestor_id not in seen:
                    seen.add(ancestor_id)
                    chain.append(
                        {
                            "memory_id": ancestor_id,
                            "edge_type": edge_type,
                            "depth": i + 1,
                            "metadata": MemoryEdge.deserialize_metadata(metadatas[i]),
                        }
                    )

        # Original sources: ancestors that are NOT the source of any
        # outgoing provenance edge within the traversal (i.e., they are
        # leaf nodes in the provenance graph).  Note: in cycles, Neo4j's
        # single-path no-revisit guarantee may produce spurious leaves;
        # when depth_limited is True, frontier nodes may also appear as
        # leaves even if they have further provenance beyond max_depth.
        original_sources = sorted(all_ancestors - non_leaf)

        return {
            "memory_id": memory_id,
            "provenance_chain": chain,
            "original_sources": original_sources,
            "depth": max_depth_reached,
            "max_depth": max_depth,
            "depth_limited": max_depth_reached >= max_depth,
        }


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _neg_iso(ts: Optional[str]) -> str:
    """Return a key that sorts in reverse chronological order when ascending.

    Neo4j stores ``created_at`` as an ISO-8601 string. ``None`` values are
    pushed to the end of the list by returning the empty string (which sorts
    before any populated ISO-8601 timestamp under reverse order). Used as a
    secondary sort key in :meth:`MemoryEdgeService.get_neighbors` to achieve
    weight-DESC then created_at-DESC ordering without two separate sorts.
    """
    if not ts:
        return ""
    # Invert every character so ascending sort becomes descending sort. Works
    # because ISO-8601 timestamps are pure ASCII and monotonic.
    return "".join(chr(255 - ord(c)) for c in ts)


__all__ = ["MemoryEdgeService"]
