"""Cypher query templates for PLAN-0759 associative linking.

This module is pure string construction. It does **not** execute any Cypher,
does **not** instantiate a driver, and imports nothing from
``app.services.graph_client``. The executor that actually runs these templates
lives in the (future) ``edge_service`` module, landed in Sprint 5.

Safety contract
---------------

1. **Whitelisted type dispatch, everything else parameterized.** Neo4j Cypher
   cannot parameterize relationship-type names — you cannot write
   ``MERGE (a)-[r:$type]->(b)``. The only way to select a relationship type at
   query-construction time is string interpolation. So these builders
   interpolate ``edge_type`` as an f-string, **but only after** validating it
   against :data:`memory_edges.VALID_EDGE_TYPES`. Any input outside that
   frozenset (including the classic injection payload
   ``"SIMILAR_TO; MATCH (n) DETACH DELETE n //"``) is rejected with
   ``ValueError`` before any string building happens. Every other value
   (``weight``, ``created_at``, ``metadata``, ``$src_id``, ``$dst_id``, ...)
   flows through as a ``$param`` placeholder and is bound by the driver.

2. **Traversal is always edge-type scoped.** The live Fusion Memory Neo4j
   instance contains a pre-existing relationship type ``INCLUDES``
   (``(:Session)-[:INCLUDES]->(:base)``, 517 edges at the time of the Phase 0
   audit) that is **not** a PLAN-0759 edge type. A blind
   ``MATCH (m:base)-[r]-()`` traversal would walk into ``Session`` territory
   and return junk. Every recall / neighbors / path builder in this module
   filters by at least one name in :data:`memory_edges.VALID_EDGE_TYPES`, and
   both endpoints are pinned to the ``:base`` label so ``:Session`` nodes are
   structurally unreachable from these queries.

3. **Rollback is scoped to ``:base``-to-``:base`` edges only.** The delete
   template refuses to match edges whose endpoints are not both ``:base``, so
   a bad ``run_id`` sweep cannot collaterally delete the live Session chain
   or any future subsystem's edges that happen to share a property name.

Schema anchor
-------------

The canonical memory-node shape is ``(:base {entity_id: <str>})``. Verified
on 2026-04-13 against ``app/services/graph_client.py`` line 20
(``NEO4J_NODE_LABEL = "base"``) and line 97 (the unique constraint on
``:base.entity_id``), and against the live Neo4j 5.19 instance via
``scripts/audit_neo4j_schema.py``. The v2 plan's illustrative placeholder
label/key snippets are not directly executable and must be rewritten to
the verified schema before any Phase 1 Cypher is merged — see ADR-0759 §7.
The regression test in ``tests/test_edge_cypher.py`` enforces that no
placeholder substrings leak into any builder output.
"""

from __future__ import annotations

from .memory_edges import VALID_EDGE_TYPES


# Private constants that pin the node schema. These are used everywhere the
# label or primary key appears in a Cypher template so that any future schema
# rename is a one-line change. See module docstring for the verification cite
# (graph_client.py:20 and scripts/audit_neo4j_schema.py).
_NODE_LABEL: str = "base"
_NODE_KEY_PROPERTY: str = "entity_id"

# Valid directions for neighbor / traversal queries. ``out`` follows outgoing
# edges, ``in`` follows incoming, ``both`` ignores direction.
_VALID_DIRECTIONS: frozenset[str] = frozenset({"out", "in", "both"})

# Upper bound on shortest-path depth. 10 is generous relative to the
# bounded-fanout design (K=10 neighbors per node in Phase 2), and keeps any
# query from accidentally devolving into a full-graph walk.
_MAX_PATH_DEPTH: int = 10


def _validate_edge_type(edge_type: str) -> None:
    """Raise ``ValueError`` if ``edge_type`` is not in the whitelist.

    This is the single choke-point that every builder in the module routes
    through before it interpolates ``edge_type`` into an f-string. If it
    passes, the value is known to be one of the nine constant strings in
    :data:`memory_edges.VALID_EDGE_TYPES` and is safe to splice into Cypher.
    """
    if edge_type not in VALID_EDGE_TYPES:
        raise ValueError(
            f"Invalid edge type: {edge_type!r} (not in VALID_EDGE_TYPES)"
        )


def _validate_edge_type_list(edge_types: list[str] | None) -> list[str]:
    """Validate every name in ``edge_types`` or expand ``None`` to the full set.

    Returns a concrete ``list[str]`` of validated names in a deterministic
    order (sorted) so the resulting Cypher string is stable across calls —
    stable strings make caching, snapshot testing, and grep-based audits easy.
    """
    if edge_types is None:
        return sorted(VALID_EDGE_TYPES)
    if not edge_types:
        # Explicitly empty list is caller error, not "all types".
        raise ValueError(
            "edge_types must be None (meaning all valid types) or a non-empty list"
        )
    for et in edge_types:
        _validate_edge_type(et)
    # Preserve caller order for explicit lists — this matters for tests that
    # assert on specific substring ordering.
    return list(edge_types)


def build_merge_edge_cypher(edge_type: str) -> str:
    """Build a ``MERGE`` Cypher template for a single edge type.

    The returned string is safe to execute with any parameter dict produced
    by :meth:`memory_edges.MemoryEdge.as_cypher_params`. The template:

    - Matches both endpoints by ``(:base {entity_id})`` so that rows for
      non-``:base`` nodes (e.g. future ``:Session`` nodes, or anything that
      accidentally shares an id) cannot be touched.
    - Uses ``MERGE`` so that repeated linker runs over the same
      ``(source, target, edge_type)`` triple are idempotent.
    - On create, stamps all bookkeeping fields.
    - On match, takes the pointwise-max of the existing and new weight
      (stronger evidence wins) and refreshes ``last_seen_at``.

    Only ``edge_type`` is interpolated, and only after whitelist validation.
    Every other value is bound via the ``$param`` placeholders.
    """
    _validate_edge_type(edge_type)
    return (
        f"MATCH (a:{_NODE_LABEL} {{{_NODE_KEY_PROPERTY}: $src_id}}), "
        f"(b:{_NODE_LABEL} {{{_NODE_KEY_PROPERTY}: $dst_id}})\n"
        f"MERGE (a)-[r:{edge_type}]->(b)\n"
        f"ON CREATE SET\n"
        f"  r.weight = $weight,\n"
        f"  r.created_at = $created_at,\n"
        f"  r.last_seen_at = $last_seen_at,\n"
        f"  r.created_by = $created_by,\n"
        f"  r.run_id = $run_id,\n"
        f"  r.edge_version = $edge_version,\n"
        f"  r.metadata = $metadata\n"
        f"ON MATCH SET\n"
        f"  r.weight = CASE WHEN r.weight < $weight THEN $weight ELSE r.weight END,\n"
        f"  r.last_seen_at = $last_seen_at\n"
        f"RETURN r"
    )


def build_delete_edges_by_run_cypher() -> str:
    """Build a Cypher template that deletes all edges tagged with a ``run_id``.

    Scoped to ``:base``-to-``:base`` edges so that a bad linker run can be
    unwound without any risk of deleting edges that belong to other
    subsystems (e.g. the live ``(:Session)-[:FOLLOWS]->(:Session)`` chain or
    the pre-existing ``(:Session)-[:INCLUDES]->(:base)`` edges). Takes a
    single ``$run_id`` parameter and returns the delete count so the caller
    can assert the expected number of edges vanished.
    """
    return (
        f"MATCH (a:{_NODE_LABEL})-[r]->(b:{_NODE_LABEL})\n"
        f"WHERE r.run_id = $run_id\n"
        f"DELETE r\n"
        f"RETURN count(r) AS deleted"
    )


def build_neighbors_cypher(
    edge_types: list[str] | None = None,
    direction: str = "both",
    min_weight: float = 0.0,
) -> str:
    """Build a Cypher template that fetches neighbors of a ``:base`` node.

    Parameters
    ----------
    edge_types:
        ``None`` means "all nine valid types" (the full
        :data:`memory_edges.VALID_EDGE_TYPES` set, sorted). Otherwise every
        name in the list is validated against the whitelist and assembled
        into a ``|``-joined Cypher relationship-type pattern.
    direction:
        One of ``"out"`` (outgoing only, ``-[r]->``), ``"in"`` (incoming
        only, ``<-[r]-``), or ``"both"`` (direction-agnostic, ``-[r]-``).
        Any other value raises ``ValueError``.
    min_weight:
        Ignored at construction time as a value — it is baked into the
        template as the ``$min_weight`` placeholder. The ``min_weight``
        argument is currently unused structurally but is retained so future
        refactors can introduce hard-coded branches for, e.g., skipping the
        weight filter when ``min_weight == 0.0``. For now it's a no-op at
        template time.

    The query binds ``$node_id`` (the seed ``entity_id``) and ``$min_weight``
    at execution time, never at template construction, so the only
    interpolated values are the validated edge-type names.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        )
    # Silence unused-var lint without relaxing the signature — min_weight is
    # kept in the signature for forward compatibility (see docstring).
    _ = float(min_weight)

    types_list = _validate_edge_type_list(edge_types)
    type_pattern = "|".join(types_list)

    if direction == "out":
        rel_clause = f"-[r:{type_pattern}]->"
    elif direction == "in":
        rel_clause = f"<-[r:{type_pattern}]-"
    else:  # "both"
        rel_clause = f"-[r:{type_pattern}]-"

    return (
        f"MATCH (a:{_NODE_LABEL} {{{_NODE_KEY_PROPERTY}: $node_id}})"
        f"{rel_clause}"
        f"(b:{_NODE_LABEL})\n"
        f"WHERE r.weight >= $min_weight\n"
        f"RETURN b.{_NODE_KEY_PROPERTY} AS neighbor_id, "
        f"type(r) AS edge_type, r.weight AS weight, "
        f"r.created_at AS created_at, r.last_seen_at AS last_seen_at\n"
        f"ORDER BY r.weight DESC"
    )


def build_path_cypher(
    edge_types: list[str] | None = None,
    max_depth: int = 4,
) -> str:
    """Build a ``shortestPath`` template between two ``:base`` nodes.

    Parameters
    ----------
    edge_types:
        ``None`` means "all nine valid types". Otherwise every name is
        validated and union-joined into the relationship filter.
    max_depth:
        Hop count upper bound, clamped to the closed interval ``[1, 10]``.
        Values outside that range raise ``ValueError`` so an accidental
        ``max_depth=0`` or ``max_depth=100`` cannot devolve into a
        full-graph walk.

    The query binds ``$start_id`` and ``$end_id`` at execution time.
    Everything else is interpolated from the validated inputs.
    """
    if not isinstance(max_depth, int) or isinstance(max_depth, bool):
        raise ValueError(f"max_depth must be an int, got {max_depth!r}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    if max_depth > _MAX_PATH_DEPTH:
        raise ValueError(
            f"max_depth must be <= {_MAX_PATH_DEPTH}, got {max_depth}"
        )

    types_list = _validate_edge_type_list(edge_types)
    type_pattern = "|".join(types_list)

    return (
        f"MATCH (a:{_NODE_LABEL} {{{_NODE_KEY_PROPERTY}: $start_id}}), "
        f"(b:{_NODE_LABEL} {{{_NODE_KEY_PROPERTY}: $end_id}})\n"
        f"MATCH p = shortestPath("
        f"(a)-[r:{type_pattern}*..{max_depth}]-(b)"
        f")\n"
        f"RETURN p"
    )
