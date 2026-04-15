"""Unit tests for ``app.services.associations.edge_cypher`` (PLAN-0759 Sprint 4).

Hermetic string-template tests: no Neo4j driver, no parameter binding, no
execution. Every assertion is a substring or regex check on the returned
Cypher text. The point is to prove

1. The injection whitelist actually works (bad edge types never reach the
   template body);
2. The ``(:base {entity_id})`` schema anchor was applied throughout, not the
   v2 plan's illustrative ``:Memory {memory_id}``;
3. Traversal queries filter by named edge types so the pre-existing
   ``INCLUDES`` relationship (``:Session->:base``, 517 edges per the Phase 0
   audit) cannot leak into recall results;
4. Bound-checking on ``max_depth`` and ``direction`` actually fires.

Cross-reference
---------------

- Sprint 4 spec: Artifact 4 tests for ``edge_cypher`` (14 required cases).
- Phase 0 schema audit: ``MEMORY/plans/PLAN-0759/phase0_schema_audit.md``.
- ADR-0759 §7: node label / key verification.
"""

from __future__ import annotations

import pytest

from app.services.associations.edge_cypher import (
    build_delete_edges_by_run_cypher,
    build_merge_edge_cypher,
    build_neighbors_cypher,
    build_path_cypher,
)
from app.services.associations.memory_edges import VALID_EDGE_TYPES


# ---------------------------------------------------------------------------
# build_merge_edge_cypher (1-5)
# ---------------------------------------------------------------------------


def test_merge_cypher_for_similar_to_contains_schema_anchors() -> None:
    """The ``SIMILAR_TO`` merge template must carry ``:base``, ``entity_id``,
    the ``SIMILAR_TO`` relationship type, and the ``$src_id`` / ``$weight``
    parameter placeholders.
    """
    q = build_merge_edge_cypher("SIMILAR_TO")
    assert ":base" in q, "node label :base must appear in merge template"
    assert ":SIMILAR_TO" in q, "relationship type must appear after validation"
    assert "entity_id" in q, "primary key property must appear in match clause"
    assert "$src_id" in q, "source id must be bound as $src_id, not interpolated"
    assert "$weight" in q, "weight must be bound as $weight, not interpolated"


def test_merge_cypher_rejects_unknown_edge_type() -> None:
    """Any edge type not in the whitelist must raise before any string building."""
    with pytest.raises(ValueError, match="Invalid edge type"):
        build_merge_edge_cypher("NOT_A_REAL_TYPE")


def test_merge_cypher_rejects_cypher_injection_payload() -> None:
    """The classic Cypher-injection payload must be rejected by the whitelist.

    If interpolation ever happened *before* whitelist validation, a malicious
    ``edge_type`` like ``"SIMILAR_TO; MATCH (n) DETACH DELETE n //"`` could
    smuggle a ``DELETE`` clause into the template body. The validator must
    see that the whole string is not an element of ``VALID_EDGE_TYPES`` and
    raise, long before any f-string interpolation happens.
    """
    payload = "SIMILAR_TO; MATCH (n) DETACH DELETE n //"
    with pytest.raises(ValueError, match="Invalid edge type"):
        build_merge_edge_cypher(payload)


def test_merge_cypher_rejects_follows_rename_guard() -> None:
    """``FOLLOWS`` must be rejected — PLAN-0759 uses ``MEMORY_FOLLOWS`` instead.

    ``FOLLOWS`` is reserved for the existing ``(:Session)-[:FOLLOWS]->(:Session)``
    chain in ``graph_client.link_session_follows``. Allowing it here would
    collide with session bookkeeping and make rollback by type unsafe.
    """
    with pytest.raises(ValueError, match="Invalid edge type"):
        build_merge_edge_cypher("FOLLOWS")


def test_merge_cypher_rejects_mentioned_by_drop_guard() -> None:
    """``MENTIONED_BY`` must be rejected — dropped from the taxonomy in Sprint 4.

    The reverse-query use case is covered by walking ``MENTIONS`` backwards.
    """
    with pytest.raises(ValueError, match="Invalid edge type"):
        build_merge_edge_cypher("MENTIONED_BY")


# ---------------------------------------------------------------------------
# build_delete_edges_by_run_cypher (6)
# ---------------------------------------------------------------------------


def test_delete_by_run_id_cypher_scopes_to_base_endpoints() -> None:
    """The rollback query must pin both endpoints to ``:base`` and match on ``$run_id``.

    Matches all three load-bearing fragments explicitly:

    - ``MATCH (a:base)-[r]->(b:base)`` — scope guard so bad rollbacks cannot
      sweep non-``:base`` edges (e.g. the existing ``(:Session)-[:FOLLOWS]->(:Session)``
      chain or the 517 ``(:Session)-[:INCLUDES]->(:base)`` edges seen in the
      Phase 0 schema audit).
    - ``WHERE r.run_id = $run_id`` — parameter-bound filter.
    - ``DELETE r`` — the actual destructive clause.
    """
    q = build_delete_edges_by_run_cypher()
    assert "MATCH (a:base)-[r]->(b:base)" in q
    assert "WHERE r.run_id = $run_id" in q
    assert "DELETE r" in q


# ---------------------------------------------------------------------------
# build_neighbors_cypher (7-10)
# ---------------------------------------------------------------------------


def test_neighbors_cypher_with_explicit_types_joins_with_pipe() -> None:
    """Explicit ``edge_types`` must be joined with ``|`` in Cypher relationship syntax.

    The query must also carry ``:base`` labels on both endpoints and the
    ``$min_weight`` parameter binding.
    """
    q = build_neighbors_cypher(
        edge_types=["SIMILAR_TO", "MENTIONS"], direction="both", min_weight=0.5
    )
    assert "SIMILAR_TO|MENTIONS" in q, (
        "edge types must be joined with Cypher's relationship-union '|' operator"
    )
    # :base on both endpoints — structural guarantee against walking into Session.
    assert q.count(":base") >= 2, (
        "both endpoints of the neighbor walk must be :base-scoped, not untyped"
    )
    assert "$min_weight" in q, "min_weight must be bound at execution time"


def test_neighbors_cypher_with_none_types_contains_all_nine() -> None:
    """``edge_types=None`` expands to every name in ``VALID_EDGE_TYPES``."""
    q = build_neighbors_cypher(edge_types=None, direction="out", min_weight=0.0)
    # Every valid type must appear in the returned string.
    for et in VALID_EDGE_TYPES:
        assert et in q, f"expected {et!r} in full-expansion neighbors cypher"
    # And the total count in the union pattern is 9.
    # We count occurrences of each — since they're all unique substrings,
    # simple membership covers it; this is an explicit redundancy.
    assert len(VALID_EDGE_TYPES) == 9


def test_neighbors_cypher_rejects_bad_direction() -> None:
    """An invalid ``direction`` raises before any string building."""
    with pytest.raises(ValueError, match="direction must be one of"):
        build_neighbors_cypher(
            edge_types=["SIMILAR_TO"], direction="sideways", min_weight=0.0
        )


def test_neighbors_cypher_rejects_bad_edge_type_in_list() -> None:
    """A bogus name anywhere in ``edge_types`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Invalid edge type"):
        build_neighbors_cypher(
            edge_types=["BOGUS"], direction="both", min_weight=0.0
        )


# ---------------------------------------------------------------------------
# build_path_cypher (11-13)
# ---------------------------------------------------------------------------


def test_path_cypher_uses_shortest_path_and_depth_bound() -> None:
    """The path query must use Cypher's ``shortestPath`` and include the depth bound."""
    q = build_path_cypher(edge_types=None, max_depth=3)
    assert "shortestPath" in q, "path query must use Cypher's shortestPath function"
    assert ":base" in q, "both endpoints must be :base-scoped"
    assert "*..3" in q, (
        "the depth bound must appear in the relationship quantifier (*..3)"
    )


def test_path_cypher_rejects_zero_max_depth() -> None:
    """``max_depth=0`` is nonsense — a self-match is not a path — and must raise."""
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        build_path_cypher(edge_types=None, max_depth=0)


def test_path_cypher_rejects_out_of_range_max_depth() -> None:
    """``max_depth > 10`` would let queries degenerate into full-graph walks."""
    with pytest.raises(ValueError, match="max_depth must be <= 10"):
        build_path_cypher(edge_types=None, max_depth=11)


# ---------------------------------------------------------------------------
# Schema regression guard (14)
# ---------------------------------------------------------------------------


def test_no_placeholder_schema_leaked_into_any_template() -> None:
    """None of the returned Cypher strings may contain ``:Memory`` or ``memory_id``.

    The v2 plan's Step 1.1 snippet uses ``(:Memory {memory_id})`` as a
    placeholder. ADR-0759 §7 corrected this to ``(:base {entity_id})`` after
    reading ``graph_client.py:20``. This test is the regression guard — if
    anyone copy-pastes the placeholder back in, every ``edge_cypher`` builder
    should still come out clean.
    """
    samples = [
        build_merge_edge_cypher("SIMILAR_TO"),
        build_merge_edge_cypher("MEMORY_FOLLOWS"),
        build_merge_edge_cypher("MENTIONS"),
        build_delete_edges_by_run_cypher(),
        build_neighbors_cypher(edge_types=None, direction="both", min_weight=0.0),
        build_neighbors_cypher(
            edge_types=["SIMILAR_TO", "CO_OCCURS"], direction="out", min_weight=0.1
        ),
        build_path_cypher(edge_types=None, max_depth=4),
        build_path_cypher(edge_types=["MEMORY_FOLLOWS"], max_depth=2),
    ]
    for q in samples:
        assert ":Memory" not in q, (
            f"placeholder :Memory leaked into cypher template: {q!r}"
        )
        assert "memory_id" not in q, (
            f"placeholder memory_id leaked into cypher template: {q!r}"
        )
