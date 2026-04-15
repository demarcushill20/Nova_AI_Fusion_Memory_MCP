"""Unit tests for ``app.services.associations.memory_edges`` (PLAN-0759 Sprint 4).

These tests are hermetic: no Neo4j, no Pinecone, no network, no filesystem
writes, no driver imports. They exercise the ``MemoryEdge`` dataclass, its
``__post_init__`` validation, its two helper methods, and the three
module-level constants.

Cross-reference
---------------

- Sprint 4 spec: Artifact 4 tests for ``memory_edges`` (19 required cases).
- ADR-0759 §6 (``FOLLOWS`` rename) and §7 (``(:base {entity_id})``) for the
  regression guards on ``VALID_EDGE_TYPES``.
"""

from __future__ import annotations

import pytest

from app.services.associations.memory_edges import (
    BIDIRECTIONAL_EDGE_TYPES,
    EDGE_VERSION,
    VALID_EDGE_TYPES,
    MemoryEdge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kwargs(**overrides: object) -> dict:
    """Return a valid ``MemoryEdge`` kwarg dict, optionally overridden per-test."""
    base: dict = {
        "source_id": "mem-src-001",
        "target_id": "mem-dst-002",
        "edge_type": "SIMILAR_TO",
        "weight": 0.82,
        "created_at": "2026-04-13T12:00:00+00:00",
        "last_seen_at": "2026-04-13T12:00:00+00:00",
        "created_by": "similarity_linker",
        "run_id": "sprint4-unit-test-run",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Construction (1): valid case smoke
# ---------------------------------------------------------------------------


def test_valid_memory_edge_construction() -> None:
    """A fully-populated valid edge constructs without error and exposes all fields."""
    edge = MemoryEdge(**_kwargs())
    assert edge.source_id == "mem-src-001"
    assert edge.target_id == "mem-dst-002"
    assert edge.edge_type == "SIMILAR_TO"
    assert edge.weight == 0.82
    assert edge.created_at == "2026-04-13T12:00:00+00:00"
    assert edge.last_seen_at == "2026-04-13T12:00:00+00:00"
    assert edge.created_by == "similarity_linker"
    assert edge.run_id == "sprint4-unit-test-run"
    # Default edge_version matches the module constant.
    assert edge.edge_version == EDGE_VERSION
    # Default metadata is None.
    assert edge.metadata is None


# ---------------------------------------------------------------------------
# __post_init__ validation (2-10)
# ---------------------------------------------------------------------------


def test_invalid_edge_type_raises() -> None:
    """An ``edge_type`` outside ``VALID_EDGE_TYPES`` must raise ``ValueError``."""
    with pytest.raises(ValueError, match="Invalid edge type"):
        MemoryEdge(**_kwargs(edge_type="NOT_A_REAL_TYPE"))


def test_weight_below_zero_raises() -> None:
    """Weight < 0.0 must raise ``ValueError`` (out of the closed unit interval)."""
    with pytest.raises(ValueError, match=r"Weight must be in \[0\.0, 1\.0\]"):
        MemoryEdge(**_kwargs(weight=-0.01))


def test_weight_above_one_raises() -> None:
    """Weight > 1.0 must raise ``ValueError`` (out of the closed unit interval)."""
    with pytest.raises(ValueError, match=r"Weight must be in \[0\.0, 1\.0\]"):
        MemoryEdge(**_kwargs(weight=1.01))


def test_weight_boundary_values_are_valid() -> None:
    """0.0 and 1.0 are both valid boundary weights (closed interval)."""
    low = MemoryEdge(**_kwargs(weight=0.0))
    high = MemoryEdge(**_kwargs(weight=1.0))
    assert low.weight == 0.0
    assert high.weight == 1.0


def test_empty_source_id_raises() -> None:
    """``source_id=""`` must raise ``ValueError``."""
    with pytest.raises(ValueError, match="source_id must be a non-empty string"):
        MemoryEdge(**_kwargs(source_id=""))


def test_empty_target_id_raises() -> None:
    """``target_id=""`` must raise ``ValueError``."""
    with pytest.raises(ValueError, match="target_id must be a non-empty string"):
        MemoryEdge(**_kwargs(target_id=""))


def test_self_loop_raises() -> None:
    """A self-loop (``source_id == target_id``) must raise ``ValueError``."""
    with pytest.raises(ValueError, match="Self-loops not permitted"):
        MemoryEdge(**_kwargs(source_id="mem-a", target_id="mem-a"))


def test_empty_created_by_raises() -> None:
    """``created_by=""`` must raise ``ValueError``."""
    with pytest.raises(ValueError, match="created_by must be a non-empty string"):
        MemoryEdge(**_kwargs(created_by=""))


def test_empty_run_id_raises() -> None:
    """``run_id=""`` must raise ``ValueError``."""
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        MemoryEdge(**_kwargs(run_id=""))


# ---------------------------------------------------------------------------
# as_cypher_params (11-12)
# ---------------------------------------------------------------------------


def test_as_cypher_params_shape_with_none_metadata() -> None:
    """``as_cypher_params()`` returns all required keys; ``metadata=None`` stays ``None``."""
    edge = MemoryEdge(**_kwargs(metadata=None))
    params = edge.as_cypher_params()
    expected_keys = {
        "src_id",
        "dst_id",
        "weight",
        "created_at",
        "last_seen_at",
        "created_by",
        "run_id",
        "edge_version",
        "metadata",
    }
    assert set(params.keys()) == expected_keys
    assert params["src_id"] == "mem-src-001"
    assert params["dst_id"] == "mem-dst-002"
    assert params["weight"] == 0.82
    assert params["created_at"] == "2026-04-13T12:00:00+00:00"
    assert params["last_seen_at"] == "2026-04-13T12:00:00+00:00"
    assert params["created_by"] == "similarity_linker"
    assert params["run_id"] == "sprint4-unit-test-run"
    assert params["edge_version"] == EDGE_VERSION
    assert params["metadata"] is None


def test_as_cypher_params_serializes_empty_dict_metadata() -> None:
    """An empty dict is JSON-serialized to ``"{}"`` for Neo4j compatibility."""
    edge = MemoryEdge(**_kwargs(metadata={}))
    params = edge.as_cypher_params()
    assert params["metadata"] == "{}"
    assert params["metadata"] is not None


def test_as_cypher_params_serializes_dict_metadata() -> None:
    """Dict metadata is JSON-serialized for Neo4j compatibility."""
    import json

    edge = MemoryEdge(**_kwargs(metadata={"reason": "conflict_detection"}))
    params = edge.as_cypher_params()
    assert isinstance(params["metadata"], str)
    assert json.loads(params["metadata"]) == {"reason": "conflict_detection"}


def test_deserialize_metadata_round_trip() -> None:
    """Serialization and deserialization are symmetric."""
    import json

    original = {"reason": "conflict_detection", "similarity": 0.91}
    edge = MemoryEdge(**_kwargs(metadata=original))
    params = edge.as_cypher_params()
    recovered = MemoryEdge.deserialize_metadata(params["metadata"])
    assert recovered == original


def test_deserialize_metadata_none() -> None:
    """None stays None through deserialization."""
    assert MemoryEdge.deserialize_metadata(None) is None


def test_deserialize_metadata_already_dict() -> None:
    """A dict passed directly (e.g. from test mocks) passes through unchanged."""
    d = {"key": "value"}
    assert MemoryEdge.deserialize_metadata(d) == d


# ---------------------------------------------------------------------------
# canonicalize_for_bidirectional (13-14)
# ---------------------------------------------------------------------------


def test_canonicalize_returns_smaller_first() -> None:
    """Canonicalization puts the lex-smaller id first, regardless of input order."""
    assert MemoryEdge.canonicalize_for_bidirectional("mem-z", "mem-a") == (
        "mem-a",
        "mem-z",
    )
    assert MemoryEdge.canonicalize_for_bidirectional("mem-a", "mem-z") == (
        "mem-a",
        "mem-z",
    )


def test_canonicalize_is_idempotent() -> None:
    """Canonicalizing an already-canonical pair is a no-op."""
    first = MemoryEdge.canonicalize_for_bidirectional("mem-b", "mem-c")
    second = MemoryEdge.canonicalize_for_bidirectional(*first)
    assert first == second == ("mem-b", "mem-c")


# ---------------------------------------------------------------------------
# Constant shape guards (15-19)
# ---------------------------------------------------------------------------


def test_valid_edge_types_has_exactly_nine_members() -> None:
    """Pin the count. Prevents silent additions via copy-paste PRs."""
    assert len(VALID_EDGE_TYPES) == 9, (
        f"Expected exactly 9 valid edge types, got {len(VALID_EDGE_TYPES)}: "
        f"{sorted(VALID_EDGE_TYPES)}"
    )


def test_valid_edge_types_excludes_renamed_and_dropped_types() -> None:
    """Regression guard for ADR-0759 §6 (``FOLLOWS`` → ``MEMORY_FOLLOWS``) and the
    dropped ``MENTIONED_BY``.

    ``FOLLOWS`` is already in use by Fusion Memory for ``(:Session)-[:FOLLOWS]->(:Session)``
    session chaining (``graph_client.py:442``). PLAN-0759 must use
    ``MEMORY_FOLLOWS`` instead. ``MENTIONED_BY`` was dropped from the taxonomy
    because reverse walks over ``MENTIONS`` cover the reverse-query case.
    """
    assert "FOLLOWS" not in VALID_EDGE_TYPES, (
        "FOLLOWS must be renamed to MEMORY_FOLLOWS (ADR-0759 §6) — Session "
        "chain already uses FOLLOWS in graph_client.link_session_follows."
    )
    assert "MENTIONED_BY" not in VALID_EDGE_TYPES, (
        "MENTIONED_BY was dropped from the taxonomy in Sprint 4 scope — reverse "
        "queries walk MENTIONS backwards."
    )
    # And the positive counterparts exist.
    assert "MEMORY_FOLLOWS" in VALID_EDGE_TYPES
    assert "MENTIONS" in VALID_EDGE_TYPES


def test_bidirectional_edge_types_is_subset_of_valid() -> None:
    """``BIDIRECTIONAL_EDGE_TYPES`` must not contain any name that isn't valid overall."""
    assert BIDIRECTIONAL_EDGE_TYPES <= VALID_EDGE_TYPES


def test_bidirectional_edge_types_exact_membership() -> None:
    """Only ``SIMILAR_TO`` and ``CO_OCCURS`` are symmetric — everything else is directed."""
    assert BIDIRECTIONAL_EDGE_TYPES == frozenset({"SIMILAR_TO", "CO_OCCURS"})


def test_edge_version_is_one() -> None:
    """The Sprint 4 contract pins schema version 1. Bumping this is a breaking change."""
    assert EDGE_VERSION == 1
