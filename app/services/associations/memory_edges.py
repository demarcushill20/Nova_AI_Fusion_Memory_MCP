"""Typed-edge data model for PLAN-0759 associative linking.

This module is pure Python: a dataclass, three constants, and two helpers.
It performs no Neo4j I/O, imports nothing from ``app.services.graph_client``,
and must not be extended with any side-effecting code. The Cypher that
actually stores these edges lives in :mod:`edge_cypher`; the executor that
runs that Cypher lives in the (future) ``edge_service`` module.

Schema alignment
----------------

- Node identity is ``(:base {entity_id})``, verified on 2026-04-13 in
  ``app/services/graph_client.py:20`` (``NEO4J_NODE_LABEL = "base"``) and
  ``graph_client.py:97`` (the unique constraint on ``:base.entity_id``) and
  against the live Neo4j 5.19 instance in the Phase 0 schema audit report at
  ``MEMORY/plans/PLAN-0759/phase0_schema_audit.md``. The v2 plan's
  illustrative placeholder label/key snippets are not directly executable —
  see ADR-0759 §7 for the rename and the verified label cite.

- The relationship type ``FOLLOWS`` is deliberately **not** used here for
  memory-to-memory temporal adjacency. ``FOLLOWS`` is already in use for
  ``(:Session)-[:FOLLOWS]->(:Session)`` chaining in
  ``graph_client.link_session_follows`` (line 442). PLAN-0759's
  memory-to-memory temporal edge has been renamed to ``MEMORY_FOLLOWS`` per
  ADR-0759 §6.

- ``MENTIONED_BY`` has been dropped from the taxonomy. Reverse entity-mention
  queries walk ``MENTIONS`` backwards at query time. Only nine edge types
  remain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


#: Schema version for the ``MemoryEdge`` on-disk shape. Bump this whenever a
#: breaking change is introduced to the properties stored on the relationship
#: (new required fields, renamed fields, changed semantics). The current writer
#: always stamps this value into ``r.edge_version`` on the MERGE so that
#: rollback / migration scripts can filter by schema generation.
EDGE_VERSION: int = 1


#: The only nine relationship types permitted for memory-to-memory edges
#: introduced by PLAN-0759. Anything else is rejected by
#: :class:`MemoryEdge` construction and by the query builders in
#: :mod:`edge_cypher`. This is the authoritative whitelist and the primary
#: defence against Cypher injection via edge-type interpolation.
#:
#: Two historical names that are intentionally absent:
#:
#: - ``FOLLOWS`` — renamed to ``MEMORY_FOLLOWS`` (see module docstring; ADR-0759 §6).
#: - ``MENTIONED_BY`` — dropped; reverse walks over ``MENTIONS`` cover the use case.
VALID_EDGE_TYPES: frozenset[str] = frozenset(
    {
        "SIMILAR_TO",
        "SUPERSEDES",
        "COMPACTED_FROM",
        "MEMORY_FOLLOWS",
        "MENTIONS",
        "PROMOTED_FROM",
        "CAUSED_BY",
        "RELATED_TASK",
        "CO_OCCURS",
    }
)


#: Edge types whose semantics are symmetric. For these, callers should
#: canonicalize ``(source_id, target_id)`` via
#: :meth:`MemoryEdge.canonicalize_for_bidirectional` before constructing the
#: edge so that ``MERGE`` produces a single stored relationship regardless of
#: which endpoint was "source" when the linker ran. All other edge types are
#: directed and the caller's ordering is preserved.
BIDIRECTIONAL_EDGE_TYPES: frozenset[str] = frozenset({"SIMILAR_TO", "CO_OCCURS"})


@dataclass
class MemoryEdge:
    """A typed, weighted edge between two ``:base`` memory nodes.

    All validation happens in ``__post_init__`` so that invalid edges are
    rejected at construction time — before they ever reach Cypher binding,
    the driver, or the database.

    Attributes
    ----------
    source_id:
        ``entity_id`` of the source ``:base`` node. Must be a non-empty string.
    target_id:
        ``entity_id`` of the target ``:base`` node. Must be a non-empty string
        and must differ from ``source_id`` (self-loops are rejected).
    edge_type:
        One of :data:`VALID_EDGE_TYPES`. Anything else raises ``ValueError``.
    weight:
        Confidence / strength in the closed interval ``[0.0, 1.0]``. Values
        outside this range raise ``ValueError``.
    created_at:
        ISO 8601 UTC timestamp (string). This module does not enforce the
        format — the caller is expected to pass ``datetime.now(tz=timezone.utc)
        .isoformat()`` or an equivalent canonical form.
    last_seen_at:
        ISO 8601 UTC timestamp (string). On first write this should equal
        ``created_at``. The MERGE template in :mod:`edge_cypher` updates this
        field on every re-observation so that recall code can recency-weight.
    created_by:
        Short component name (e.g. ``"similarity_linker"``). Non-empty. Used
        for attribution and per-component rollback.
    run_id:
        Per-run identifier used by :func:`edge_cypher.build_delete_edges_by_run_cypher`
        to unwind a bad linker invocation without touching unrelated edges.
        Non-empty.
    edge_version:
        Defaults to :data:`EDGE_VERSION`. Stamped into ``r.edge_version`` so
        migration / rollback scripts can scope by schema generation.
    metadata:
        Optional dict of edge-specific extra fields. ``None`` is preserved
        as ``None`` and an empty ``dict`` is preserved as an empty ``dict``
        (no silent collapsing — callers may want to distinguish "no metadata
        at all" from "metadata object exists but is empty").
    """

    source_id: str
    target_id: str
    edge_type: str
    weight: float
    created_at: str
    last_seen_at: str
    created_by: str
    run_id: str
    edge_version: int = EDGE_VERSION
    metadata: dict | None = None

    def __post_init__(self) -> None:
        # edge_type must be one of the nine permitted values. This is also
        # the first line of defence against Cypher injection when the type
        # name gets interpolated into a query template downstream.
        if self.edge_type not in VALID_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge type: {self.edge_type!r} (not in VALID_EDGE_TYPES)"
            )

        # Weight must be in the closed unit interval. Boundary values are
        # valid (0.0 = certain-no-affinity, 1.0 = maximum affinity).
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise ValueError(
                f"Weight must be a float in [0.0, 1.0], got {self.weight!r}"
            )
        if not (0.0 <= float(self.weight) <= 1.0):
            raise ValueError(f"Weight must be in [0.0, 1.0], got {self.weight}")

        # Endpoint IDs must be non-empty strings. This catches both ``""``
        # and the common ``None``-bleed mistake where a caller forgot to fill
        # in one side before constructing the edge.
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError(
                f"source_id must be a non-empty string, got {self.source_id!r}"
            )
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError(
                f"target_id must be a non-empty string, got {self.target_id!r}"
            )

        # Self-loops make no semantic sense for any of the nine edge types
        # and would also poison traversal queries. Reject up front.
        if self.source_id == self.target_id:
            raise ValueError("Self-loops not permitted")

        # Attribution fields must be non-empty so per-component /
        # per-run rollback is actually possible.
        if not isinstance(self.created_by, str) or not self.created_by:
            raise ValueError(
                f"created_by must be a non-empty string, got {self.created_by!r}"
            )
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError(
                f"run_id must be a non-empty string, got {self.run_id!r}"
            )

    def as_cypher_params(self) -> dict[str, Any]:
        """Return the parameter dict that binds into the MERGE template.

        The returned keys line up one-for-one with the ``$param`` placeholders
        in :func:`edge_cypher.build_merge_edge_cypher` so callers can pass the
        result straight to ``session.run(query, **edge.as_cypher_params())``.

        ``metadata=None`` is preserved as ``None``. Dict metadata is
        JSON-serialized to a string because Neo4j relationship properties
        cannot hold Map values.
        """
        serialized_metadata = self.metadata
        if isinstance(self.metadata, dict):
            serialized_metadata = json.dumps(self.metadata)
        return {
            "src_id": self.source_id,
            "dst_id": self.target_id,
            "weight": float(self.weight),
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "created_by": self.created_by,
            "run_id": self.run_id,
            "edge_version": self.edge_version,
            "metadata": serialized_metadata,
        }

    @staticmethod
    def deserialize_metadata(raw: str | dict | None) -> dict | None:
        """Deserialize metadata from its Neo4j string representation."""
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {"_raw": raw}
        return None

    @classmethod
    def canonicalize_for_bidirectional(
        cls, source_id: str, target_id: str
    ) -> tuple[str, str]:
        """Return ``(source, target)`` with the lexicographically smaller id first.

        For bidirectional edge types (``SIMILAR_TO``, ``CO_OCCURS``) the
        relationship is symmetric, so storing both ``(a)-[:SIMILAR_TO]->(b)``
        and ``(b)-[:SIMILAR_TO]->(a)`` would produce two Neo4j edges for the
        same logical fact, double-counting in traversal queries and
        degree-weighted ranking. Callers of the MERGE template for
        bidirectional types should pass endpoints through this method first
        so the ``(a, b)`` pair is canonical regardless of which side the
        linker observed as "source".

        This function is idempotent and makes no guarantee about which input
        becomes "source" beyond "the smaller id under Python string ordering
        wins".
        """
        if source_id <= target_id:
            return (source_id, target_id)
        return (target_id, source_id)
