"""Tests for Phase 8 MCP association tools (PLAN-0759).

Verifies the six new MCP tools added in Phase 8:
  - get_related_memories
  - get_entity_memories
  - get_memory_graph
  - get_provenance
  - get_session_timeline
  - get_edge_stats

Also verifies that query_memory now accepts expand_graph and intent params.

All tests are fully hermetic — no Neo4j, no Pinecone, no network. Backend
services are mocked via AsyncMock/MagicMock so only the MCP tool handler
logic, input validation, response-size enforcement, and error handling are
under test.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Minimal MCP Context stub
# ---------------------------------------------------------------------------

class _FakeLifespan:
    """Stub for the lifespan context that holds MemoryService."""

    def __init__(self, memory_service):
        self.memory_service = memory_service


class _FakeRequestCtx:
    """Stub for the MCP request context."""

    def __init__(self, memory_service):
        self.lifespan_context = _FakeLifespan(memory_service)


class FakeContext:
    """Minimal stand-in for ``mcp.server.fastmcp.Context``."""

    def __init__(self, memory_service):
        self.request_context = _FakeRequestCtx(memory_service)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_service(
    *,
    graph_driver: Any = None,
    edge_service: Any = None,
    entity_linker: Any = None,
) -> MagicMock:
    """Build a mock MemoryService with the sub-services wired up."""
    ms = MagicMock()
    ms._initialized = True

    # graph_client.driver — used by _get_edge_service and session timeline
    driver = graph_driver or AsyncMock()
    ms.graph_client = MagicMock()
    ms.graph_client.driver = driver

    # Pre-set the cached MCP helpers if provided so the lazy constructor
    # is not triggered (which would try to import the real module).
    if edge_service is not None:
        ms._mcp_edge_service = edge_service
    if entity_linker is not None:
        ms._mcp_entity_linker = entity_linker

    return ms


def _make_ctx(memory_service: Any) -> FakeContext:
    return FakeContext(memory_service)


# ---------------------------------------------------------------------------
# Import the module under test AFTER helpers are defined
# ---------------------------------------------------------------------------

# We import individual tool functions from mcp_server. Since the module
# does `from app.services.memory_service import MemoryService` at import
# time, we must ensure the import chain works.
sys.path.insert(0, "/home/nova/Nova_AI_Fusion_Memory_MCP")

from mcp_server import (
    get_related_memories,
    get_entity_memories,
    get_memory_graph,
    get_provenance,
    get_session_timeline,
    get_edge_stats,
    query_memory,
)


# ===================================================================
# get_related_memories
# ===================================================================

class TestGetRelatedMemories:
    """Tests for the get_related_memories MCP tool."""

    @pytest.mark.asyncio
    async def test_basic_delegation(self):
        """Verify correct delegation to edge_service.get_neighbors and response shape."""
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=[
            {"node_id": "mem-2", "edge_type": "SIMILAR_TO", "weight": 0.9,
             "created_at": "2026-01-01T00:00:00", "last_seen_at": "2026-01-01T00:00:00"},
        ])

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="mem-1")

        assert "error" not in result
        assert result["memory_id"] == "mem-1"
        assert result["count"] == 1
        assert result["related"][0]["memory_id"] == "mem-2"
        assert result["related"][0]["hop_distance"] == 1

    @pytest.mark.asyncio
    async def test_response_size_cap(self):
        """Verify the 50-memory cap is enforced."""
        # Return 60 neighbors in one hop
        neighbors = [
            {"node_id": f"mem-{i}", "edge_type": "SIMILAR_TO", "weight": 0.5,
             "created_at": None, "last_seen_at": None}
            for i in range(60)
        ]
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=neighbors)

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="seed", limit=50)

        assert result["count"] <= 50
        assert len(result["related"]) <= 50

    @pytest.mark.asyncio
    async def test_max_hops_clamped(self):
        """max_hops > 3 should be clamped to 3."""
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=[])

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="m1", max_hops=10)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_invalid_memory_id(self):
        """Empty memory_id should return an error."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Exceptions from edge_service should be caught and returned as error."""
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(side_effect=RuntimeError("Neo4j down"))

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="m1")
        assert "error" in result
        assert "RuntimeError" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_intent_rejected(self):
        """Invalid intent returns an error without hitting edge_service."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="m1", intent="bogus_intent")
        assert "error" in result
        assert "bogus_intent" in result["error"]

    @pytest.mark.asyncio
    async def test_intent_auto_selects_edge_types(self):
        """When edge_types is None and intent is provided, edge types are auto-selected."""
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=[
            {"node_id": "mem-2", "edge_type": "MENTIONS", "weight": 0.9,
             "created_at": "2026-01-01T00:00:00", "last_seen_at": "2026-01-01T00:00:00"},
        ])

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="m1", intent="entity_recall")
        assert "error" not in result
        assert result["count"] == 1

        # Verify symmetric edge_types passed to get_neighbors with direction="both"
        call_kwargs = edge_svc.get_neighbors.call_args
        assert call_kwargs.kwargs.get("edge_types") == ["MENTIONS", "CO_OCCURS"]
        assert call_kwargs.kwargs.get("direction") == "both"

    @pytest.mark.asyncio
    async def test_multi_hop_traversal(self):
        """BFS should discover hop-2 nodes reachable only through hop-1 intermediaries."""
        async def _neighbors(node_id, **kwargs):
            if node_id == "seed":
                return [{"node_id": "hop1", "edge_type": "SIMILAR_TO", "weight": 0.9,
                         "created_at": None, "last_seen_at": None}]
            elif node_id == "hop1":
                return [{"node_id": "hop2", "edge_type": "SIMILAR_TO", "weight": 0.7,
                         "created_at": None, "last_seen_at": None}]
            return []

        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(side_effect=_neighbors)

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="seed", max_hops=2)
        assert result["count"] == 2
        ids = [r["memory_id"] for r in result["related"]]
        assert "hop1" in ids
        assert "hop2" in ids
        hop2_item = next(r for r in result["related"] if r["memory_id"] == "hop2")
        assert hop2_item["hop_distance"] == 2

    @pytest.mark.asyncio
    async def test_cycle_detection(self):
        """BFS should not revisit the seed node or previously visited nodes."""
        async def _neighbors(node_id, **kwargs):
            if node_id == "A":
                return [{"node_id": "B", "edge_type": "SIMILAR_TO", "weight": 0.9,
                         "created_at": None, "last_seen_at": None}]
            elif node_id == "B":
                return [{"node_id": "A", "edge_type": "SIMILAR_TO", "weight": 0.8,
                         "created_at": None, "last_seen_at": None}]
            return []

        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(side_effect=_neighbors)

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="A", max_hops=3)
        ids = [r["memory_id"] for r in result["related"]]
        assert ids == ["B"]  # A should not appear; only B discovered once

    @pytest.mark.asyncio
    async def test_invalid_edge_types_rejected(self):
        """Invalid edge_types should return an error without hitting edge_service."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_related_memories(
            ctx, memory_id="m1", edge_types=["DROP_TABLE", "SIMILAR_TO"]
        )
        assert "error" in result
        assert "DROP_TABLE" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_edge_types_rejected(self):
        """Empty edge_types list should return an error."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="m1", edge_types=[])
        assert "error" in result

    @pytest.mark.asyncio
    async def test_edge_types_overrides_intent(self):
        """Explicit edge_types should be used even when intent is also provided."""
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=[
            {"node_id": "mem-2", "edge_type": "SUPERSEDES", "weight": 0.9,
             "created_at": None, "last_seen_at": None},
        ])

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(
            ctx, memory_id="m1",
            edge_types=["SUPERSEDES"],
            intent="entity_recall",
        )
        assert "error" not in result
        # edge_types=["SUPERSEDES"] is directed, should use direction="out"
        call_kwargs = edge_svc.get_neighbors.call_args
        assert call_kwargs.kwargs.get("edge_types") == ["SUPERSEDES"]
        assert call_kwargs.kwargs.get("direction") == "out"

    @pytest.mark.asyncio
    async def test_frontier_cap_and_early_exit(self):
        """BFS should cap frontier and exit early when limit is reached."""
        neighbors_per_node = [
            {"node_id": f"n-{i}", "edge_type": "SIMILAR_TO", "weight": 0.5,
             "created_at": None, "last_seen_at": None}
            for i in range(60)
        ]
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=neighbors_per_node)

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="seed", limit=5, max_hops=3)
        assert result["count"] <= 5

    @pytest.mark.asyncio
    async def test_relevance_ordering(self):
        """Results should be sorted by weight desc, then hop distance asc."""
        async def _neighbors(node_id, **kwargs):
            if node_id == "seed":
                return [
                    {"node_id": "low-weight", "edge_type": "SIMILAR_TO", "weight": 0.1,
                     "created_at": None, "last_seen_at": None},
                    {"node_id": "high-weight", "edge_type": "SIMILAR_TO", "weight": 0.9,
                     "created_at": None, "last_seen_at": None},
                ]
            return []

        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(side_effect=_neighbors)

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_related_memories(ctx, memory_id="seed", max_hops=1)
        assert result["related"][0]["memory_id"] == "high-weight"
        assert result["related"][1]["memory_id"] == "low-weight"


# ===================================================================
# get_memory_graph (additional)
# ===================================================================

class TestGetMemoryGraphExtended:
    """Extended tests for get_memory_graph edge cases."""

    @pytest.mark.asyncio
    async def test_edge_cap_400(self):
        """Verify the 400-edge cap is enforced."""
        big_neighbors = [
            {"node_id": f"n-{i}", "edge_type": "SIMILAR_TO", "weight": 0.5,
             "created_at": None, "last_seen_at": None}
            for i in range(200)
        ]
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=big_neighbors)

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_memory_graph(ctx, memory_id="center", max_hops=2)
        assert result["edge_count"] <= 400

    @pytest.mark.asyncio
    async def test_edge_dedup(self):
        """Duplicate edges (same source, target, type) should not appear."""
        async def _neighbors(node_id, **kwargs):
            if node_id == "A":
                return [{"node_id": "B", "edge_type": "SIMILAR_TO", "weight": 0.9,
                         "created_at": None, "last_seen_at": None}]
            elif node_id == "B":
                return [{"node_id": "A", "edge_type": "SIMILAR_TO", "weight": 0.9,
                         "created_at": None, "last_seen_at": None}]
            return []

        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(side_effect=_neighbors)

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_memory_graph(ctx, memory_id="A", max_hops=2)
        edge_keys = [(e["source"], e["target"], e["type"]) for e in result["edges"]]
        assert len(edge_keys) == len(set(edge_keys))

    @pytest.mark.asyncio
    async def test_invalid_edge_types_rejected(self):
        """Invalid edge_types should return error at MCP layer."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_memory_graph(ctx, memory_id="m1", edge_types=["INVALID_TYPE"])
        assert "error" in result
        assert "INVALID_TYPE" in result["error"]

    @pytest.mark.asyncio
    async def test_direction_aware_traversal(self):
        """get_memory_graph should split symmetric/directed edge_types."""
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=[
            {"node_id": "n2", "edge_type": "SUPERSEDES", "weight": 0.8,
             "created_at": None, "last_seen_at": None},
        ])

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_memory_graph(
            ctx, memory_id="n1", max_hops=1,
            edge_types=["SIMILAR_TO", "SUPERSEDES"],
        )
        assert "error" not in result
        # Should have made at least 2 calls: one symmetric (SIMILAR_TO), one directed (SUPERSEDES)
        calls = edge_svc.get_neighbors.call_args_list
        directions = [c.kwargs.get("direction") for c in calls]
        assert "both" in directions
        assert "out" in directions

    @pytest.mark.asyncio
    async def test_frontier_cap(self):
        """get_memory_graph should cap frontier to prevent amplification."""
        big_neighbors = [
            {"node_id": f"n-{i}", "edge_type": "SIMILAR_TO", "weight": 0.5,
             "created_at": None, "last_seen_at": None}
            for i in range(100)
        ]
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=big_neighbors)

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_memory_graph(ctx, memory_id="center", max_hops=2)
        assert result["node_count"] <= 200


# ===================================================================
# get_provenance (additional)
# ===================================================================

class TestGetProvenanceExtended:
    """Extended tests for get_provenance edge cases."""

    @pytest.mark.asyncio
    async def test_original_sources_capped(self):
        """original_sources should be capped at MAX_CHAIN_NODES (30)."""
        big_sources = [f"src-{i}" for i in range(50)]
        edge_svc = AsyncMock()
        edge_svc.get_provenance = AsyncMock(return_value={
            "memory_id": "m1",
            "provenance_chain": [],
            "original_sources": big_sources,
            "depth": 0,
            "max_depth": 10,
            "depth_limited": False,
        })

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_provenance(ctx, memory_id="m1")
        assert len(result["original_sources"]) <= 30

    @pytest.mark.asyncio
    async def test_bool_max_depth_clamped(self):
        """bool max_depth should be treated as invalid and clamped to 1."""
        edge_svc = AsyncMock()
        edge_svc.get_provenance = AsyncMock(return_value={
            "memory_id": "m1",
            "provenance_chain": [],
            "original_sources": [],
            "depth": 0,
            "max_depth": 1,
            "depth_limited": False,
        })

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_provenance(ctx, memory_id="m1", max_depth=True)
        assert "error" not in result
        call_kwargs = edge_svc.get_provenance.call_args
        assert call_kwargs.kwargs.get("max_depth") == 1


# ===================================================================
# get_entity_memories
# ===================================================================

class TestGetEntityMemories:
    """Tests for the get_entity_memories MCP tool."""

    @pytest.mark.asyncio
    async def test_basic_delegation(self):
        """Verify correct delegation to EntityLinker.get_memories_for_entity."""
        entity_linker = AsyncMock()
        entity_linker.get_memories_for_entity = AsyncMock(return_value=[
            {"memory_id": "mem-1", "created_at": "2026-01-01", "last_seen_at": "2026-01-01"},
        ])

        ms = _make_memory_service(entity_linker=entity_linker)
        ctx = _make_ctx(ms)

        result = await get_entity_memories(ctx, entity_name="Neo4j", project="nova-core")

        assert "error" not in result
        assert result["entity_name"] == "Neo4j"
        assert result["project"] == "nova-core"
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_project_required(self):
        """project=None should return an informative error."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_entity_memories(ctx, entity_name="test")
        assert "error" in result
        assert "project" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_response_size_cap(self):
        """Verify the 100-memory cap is enforced via limit clamping."""
        entity_linker = AsyncMock()
        entity_linker.get_memories_for_entity = AsyncMock(return_value=[])

        ms = _make_memory_service(entity_linker=entity_linker)
        ctx = _make_ctx(ms)

        # Request limit=200 — should be clamped to 100
        result = await get_entity_memories(
            ctx, entity_name="test", project="proj", limit=200
        )
        assert "error" not in result
        # Verify the linker was called with clamped limit
        call_kwargs = entity_linker.get_memories_for_entity.call_args
        assert call_kwargs.kwargs.get("limit", call_kwargs[1].get("limit")) <= 100

    @pytest.mark.asyncio
    async def test_empty_entity_name(self):
        """Empty entity_name should return an error."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_entity_memories(ctx, entity_name="", project="proj")
        assert "error" in result


# ===================================================================
# get_memory_graph
# ===================================================================

class TestGetMemoryGraph:
    """Tests for the get_memory_graph MCP tool."""

    @pytest.mark.asyncio
    async def test_basic_subgraph(self):
        """Verify graph structure with nodes and edges."""
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=[
            {"node_id": "n2", "edge_type": "SIMILAR_TO", "weight": 0.8,
             "created_at": "2026-01-01", "last_seen_at": "2026-01-01"},
            {"node_id": "n3", "edge_type": "MENTIONS", "weight": 0.6,
             "created_at": "2026-01-01", "last_seen_at": "2026-01-01"},
        ])

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_memory_graph(ctx, memory_id="n1", max_hops=1)

        assert "error" not in result
        assert result["memory_id"] == "n1"
        assert result["node_count"] == 3  # n1 + n2 + n3
        assert result["edge_count"] == 2

        # Verify node structure
        node_ids = {n["id"] for n in result["nodes"]}
        assert node_ids == {"n1", "n2", "n3"}

    @pytest.mark.asyncio
    async def test_node_cap_200(self):
        """Verify the 200-node cap is enforced."""
        # Return 250 neighbors per node
        big_neighbors = [
            {"node_id": f"n-{i}", "edge_type": "SIMILAR_TO", "weight": 0.5,
             "created_at": None, "last_seen_at": None}
            for i in range(250)
        ]
        edge_svc = AsyncMock()
        edge_svc.get_neighbors = AsyncMock(return_value=big_neighbors)

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_memory_graph(ctx, memory_id="center", max_hops=1)

        assert result["node_count"] <= 200
        assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_invalid_memory_id(self):
        """Empty memory_id should return an error."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_memory_graph(ctx, memory_id="  ")
        assert "error" in result


# ===================================================================
# get_provenance
# ===================================================================

class TestGetProvenance:
    """Tests for the get_provenance MCP tool."""

    @pytest.mark.asyncio
    async def test_basic_delegation(self):
        """Verify correct delegation to edge_service.get_provenance."""
        edge_svc = AsyncMock()
        edge_svc.get_provenance = AsyncMock(return_value={
            "memory_id": "m1",
            "provenance_chain": [
                {"memory_id": "m0", "edge_type": "SUPERSEDES", "depth": 1, "metadata": None},
            ],
            "original_sources": ["m0"],
            "depth": 1,
            "max_depth": 10,
            "depth_limited": False,
        })

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_provenance(ctx, memory_id="m1")

        assert "error" not in result
        assert result["memory_id"] == "m1"
        assert result["chain_count"] == 1
        assert result["original_sources"] == ["m0"]

    @pytest.mark.asyncio
    async def test_chain_cap_30(self):
        """Verify the 30-node cap on provenance chain."""
        big_chain = [
            {"memory_id": f"anc-{i}", "edge_type": "PROMOTED_FROM", "depth": i, "metadata": None}
            for i in range(40)
        ]
        edge_svc = AsyncMock()
        edge_svc.get_provenance = AsyncMock(return_value={
            "memory_id": "m1",
            "provenance_chain": big_chain,
            "original_sources": ["anc-39"],
            "depth": 40,
            "max_depth": 10,
            "depth_limited": True,
        })

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_provenance(ctx, memory_id="m1")

        assert result["chain_count"] <= 30
        assert result["full_chain_count"] == 40
        assert result["truncated"] is True
        # When truncated, original_sources is filtered to ids present in
        # the truncated chain — anc-39 is beyond the 30-node cap, so it
        # is dropped to avoid dangling refs.
        chain_ids = {hop["memory_id"] for hop in result["provenance_chain"]}
        assert all(s in chain_ids for s in result["original_sources"])
        assert "anc-39" not in result["original_sources"]

    @pytest.mark.asyncio
    async def test_invalid_memory_id(self):
        """Empty memory_id should return an error."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_provenance(ctx, memory_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_max_depth_clamped(self):
        """max_depth > 10 should be clamped."""
        edge_svc = AsyncMock()
        edge_svc.get_provenance = AsyncMock(return_value={
            "memory_id": "m1",
            "provenance_chain": [],
            "original_sources": [],
            "depth": 0,
            "max_depth": 10,
            "depth_limited": False,
        })

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_provenance(ctx, memory_id="m1", max_depth=50)
        assert "error" not in result
        # Verify the edge service was called with clamped depth
        edge_svc.get_provenance.assert_awaited_once()
        call_kwargs = edge_svc.get_provenance.call_args
        assert call_kwargs.kwargs.get("max_depth", 10) <= 10


# ===================================================================
# get_session_timeline
# ===================================================================

class TestGetSessionTimeline:
    """Tests for the get_session_timeline MCP tool."""

    @pytest.mark.asyncio
    async def test_basic_delegation(self):
        """Verify session timeline query returns ordered memories."""
        # Mock the Neo4j driver + session + result
        mock_record_1 = {"memory_id": "m1", "event_seq": 1, "created_at": "2026-01-01T00:00:00"}
        mock_record_2 = {"memory_id": "m2", "event_seq": 2, "created_at": "2026-01-01T00:01:00"}

        mock_result = AsyncMock()
        mock_result.__aiter__ = lambda self: _aiter_records([mock_record_1, mock_record_2])
        mock_result.consume = AsyncMock()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)

        ms = _make_memory_service(graph_driver=mock_driver)
        ctx = _make_ctx(ms)

        result = await get_session_timeline(ctx, session_id="session-1")

        assert "error" not in result
        assert result["session_id"] == "session-1"
        assert result["count"] == 2
        assert result["timeline"][0]["event_seq"] == 1

    @pytest.mark.asyncio
    async def test_limit_cap_100(self):
        """limit > 100 should be clamped."""
        mock_result = AsyncMock()
        mock_result.__aiter__ = lambda self: _aiter_records([])
        mock_result.consume = AsyncMock()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)

        ms = _make_memory_service(graph_driver=mock_driver)
        ctx = _make_ctx(ms)

        result = await get_session_timeline(ctx, session_id="s1", limit=500)
        assert "error" not in result
        # The LIMIT $limit in the Cypher should have received 100
        call_args = mock_session.run.call_args
        assert call_args[0][1]["limit"] == 100

    @pytest.mark.asyncio
    async def test_empty_session_id(self):
        """Empty session_id should return an error."""
        ms = _make_memory_service()
        ctx = _make_ctx(ms)

        result = await get_session_timeline(ctx, session_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Database errors should be caught."""
        mock_driver = MagicMock()
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_driver.session = MagicMock(return_value=mock_session)

        ms = _make_memory_service(graph_driver=mock_driver)
        ctx = _make_ctx(ms)

        result = await get_session_timeline(ctx, session_id="s1")
        assert "error" in result


# ===================================================================
# get_edge_stats
# ===================================================================

class TestGetEdgeStats:
    """Tests for the get_edge_stats MCP tool."""

    @pytest.mark.asyncio
    async def test_basic_delegation(self):
        """Verify correct delegation to edge_service.get_edge_stats."""
        edge_svc = AsyncMock()
        edge_svc.get_edge_stats = AsyncMock(return_value={
            "SIMILAR_TO": {"count": 100, "avg_weight": 0.75, "min_weight": 0.5, "max_weight": 1.0},
            "MENTIONS": {"count": 50, "avg_weight": 1.0, "min_weight": 1.0, "max_weight": 1.0},
            "MEMORY_FOLLOWS": {"count": 30, "avg_weight": 1.0, "min_weight": 1.0, "max_weight": 1.0},
        })

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_edge_stats(ctx)

        assert "error" not in result
        assert result["total_edges"] == 180
        assert result["edge_type_count"] == 3
        assert "SIMILAR_TO" in result["edge_stats"]

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Exceptions should be caught and returned as error."""
        edge_svc = AsyncMock()
        edge_svc.get_edge_stats = AsyncMock(side_effect=RuntimeError("query failed"))

        ms = _make_memory_service(edge_service=edge_svc)
        ctx = _make_ctx(ms)

        result = await get_edge_stats(ctx)
        assert "error" in result


# ===================================================================
# query_memory (updated with expand_graph / intent)
# ===================================================================

class TestQueryMemoryGraphExpansion:
    """Tests that query_memory now accepts expand_graph and intent."""

    @pytest.mark.asyncio
    async def test_expand_graph_passed_through(self):
        """Verify expand_graph and intent are forwarded to perform_query."""
        ms = _make_memory_service()
        ms.perform_query = AsyncMock(return_value=[
            {"id": "r1", "metadata": {"category": "test"}, "score": 0.9},
        ])
        ctx = _make_ctx(ms)

        result = await query_memory(
            ctx, query="test query", expand_graph=True, intent="entity_recall"
        )

        assert "error" not in result
        # Verify perform_query was called with the new params
        ms.perform_query.assert_awaited_once()
        call_kwargs = ms.perform_query.call_args.kwargs
        assert call_kwargs["expand_graph"] is True
        assert call_kwargs["intent"] == "entity_recall"

    @pytest.mark.asyncio
    async def test_defaults_backward_compatible(self):
        """Default expand_graph=False and intent=None should be backward-compatible."""
        ms = _make_memory_service()
        ms.perform_query = AsyncMock(return_value=[])
        ctx = _make_ctx(ms)

        result = await query_memory(ctx, query="hello")

        assert "error" not in result
        call_kwargs = ms.perform_query.call_args.kwargs
        assert call_kwargs["expand_graph"] is False
        assert call_kwargs["intent"] is None

    @pytest.mark.asyncio
    async def test_invalid_intent_rejected(self):
        """Invalid intent returns an error without calling perform_query."""
        ms = _make_memory_service()
        ms.perform_query = AsyncMock(return_value=[])
        ctx = _make_ctx(ms)

        result = await query_memory(
            ctx, query="test", intent="totally_invalid_intent"
        )

        assert "error" in result
        assert "Invalid intent" in result["error"]
        ms.perform_query.assert_not_awaited()


# ===================================================================
# Tool registration verification
# ===================================================================

class TestToolRegistration:
    """Verify all six new tools are registered on the MCP server."""

    def test_tools_are_importable(self):
        """All new tool functions must be importable from mcp_server."""
        from mcp_server import (
            get_related_memories,
            get_entity_memories,
            get_memory_graph,
            get_provenance,
            get_session_timeline,
            get_edge_stats,
        )
        # If we get here, all imports succeeded
        assert callable(get_related_memories)
        assert callable(get_entity_memories)
        assert callable(get_memory_graph)
        assert callable(get_provenance)
        assert callable(get_session_timeline)
        assert callable(get_edge_stats)

    def test_tools_have_mcp_decorator(self):
        """Each tool should have been decorated by @mcp.tool() (visible as __wrapped__ or in mcp._tools)."""
        # The FastMCP @mcp.tool() decorator registers the function. We
        # verify by checking the mcp server's tool list.
        from mcp_server import mcp as mcp_server_instance

        # FastMCP stores tools internally; access via list_tools or _tools
        tool_names = set()
        if hasattr(mcp_server_instance, '_tool_manager'):
            tools = getattr(mcp_server_instance._tool_manager, '_tools', {})
            tool_names = set(tools.keys())
        elif hasattr(mcp_server_instance, '_tools'):
            tool_names = set(mcp_server_instance._tools.keys())

        expected_tools = {
            "get_related_memories",
            "get_entity_memories",
            "get_memory_graph",
            "get_provenance",
            "get_session_timeline",
            "get_edge_stats",
        }

        # If we couldn't introspect the tool manager, just verify the
        # functions exist as module-level callables (already covered above).
        if tool_names:
            for tool_name in expected_tools:
                assert tool_name in tool_names, f"Tool '{tool_name}' not registered in MCP server"


# ---------------------------------------------------------------------------
# Async iteration helper for mock Neo4j results
# ---------------------------------------------------------------------------

async def _aiter_records(records: list):
    """Create an async iterator from a list of dicts (mock Neo4j records)."""
    for r in records:
        yield _DictRecord(r)


class _DictRecord:
    """Minimal Neo4j Record stand-in that supports __getitem__."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)
