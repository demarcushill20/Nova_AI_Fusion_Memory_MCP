"""Unit tests for :class:`AssociativeRecall` (PLAN-0759 Phase 6).

Design
------

These tests are **fully hermetic**. They do not touch real Neo4j,
Pinecone, or any other live service. The ``MemoryEdgeService`` and the
content fetcher are replaced with ``unittest.mock.AsyncMock`` instances
so the tests isolate the traversal logic: decay math, hop boundary,
deduplication, cycle safety, intent-aware edge selection, and the
fail-open contract at the ``MemoryService.perform_query`` integration
point.

Every test asserts either:

1. A contract on the *outputs* of ``expand()`` (correct score decay,
   correct hop tagging, correct dedup of seeds vs expansion), or
2. A contract on the *inputs* passed to the edge service (which edge
   types are requested for which intent), or
3. Integration behavior at the ``perform_query`` level (flag gating,
   default-off behavior, fail-open).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Make the ``app`` package importable ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

from app.services.associations.associative_recall import (
    INTENT_EDGE_FILTER,
    AssociativeRecall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(mid: str, score: float = 1.0, metadata: dict | None = None) -> dict:
    """Build a minimal seed result dict."""
    return {
        "id": mid,
        "rrf_score": score,
        "score": score,
        "metadata": metadata or {"text": f"text for {mid}"},
    }


def _neighbor(node_id: str, edge_type: str = "SIMILAR_TO", weight: float = 0.8) -> dict:
    """Build a neighbor dict matching MemoryEdgeService.get_neighbors output."""
    return {
        "node_id": node_id,
        "edge_type": edge_type,
        "weight": weight,
        "created_at": "2026-01-01T00:00:00Z",
        "last_seen_at": "2026-01-01T00:00:00Z",
    }


def _make_edge_service(neighbor_map: dict[str, list[dict]] | None = None) -> AsyncMock:
    """Create a mock MemoryEdgeService with controllable get_neighbors.

    MENTIONS edges are routed through ``get_memory_neighbors_via_mentions``
    by :class:`AssociativeRecall`, so this mock splits them automatically:
    ``get_neighbors`` only sees non-MENTIONS edges, while MENTIONS neighbors
    are returned by ``get_memory_neighbors_via_mentions``.
    """
    svc = AsyncMock()
    if neighbor_map is None:
        neighbor_map = {}

    async def get_neighbors(node_id: str, edge_types=None, min_weight=0.0, limit=20, **kw):
        neighbors = neighbor_map.get(node_id, [])
        # Filter by edge_types if provided
        if edge_types:
            neighbors = [n for n in neighbors if n["edge_type"] in edge_types]
        # Filter by min_weight
        neighbors = [n for n in neighbors if n["weight"] >= min_weight]
        return neighbors[:limit]

    async def get_memory_neighbors_via_mentions(node_id: str, hub_threshold=50, limit=5, **kw):
        """Return MENTIONS-type neighbors for the given node."""
        neighbors = neighbor_map.get(node_id, [])
        mentions = [n for n in neighbors if n["edge_type"] == "MENTIONS"]
        return mentions[:limit]

    svc.get_neighbors = AsyncMock(side_effect=get_neighbors)
    svc.get_memory_neighbors_via_mentions = AsyncMock(side_effect=get_memory_neighbors_via_mentions)
    return svc


def _make_content_fetcher(content_map: dict[str, dict] | None = None) -> AsyncMock:
    """Create a mock content fetcher."""
    if content_map is None:
        content_map = {}

    async def fetch(ids: list[str]) -> list[dict]:
        results = []
        for mid in ids:
            if mid in content_map:
                results.append(content_map[mid])
            else:
                results.append({
                    "id": mid,
                    "metadata": {"text": f"fetched text for {mid}"},
                    "score": 0.0,
                })
        return results

    return AsyncMock(side_effect=fetch)


# ---------------------------------------------------------------------------
# Test: expand() with general intent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_general_intent():
    """Seeds + expansion candidates returned, expansion scores follow decay."""
    neighbor_map = {
        "seed1": [_neighbor("n1", "SIMILAR_TO", 0.9)],
        "seed2": [_neighbor("n2", "MENTIONS", 0.7)],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0), _seed("seed2", 0.8)]
    results = await recall.expand(seeds, intent="general")

    # Seeds pass through unchanged
    assert results[0] is seeds[0]
    assert results[1] is seeds[1]

    # Expansion candidates are appended
    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) >= 1

    # Verify edge types used match general intent
    used_edge_types = set()
    for call in edge_svc.get_neighbors.call_args_list:
        if call.kwargs.get("edge_types"):
            for et in call.kwargs["edge_types"]:
                used_edge_types.add(et)
    assert used_edge_types <= set(INTENT_EDGE_FILTER["general"])


# ---------------------------------------------------------------------------
# Test: expand() with temporal intent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_temporal_intent():
    """Only MEMORY_FOLLOWS and SIMILAR_TO edges used for temporal_recall."""
    neighbor_map = {
        "seed1": [
            _neighbor("n1", "MEMORY_FOLLOWS", 0.9),
            _neighbor("n2", "MENTIONS", 0.8),  # Should be filtered by intent
        ],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="temporal_recall")

    # The mock filters by edge_types, so only MEMORY_FOLLOWS should survive
    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) == 1
    assert expansion[0]["expansion_edge_type"] == "MEMORY_FOLLOWS"


# ---------------------------------------------------------------------------
# Test: expand() with decision intent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_decision_intent():
    """Only SUPERSEDES, SIMILAR_TO, PROMOTED_FROM edges used."""
    neighbor_map = {
        "seed1": [
            _neighbor("n1", "SUPERSEDES", 0.9),
            _neighbor("n2", "SIMILAR_TO", 0.8),
            _neighbor("n3", "PROMOTED_FROM", 0.7),
            _neighbor("n4", "MENTIONS", 0.6),  # Not in decision intent
        ],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="decision_recall")

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    edge_types_found = {e["expansion_edge_type"] for e in expansion}
    # All found types must be in the decision_recall priority list
    assert edge_types_found <= {"SUPERSEDES", "SIMILAR_TO", "PROMOTED_FROM"}
    # MENTIONS should not appear
    assert "MENTIONS" not in edge_types_found


# ---------------------------------------------------------------------------
# Test: max hops boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_max_hops_boundary():
    """Expansion stops at 2 hops — no hop-3 candidates."""
    neighbor_map = {
        "seed1": [_neighbor("hop1_a", "SIMILAR_TO", 0.9)],
        "hop1_a": [_neighbor("hop2_a", "SIMILAR_TO", 0.8)],
        "hop2_a": [_neighbor("hop3_a", "SIMILAR_TO", 0.7)],  # Should NOT be reached
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="general")

    expansion_ids = {r["id"] for r in results if r.get("source") == "graph_expansion"}
    assert "hop1_a" in expansion_ids
    assert "hop2_a" in expansion_ids
    assert "hop3_a" not in expansion_ids


# ---------------------------------------------------------------------------
# Test: max expansion cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_max_expansion_cap():
    """No more than MAX_EXPANSION additional candidates."""
    # Create 30 neighbors for seed1 (well above MAX_EXPANSION=20)
    neighbors = [_neighbor(f"n{i}", "SIMILAR_TO", 0.9) for i in range(30)]
    neighbor_map = {"seed1": neighbors}
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="general")

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) <= AssociativeRecall.MAX_EXPANSION


# ---------------------------------------------------------------------------
# Test: min edge weight filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_min_edge_weight_filter():
    """Edges below 0.5 weight excluded."""
    neighbor_map = {
        "seed1": [
            _neighbor("n_high", "SIMILAR_TO", 0.8),
            _neighbor("n_low", "SIMILAR_TO", 0.3),  # Below MIN_EDGE_WEIGHT
        ],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="general")

    expansion_ids = {r["id"] for r in results if r.get("source") == "graph_expansion"}
    assert "n_high" in expansion_ids
    assert "n_low" not in expansion_ids


# ---------------------------------------------------------------------------
# Test: decay scoring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_decay_scoring():
    """Verify: hop1_score = seed_score * edge_weight * DECAY_PER_HOP,
    hop2_score = hop1_score * edge_weight * DECAY_PER_HOP."""
    neighbor_map = {
        "seed1": [_neighbor("hop1", "SIMILAR_TO", 0.8)],
        "hop1": [_neighbor("hop2", "SIMILAR_TO", 0.9)],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)
    decay = recall.DECAY_PER_HOP  # 0.5 after session-2 tuning

    seed_score = 1.0
    seeds = [_seed("seed1", seed_score)]
    results = await recall.expand(seeds, intent="general")

    expansion = {r["id"]: r for r in results if r.get("source") == "graph_expansion"}

    # Hop 1: seed_score * edge_weight * DECAY_PER_HOP
    expected_hop1 = seed_score * 0.8 * decay
    assert "hop1" in expansion
    assert abs(expansion["hop1"]["expansion_score"] - expected_hop1) < 1e-9
    assert expansion["hop1"]["expansion_hop"] == 1

    # Hop 2: hop1_score * edge_weight * DECAY_PER_HOP
    expected_hop2 = expected_hop1 * 0.9 * decay
    assert "hop2" in expansion
    assert abs(expansion["hop2"]["expansion_score"] - expected_hop2) < 1e-9
    assert expansion["hop2"]["expansion_hop"] == 2


# ---------------------------------------------------------------------------
# Test: dedup seed in neighbors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_dedup_seed_in_neighbors():
    """Neighbor that is already a seed is skipped."""
    neighbor_map = {
        "seed1": [
            _neighbor("seed2", "SIMILAR_TO", 0.9),  # seed2 is already a seed
            _neighbor("n1", "SIMILAR_TO", 0.8),
        ],
        "seed2": [_neighbor("n2", "SIMILAR_TO", 0.7)],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0), _seed("seed2", 0.8)]
    results = await recall.expand(seeds, intent="general")

    expansion_ids = [r["id"] for r in results if r.get("source") == "graph_expansion"]
    # seed2 should NOT appear as an expansion candidate
    assert "seed2" not in expansion_ids
    # n1 and n2 should appear
    assert "n1" in expansion_ids
    assert "n2" in expansion_ids


# ---------------------------------------------------------------------------
# Test: empty seeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_empty_seeds():
    """Returns empty list for empty seeds."""
    edge_svc = _make_edge_service()
    recall = AssociativeRecall(edge_svc)

    results = await recall.expand([], intent="general")
    assert results == []


# ---------------------------------------------------------------------------
# Test: no neighbors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_no_neighbors():
    """Returns seeds unchanged when graph has no neighbors."""
    edge_svc = _make_edge_service({})  # No neighbors for any node
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0), _seed("seed2", 0.8)]
    results = await recall.expand(seeds, intent="general")

    # Seeds returned unchanged, no expansion
    assert len(results) == 2
    assert results[0] is seeds[0]
    assert results[1] is seeds[1]


# ---------------------------------------------------------------------------
# Test: cycle safety (A -> B -> A)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_cycle_safety():
    """A->B->A cycle does not cause infinite loop or duplicate entries."""
    neighbor_map = {
        "seed1": [_neighbor("nodeB", "SIMILAR_TO", 0.8)],
        "nodeB": [_neighbor("seed1", "SIMILAR_TO", 0.8)],  # Points back to seed
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="general")

    # seed1 should appear exactly once (as the original seed)
    seed1_count = sum(1 for r in results if r["id"] == "seed1")
    assert seed1_count == 1

    # nodeB should appear exactly once (as expansion)
    nodeB_count = sum(1 for r in results if r["id"] == "nodeB")
    assert nodeB_count == 1


# ---------------------------------------------------------------------------
# Test: with content fetcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_with_content_fetcher():
    """Content fetcher is called for expansion IDs and enriches results."""
    neighbor_map = {
        "seed1": [_neighbor("n1", "SIMILAR_TO", 0.8)],
    }
    content_map = {
        "n1": {
            "id": "n1",
            "metadata": {"text": "rich text for n1", "category": "decision"},
            "score": 0.0,
        }
    }
    edge_svc = _make_edge_service(neighbor_map)
    fetcher = _make_content_fetcher(content_map)
    recall = AssociativeRecall(edge_svc, content_fetcher=fetcher)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="general")

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) == 1
    # Content should be enriched from fetcher
    assert expansion[0]["metadata"]["text"] == "rich text for n1"
    assert expansion[0]["metadata"]["category"] == "decision"
    # Fetcher was called
    fetcher.assert_called_once()


# ---------------------------------------------------------------------------
# Test: without content fetcher (IDs only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_without_content_fetcher():
    """Works without content fetcher, returning expansion candidates with empty metadata."""
    neighbor_map = {
        "seed1": [_neighbor("n1", "SIMILAR_TO", 0.8)],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc, content_fetcher=None)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="general")

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) == 1
    assert expansion[0]["id"] == "n1"
    assert expansion[0]["metadata"] == {}  # No content fetcher -> empty metadata


# ---------------------------------------------------------------------------
# Test: perform_query expand_graph=False (default behavior unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perform_query_expand_graph_false():
    """Default expand_graph=False: no graph expansion occurs, result unchanged."""
    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", return_value=[0.1] * 1536), \
         patch("app.services.memory_service.extract_entities", return_value=[]):

        # Configure required settings
        mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_settings.ASSOC_GRAPH_RECALL_ENABLED = True  # Even if True, expand_graph=False should skip
        mock_settings.TEMPORAL_DECAY_ENABLED = False
        mock_settings.MMR_ENABLED = False
        mock_settings.RERANKER_MODEL_NAME = None

        from app.services.memory_service import MemoryService

        svc = MemoryService()
        svc._initialized = True

        # Mock the query router
        from app.services.query_router import RoutingMode
        svc.query_router.route = MagicMock(return_value=RoutingMode.HYBRID)

        # Mock _semantic_query to return known results
        expected_results = [_seed("r1", 1.0), _seed("r2", 0.8)]
        svc._semantic_query = AsyncMock(return_value=expected_results)

        results = await svc.perform_query("test query", expand_graph=False)

        # AssociativeRecall should never have been created
        assert svc._associative_recall is None


# ---------------------------------------------------------------------------
# Test: perform_query expand_graph=True but flag off
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perform_query_expand_graph_true_flag_off():
    """Flag off = no expansion even if expand_graph=True."""
    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", return_value=[0.1] * 1536), \
         patch("app.services.memory_service.extract_entities", return_value=[]):

        # Configure required settings
        mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_settings.ASSOC_GRAPH_RECALL_ENABLED = False  # Flag OFF
        mock_settings.TEMPORAL_DECAY_ENABLED = False
        mock_settings.MMR_ENABLED = False
        mock_settings.RERANKER_MODEL_NAME = None

        from app.services.memory_service import MemoryService

        svc = MemoryService()
        svc._initialized = True

        # Mock the query router
        from app.services.query_router import RoutingMode
        svc.query_router.route = MagicMock(return_value=RoutingMode.HYBRID)

        # Mock _semantic_query
        expected_results = [_seed("r1", 1.0), _seed("r2", 0.8)]
        svc._semantic_query = AsyncMock(return_value=expected_results)

        results = await svc.perform_query("test query", expand_graph=True)

        # AssociativeRecall should never have been created
        assert svc._associative_recall is None


# ---------------------------------------------------------------------------
# Test: custom max_expansion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_custom_max_expansion():
    """Caller-supplied max_expansion overrides the class default."""
    neighbors = [_neighbor(f"n{i}", "SIMILAR_TO", 0.9) for i in range(10)]
    neighbor_map = {"seed1": neighbors}
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="general", max_expansion=3)

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) == 3


# ---------------------------------------------------------------------------
# Test: unknown intent falls back to general
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_unknown_intent_falls_back_to_general():
    """Unknown intent uses general edge priority."""
    neighbor_map = {
        "seed1": [
            _neighbor("n1", "SIMILAR_TO", 0.8),
            _neighbor("n2", "MENTIONS", 0.7),
        ],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="nonexistent_intent")

    # Should have used general intent edge types
    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) >= 1

    # Verify edge_types passed to get_neighbors (MENTIONS is routed through
    # get_memory_neighbors_via_mentions instead, so it won't appear here).
    all_edge_types = set()
    for call in edge_svc.get_neighbors.call_args_list:
        if call.kwargs.get("edge_types"):
            all_edge_types.update(call.kwargs["edge_types"])
    general_non_mentions = {et for et in INTENT_EDGE_FILTER["general"] if et != "MENTIONS"}
    assert all_edge_types == general_non_mentions
    # MENTIONS handled via the entity-mediated helper
    assert edge_svc.get_memory_neighbors_via_mentions.called


# ---------------------------------------------------------------------------
# Test: expansion candidates have correct format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_candidate_format():
    """Expansion candidates have all required keys."""
    neighbor_map = {
        "seed1": [_neighbor("n1", "SIMILAR_TO", 0.8)],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc, content_fetcher=None)

    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="general")

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) == 1

    candidate = expansion[0]
    required_keys = {"id", "metadata", "score", "composite_score", "source",
                     "expansion_score", "expansion_hop", "expansion_edge_type"}
    assert required_keys <= set(candidate.keys())
    assert candidate["source"] == "graph_expansion"
    assert candidate["expansion_hop"] in (1, 2)
    assert isinstance(candidate["expansion_score"], float)
    assert isinstance(candidate["expansion_edge_type"], str)


# ---------------------------------------------------------------------------
# Test: seeds without id key use entity_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_seeds_with_entity_id():
    """Seeds that use entity_id instead of id are handled."""
    neighbor_map = {
        "eid1": [_neighbor("n1", "SIMILAR_TO", 0.8)],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [{"entity_id": "eid1", "rrf_score": 1.0, "metadata": {"text": "test"}}]
    results = await recall.expand(seeds, intent="general")

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) == 1
    assert expansion[0]["id"] == "n1"


# ---------------------------------------------------------------------------
# Test: edge service error is contained per-seed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_edge_service_error_per_seed():
    """Edge service error for one seed does not prevent expansion of others."""
    call_count = 0

    async def flaky_neighbors(node_id, edge_types=None, min_weight=0.0, limit=20, **kw):
        nonlocal call_count
        call_count += 1
        if node_id == "seed1":
            raise RuntimeError("Neo4j timeout")
        return [_neighbor("n2", "SIMILAR_TO", 0.8)]

    edge_svc = AsyncMock()
    edge_svc.get_neighbors = AsyncMock(side_effect=flaky_neighbors)
    recall = AssociativeRecall(edge_svc)

    seeds = [_seed("seed1", 1.0), _seed("seed2", 0.8)]
    results = await recall.expand(seeds, intent="general")

    # seed1 expansion fails, seed2 expansion succeeds
    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) >= 1
    assert any(r["id"] == "n2" for r in expansion)


# ---------------------------------------------------------------------------
# Test: perform_query expand_graph=True, flag on (integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perform_query_expand_graph_true_flag_on():
    """Flag on + expand_graph=True: graph expansion runs and results are returned."""
    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", return_value=[0.1] * 1536), \
         patch("app.services.memory_service.extract_entities", return_value=[]):

        mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_settings.ASSOC_GRAPH_RECALL_ENABLED = True
        mock_settings.TEMPORAL_DECAY_ENABLED = False
        mock_settings.MMR_ENABLED = False
        mock_settings.RERANKER_MODEL_NAME = None

        from app.services.memory_service import MemoryService

        svc = MemoryService()
        svc._initialized = True

        from app.services.query_router import RoutingMode
        svc.query_router.route = MagicMock(return_value=RoutingMode.HYBRID)

        seed_results = [_seed("r1", 1.0), _seed("r2", 0.8)]
        svc._semantic_query = AsyncMock(return_value=seed_results)

        # Mock graph_client.driver so lazy init proceeds
        svc.graph_client = MagicMock()
        svc.graph_client.driver = MagicMock()  # Not None

        # Mock the AssociativeRecall that will be lazily created
        mock_recall = AsyncMock()
        expanded = seed_results + [
            {"id": "exp1", "score": 0.5, "rrf_score": 0.5, "source": "graph_expansion",
             "expansion_score": 0.5, "expansion_hop": 1, "expansion_edge_type": "SIMILAR_TO",
             "metadata": {"text": "expanded"}},
        ]
        mock_recall.expand = AsyncMock(return_value=expanded)

        svc._associative_recall = mock_recall

        results = await svc.perform_query("test query", expand_graph=True, intent="entity_recall")

        # Verify expand() was called
        mock_recall.expand.assert_awaited_once()
        call_kwargs = mock_recall.expand.call_args
        assert call_kwargs.kwargs.get("intent") == "entity_recall" or call_kwargs[1].get("intent") == "entity_recall"

        # Results should include expansion candidates (capped to top_k_final=15)
        assert len(results) <= 15


@pytest.mark.asyncio
async def test_perform_query_expand_graph_true_exception_fail_open():
    """Expansion failure returns seeds unchanged (fail-open)."""
    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", return_value=[0.1] * 1536), \
         patch("app.services.memory_service.extract_entities", return_value=[]):

        mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_settings.ASSOC_GRAPH_RECALL_ENABLED = True
        mock_settings.TEMPORAL_DECAY_ENABLED = False
        mock_settings.MMR_ENABLED = False
        mock_settings.RERANKER_MODEL_NAME = None

        from app.services.memory_service import MemoryService

        svc = MemoryService()
        svc._initialized = True

        from app.services.query_router import RoutingMode
        svc.query_router.route = MagicMock(return_value=RoutingMode.HYBRID)

        seed_results = [_seed("r1", 1.0), _seed("r2", 0.8)]
        svc._semantic_query = AsyncMock(return_value=seed_results)

        # Mock _associative_recall that raises
        mock_recall = AsyncMock()
        mock_recall.expand = AsyncMock(side_effect=RuntimeError("Neo4j exploded"))
        svc._associative_recall = mock_recall
        svc.graph_client = MagicMock()
        svc.graph_client.driver = MagicMock()

        results = await svc.perform_query("test query", expand_graph=True, intent="general")

        # Fail-open: seeds returned despite expansion failure
        assert len(results) == 2
        assert results[0]["id"] == "r1"
        assert results[1]["id"] == "r2"


# ---------------------------------------------------------------------------
# Test: duplicate candidate best-score-wins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_duplicate_candidate_best_score_wins():
    """When two seeds reach the same neighbor, the higher-scoring path wins."""
    neighbor_map = {
        "seed1": [_neighbor("shared_n", "SIMILAR_TO", 0.9)],
        "seed2": [_neighbor("shared_n", "SIMILAR_TO", 0.6)],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)
    decay = recall.DECAY_PER_HOP

    seeds = [_seed("seed1", 1.0), _seed("seed2", 0.5)]
    results = await recall.expand(seeds, intent="general")

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) == 1
    assert expansion[0]["id"] == "shared_n"
    # Best path: seed1 -> shared_n
    expected_score = 1.0 * 0.9 * decay
    assert abs(expansion[0]["expansion_score"] - expected_score) < 1e-9


# ---------------------------------------------------------------------------
# Test: seeds with composite_score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_seeds_with_composite_score():
    """Seeds using composite_score key are handled correctly."""
    neighbor_map = {
        "seed1": [_neighbor("n1", "SIMILAR_TO", 0.8)],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    seeds = [{"id": "seed1", "composite_score": 0.95, "metadata": {"text": "test"}}]
    results = await recall.expand(seeds, intent="general")
    decay = recall.DECAY_PER_HOP

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    assert len(expansion) == 1
    expected = 0.95 * 0.8 * decay
    assert abs(expansion[0]["expansion_score"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# Test: mixed edge types all returned when allowed by filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_mixed_edge_types_all_returned():
    """All neighbor edge types allowed by the intent filter are returned."""
    neighbor_map = {
        "seed1": [
            _neighbor("n1", "SIMILAR_TO", 0.9),
            _neighbor("n2", "MENTIONS", 0.8),
            _neighbor("n3", "MEMORY_FOLLOWS", 0.7),
        ],
    }
    edge_svc = _make_edge_service(neighbor_map)
    recall = AssociativeRecall(edge_svc)

    # General intent allows SIMILAR_TO, MENTIONS, MEMORY_FOLLOWS
    seeds = [_seed("seed1", 1.0)]
    results = await recall.expand(seeds, intent="general")

    expansion = [r for r in results if r.get("source") == "graph_expansion"]
    expansion_types = {e["expansion_edge_type"] for e in expansion}
    assert "SIMILAR_TO" in expansion_types
    assert "MENTIONS" in expansion_types
    assert "MEMORY_FOLLOWS" in expansion_types
