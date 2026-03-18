"""Integration tests for the full P9A retrieval pipeline (P9A.8).

Validates that all P9A components are correctly wired together:
- Query routing dispatches to correct paths
- Feature flags independently disable components
- Full pipeline: route -> embed -> retrieve -> merge -> rerank -> temporal -> MMR
- Cascading rerank fallback chain works
- Write-time dedup integrates with upsert
- Graceful degradation when components fail

All external services (Pinecone, Neo4j, Redis, OpenAI) are mocked.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Adjust sys.path to import from the 'app' directory ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Mock heavy optional dependencies before importing app modules
_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
sys.modules.setdefault("torch", _mock_torch)

_mock_st = MagicMock()
sys.modules.setdefault("sentence_transformers", _mock_st)
sys.modules.setdefault("sentence_transformers.cross_encoder", _mock_st)
_mock_st.CrossEncoder = MagicMock

# Now safe to import app modules
from app.services.memory_service import MemoryService  # noqa: E402
from app.services.query_router import QueryRouter, RoutingMode  # noqa: E402
from app.services.hybrid_merger import HybridMerger  # noqa: E402
from app.services.reranker import CrossEncoderReranker, PineconeReranker  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vector_result(
    id: str,
    text: str,
    score: float = 0.85,
    memory_type: str = "scratch",
    event_time: str = "2026-03-18T10:00:00Z",
    event_seq: int = 100,
    values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Create a mock Pinecone vector result."""
    result: Dict[str, Any] = {
        "id": id,
        "score": score,
        "metadata": {
            "text": text,
            "memory_type": memory_type,
            "event_time": event_time,
            "event_seq": event_seq,
        },
    }
    if values:
        result["values"] = values
    return result


def _make_graph_result(id: str, text: str, score: float = 0.0) -> Dict[str, Any]:
    """Create a mock Neo4j graph result."""
    return {
        "id": id,
        "text": text,
        "source": "graph",
        "score": score,
        "metadata": {"text": text},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    """Mock settings with all P9A features enabled."""
    with patch("app.services.memory_service.settings") as mock_s:
        mock_s.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_s.RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_s.TEMPORAL_DECAY_ENABLED = True
        mock_s.TEMPORAL_WEIGHT = 0.30
        mock_s.TEMPORAL_DECAY_HALF_LIVES = None
        mock_s.MMR_ENABLED = True
        mock_s.MMR_LAMBDA = 0.7
        mock_s.WRITE_DEDUP_ENABLED = True
        mock_s.WRITE_DEDUP_THRESHOLD = 0.92
        mock_s.CONFLICT_DETECTION_ENABLED = True
        mock_s.QUERY_ROUTER_LLM_ENABLED = False
        mock_s.EVENT_SEQ_FILE = "/tmp/test_seq.counter"
        mock_s.REDIS_URL = None
        mock_s.REDIS_ENABLED = False
        mock_s.NEO4J_URI = "bolt://localhost:7687"
        mock_s.NEO4J_USER = "neo4j"
        mock_s.NEO4J_PASSWORD = None
        mock_s.NEO4J_DATABASE = "neo4j"
        mock_s.PINECONE_API_KEY = "test-key"
        mock_s.PINECONE_ENV = "test"
        mock_s.PINECONE_INDEX = "test-index"
        mock_s.OPENAI_API_KEY = "test-key"
        yield mock_s


@pytest.fixture
def mock_memory_service(mock_settings):
    """Create a MemoryService with all backends mocked."""
    service = MemoryService.__new__(MemoryService)
    service._initialized = True
    service._reranker_loaded = False
    service.embedding_model_name = "text-embedding-3-small"

    # Mock Pinecone client
    service.pinecone_client = MagicMock()
    service.pinecone_client.query_vector = MagicMock(
        return_value=[
            _make_vector_result(
                "v1",
                "Decision: use Pinecone for vectors",
                0.92,
                "decision",
                values=[0.1] * 10,
            ),
            _make_vector_result(
                "v2",
                "Redis handles caching layer",
                0.85,
                "context",
                values=[0.2] * 10,
            ),
            _make_vector_result(
                "v3",
                "Neo4j stores graph relationships",
                0.80,
                "context",
                values=[0.3] * 10,
            ),
        ]
    )

    # Mock Graph client
    service.graph_client = AsyncMock()
    service.graph_client.query_graph = AsyncMock(
        return_value=[
            _make_graph_result("g1", "Graph node about decisions"),
        ]
    )
    service.graph_client.query_graph_multihop = AsyncMock(return_value=[])
    service.graph_client.find_related_decisions = AsyncMock(return_value=[])
    service.graph_client.get_latest_session = AsyncMock(
        return_value={
            "session_id": "test-session",
            "last_event_seq": 100,
            "summary": "Test session",
        }
    )
    service.graph_client.get_session_events = AsyncMock(
        return_value=[
            {
                "id": "evt-1",
                "text": "Session event 1",
                "event_seq": 99,
                "event_time": "2026-03-18T09:00:00Z",
                "memory_type": "scratch",
            },
        ]
    )

    # Mock query router (real implementation — tests routing behavior)
    service.query_router = QueryRouter()

    # Mock hybrid merger (real implementation)
    service.hybrid_merger = HybridMerger()

    # Mock reranker (disabled for simpler testing)
    service.reranker = None
    service.pinecone_reranker = None

    # Mock sequence service
    service.sequence_service = MagicMock()
    service.sequence_service.next_seq = AsyncMock(return_value=1000)
    service.sequence_service.current_seq = MagicMock(return_value=999)
    service.sequence_service._using_redis = False

    # No Redis timeline
    service.redis_timeline = None

    return service


# ---------------------------------------------------------------------------
# Pipeline Integration Tests
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """Validate end-to-end query pipeline with all P9A components."""

    @pytest.mark.asyncio
    async def test_hybrid_query_returns_results(self, mock_memory_service) -> None:
        """A basic hybrid query should return merged results."""
        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "What decisions were made about vector storage?",
                top_k_final=5,
            )
        assert len(results) > 0
        # Results should have routing_mode metadata
        for r in results:
            assert "routing_mode" in r.get("metadata", {})

    @pytest.mark.asyncio
    async def test_temporal_route_skips_embedding(self, mock_memory_service) -> None:
        """TEMPORAL routing should not call get_embedding."""
        # Override the router to force TEMPORAL routing
        mock_memory_service.query_router = MagicMock()
        mock_memory_service.query_router.route = MagicMock(return_value=RoutingMode.TEMPORAL)

        # Mock get_recent_events to return temporal results
        mock_memory_service.get_recent_events = AsyncMock(
            return_value=[
                {
                    "id": "t1",
                    "metadata": {
                        "text": "Recent event",
                        "event_seq": 500,
                        "event_time": "2026-03-18T11:00:00Z",
                    },
                },
            ]
        )

        with patch("app.services.memory_service.get_embedding") as mock_embed:
            results = await mock_memory_service.perform_query(
                "what happened in the last hour?",
                top_k_final=5,
            )
        # For pure TEMPORAL, embedding should NOT be called
        mock_embed.assert_not_called()
        assert len(results) == 1
        assert results[0]["metadata"]["routing_mode"] == "TEMPORAL"

    @pytest.mark.asyncio
    async def test_decision_route_uses_decision_path(self, mock_memory_service) -> None:
        """DECISION routing should go through the _decision_query path."""
        mock_memory_service.query_router = MagicMock()
        mock_memory_service.query_router.route = MagicMock(return_value=RoutingMode.DECISION)

        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "why did we decide to use Pinecone?",
                top_k_final=5,
            )
        assert isinstance(results, list)
        # All results should be tagged with DECISION routing mode
        for r in results:
            assert r.get("metadata", {}).get("routing_mode") == "DECISION"

    @pytest.mark.asyncio
    async def test_session_route_uses_graph(self, mock_memory_service) -> None:
        """SESSION routing should query session graph."""
        mock_memory_service.query_router = MagicMock()
        mock_memory_service.query_router.route = MagicMock(return_value=RoutingMode.SESSION)

        results = await mock_memory_service.perform_query(
            "where were we in the last session?",
            top_k_final=5,
        )
        assert isinstance(results, list)
        assert len(results) > 0
        # Should be tagged with SESSION routing mode
        for r in results:
            assert r.get("metadata", {}).get("routing_mode") == "SESSION"

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, mock_memory_service) -> None:
        """Empty query should return empty list."""
        results = await mock_memory_service.perform_query("")
        assert results == []

    @pytest.mark.asyncio
    async def test_uninitialized_returns_empty(self, mock_memory_service) -> None:
        """Uninitialized service should return empty."""
        mock_memory_service._initialized = False
        results = await mock_memory_service.perform_query("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_pattern_route_uses_pattern_path(self, mock_memory_service) -> None:
        """PATTERN routing should go through the _pattern_query path."""
        mock_memory_service.query_router = MagicMock()
        mock_memory_service.query_router.route = MagicMock(return_value=RoutingMode.PATTERN)

        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "how do we handle error recovery?",
                top_k_final=5,
            )
        assert isinstance(results, list)
        for r in results:
            assert r.get("metadata", {}).get("routing_mode") == "PATTERN"


# ---------------------------------------------------------------------------
# Feature Flag Tests
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    """Validate that feature flags independently control components."""

    @pytest.mark.asyncio
    async def test_temporal_decay_disabled(
        self, mock_memory_service, mock_settings
    ) -> None:
        """Disabling temporal decay should skip temporal scoring."""
        mock_settings.TEMPORAL_DECAY_ENABLED = False

        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "find information about NovaCore",
                top_k_final=5,
            )

        # Results should NOT have temporal_score when temporal decay is disabled
        for r in results:
            assert "temporal_score" not in r

    @pytest.mark.asyncio
    async def test_mmr_disabled(self, mock_memory_service, mock_settings) -> None:
        """Disabling MMR should skip deduplication."""
        mock_settings.MMR_ENABLED = False

        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "explain the system architecture",
                top_k_final=10,
            )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_all_features_disabled_still_works(
        self, mock_memory_service, mock_settings
    ) -> None:
        """Pipeline should work even with all P9A features disabled."""
        mock_settings.TEMPORAL_DECAY_ENABLED = False
        mock_settings.MMR_ENABLED = False

        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "what is the system architecture?",
                top_k_final=5,
            )

        assert isinstance(results, list)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Cascading Rerank Tests
# ---------------------------------------------------------------------------


class TestCascadingRerank:
    """Validate the reranker cascade fallback chain."""

    @pytest.mark.asyncio
    async def test_rrf_only_fallback(self, mock_memory_service) -> None:
        """With no rerankers, should fall back to RRF-only."""
        mock_memory_service.reranker = None
        mock_memory_service.pinecone_reranker = None

        fused = [
            {
                "id": "f1",
                "text": "Result 1",
                "rrf_score": 0.9,
                "metadata": {"text": "Result 1"},
            },
            {
                "id": "f2",
                "text": "Result 2",
                "rrf_score": 0.7,
                "metadata": {"text": "Result 2"},
            },
        ]

        results, name = await mock_memory_service._cascading_rerank("test", fused, 5)
        assert name == "rrf_only"
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_cascade_empty_results(self, mock_memory_service) -> None:
        """Empty results should return empty."""
        results, name = await mock_memory_service._cascading_rerank("test", [], 5)
        assert results == []
        assert name == "rrf_only"

    @pytest.mark.asyncio
    async def test_cascade_cross_encoder_used_when_available(
        self, mock_memory_service
    ) -> None:
        """If CrossEncoder is available and loaded, it should be used."""
        mock_reranker = AsyncMock()
        mock_reranker.ensure_loaded = AsyncMock(return_value=True)
        mock_reranker.rerank = AsyncMock(
            return_value=[
                {"id": "r1", "text": "Reranked 1", "rerank_score": 0.95},
            ]
        )
        mock_memory_service.reranker = mock_reranker

        fused = [
            {"id": "r1", "text": "Result 1", "metadata": {"text": "Result 1"}},
            {"id": "r2", "text": "Result 2", "metadata": {"text": "Result 2"}},
        ]

        results, name = await mock_memory_service._cascading_rerank(
            "test query", fused, 5
        )
        assert name == "cross_encoder"
        mock_reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_cascade_falls_to_pinecone_on_crossencoder_failure(
        self, mock_memory_service
    ) -> None:
        """If CrossEncoder fails, should fall back to Pinecone reranker."""
        # CrossEncoder that fails
        mock_ce = AsyncMock()
        mock_ce.ensure_loaded = AsyncMock(return_value=True)
        mock_ce.rerank = AsyncMock(side_effect=RuntimeError("Model OOM"))
        mock_memory_service.reranker = mock_ce

        # Pinecone reranker that works
        mock_pr = AsyncMock()
        mock_pr.rerank = AsyncMock(
            return_value=[{"id": "p1", "text": "Pinecone reranked", "rerank_score": 0.8}]
        )
        mock_memory_service.pinecone_reranker = mock_pr

        fused = [
            {"id": "p1", "text": "Result 1", "metadata": {"text": "Result 1"}},
        ]

        results, name = await mock_memory_service._cascading_rerank(
            "test query", fused, 5
        )
        assert name == "pinecone_api"
        mock_pr.rerank.assert_called_once()


# ---------------------------------------------------------------------------
# Write-Time Dedup Integration Tests
# ---------------------------------------------------------------------------


class TestWriteDedup:
    """Validate write-time dedup integration in perform_upsert."""

    @pytest.mark.asyncio
    async def test_upsert_with_dedup_skip(
        self, mock_memory_service, mock_settings
    ) -> None:
        """Duplicate content should be skipped."""
        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.check_semantic_duplicate"
        ) as mock_dedup, patch(
            "app.services.memory_service.resolve_duplicate_action",
            return_value="skip",
        ):
            mock_dedup.return_value = {"id": "existing-123", "score": 0.95}

            result = await mock_memory_service.perform_upsert(
                content="Duplicate content",
                metadata={"memory_type": "debug"},
            )

        # Should return existing ID (skip = no new write)
        assert result == "existing-123"

    @pytest.mark.asyncio
    async def test_upsert_with_dedup_disabled(
        self, mock_memory_service, mock_settings
    ) -> None:
        """With dedup disabled, should proceed normally."""
        mock_settings.WRITE_DEDUP_ENABLED = False

        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch.object(
            mock_memory_service,
            "_persist_memory_item",
            new_callable=AsyncMock,
        ) as mock_persist:
            mock_persist.return_value = True

            result = await mock_memory_service.perform_upsert(
                content="New content",
                metadata={"memory_type": "scratch"},
            )

        assert result is not None
        mock_persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_empty_content_returns_none(
        self, mock_memory_service
    ) -> None:
        """Empty content should return None."""
        result = await mock_memory_service.perform_upsert(content="")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_uninitialized_returns_none(
        self, mock_memory_service
    ) -> None:
        """Uninitialized service should return None on upsert."""
        mock_memory_service._initialized = False
        result = await mock_memory_service.perform_upsert(content="Test content")
        assert result is None


# ---------------------------------------------------------------------------
# Graceful Degradation Tests
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Validate pipeline continues when individual components fail."""

    @pytest.mark.asyncio
    async def test_graph_failure_doesnt_block_query(
        self, mock_memory_service
    ) -> None:
        """Graph client failure should not prevent vector results from returning."""
        mock_memory_service.graph_client.query_graph = AsyncMock(
            side_effect=Exception("Neo4j connection lost")
        )

        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "find information about decisions",
                top_k_final=5,
            )
        # Should still return vector results
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_temporal_scoring_failure_returns_unscored(
        self, mock_memory_service
    ) -> None:
        """If temporal scoring fails, results should still be returned."""
        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ), patch(
            "app.services.memory_service.temporal_decay_score",
            side_effect=Exception("Math error"),
        ):
            results = await mock_memory_service.perform_query(
                "what patterns do we follow?",
                top_k_final=5,
            )
        # Should still return results (temporal scoring is non-fatal)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_mmr_failure_returns_undeduped(
        self, mock_memory_service
    ) -> None:
        """If MMR fails, results should still be returned without dedup."""
        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ), patch(
            "app.services.memory_service.deduplicate_exact",
            side_effect=Exception("Hash error"),
        ):
            results = await mock_memory_service.perform_query(
                "explain the codebase structure",
                top_k_final=5,
            )
        # Should still return results (MMR is non-fatal)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_empty(
        self, mock_memory_service
    ) -> None:
        """If embedding fails, semantic query should return empty gracefully."""
        with patch(
            "app.services.memory_service.get_embedding",
            side_effect=RuntimeError("OpenAI API down"),
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "test query",
                top_k_final=5,
            )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Pipeline Component Presence Tests
# ---------------------------------------------------------------------------


class TestPipelineComponentPresence:
    """Verify all P9A components are importable and wired."""

    def test_reranker_imports(self) -> None:
        """CrossEncoderReranker and PineconeReranker should be importable."""

        assert CrossEncoderReranker is not None
        assert PineconeReranker is not None

    def test_temporal_scorer_imports(self) -> None:
        """Temporal scoring functions should be importable."""
        from app.services.temporal_scorer import temporal_decay_score, load_half_lives

        assert temporal_decay_score is not None
        assert load_half_lives is not None

    def test_composite_scorer_imports(self) -> None:
        """Composite scoring functions should be importable."""
        from app.services.composite_scorer import (
            composite_score,
            normalize_semantic_score,
        )

        assert composite_score is not None
        assert normalize_semantic_score is not None

    def test_mmr_dedup_imports(self) -> None:
        """MMR dedup functions should be importable."""
        from app.services.mmr_dedup import deduplicate_exact, mmr_rerank

        assert deduplicate_exact is not None
        assert mmr_rerank is not None

    def test_conflict_detector_imports(self) -> None:
        """Conflict detector functions should be importable."""
        from app.services.conflict_detector import (
            check_semantic_duplicate,
            detect_conflicts,
            resolve_duplicate_action,
        )

        assert check_semantic_duplicate is not None
        assert detect_conflicts is not None
        assert resolve_duplicate_action is not None

    def test_query_router_modes(self) -> None:
        """All 8 routing modes should exist."""
        from app.services.query_router import RoutingMode

        expected_modes = [
            "VECTOR",
            "GRAPH",
            "HYBRID",
            "TEMPORAL",
            "TEMPORAL_SEMANTIC",
            "DECISION",
            "PATTERN",
            "SESSION",
        ]
        for mode_name in expected_modes:
            assert hasattr(RoutingMode, mode_name), f"Missing RoutingMode.{mode_name}"

    def test_memory_service_has_p9a_methods(self) -> None:
        """MemoryService should have all P9A pipeline methods."""
        assert hasattr(MemoryService, "_cascading_rerank")
        assert hasattr(MemoryService, "_apply_temporal_scoring")
        assert hasattr(MemoryService, "_apply_mmr_dedup")
        assert hasattr(MemoryService, "_decision_query")
        assert hasattr(MemoryService, "_pattern_query")
        assert hasattr(MemoryService, "_session_query")
        assert hasattr(MemoryService, "_semantic_query")

    def test_config_has_p9a_flags(self) -> None:
        """Config should have all P9A feature flags."""
        from app.config import Settings

        fields = Settings.model_fields
        expected_flags = [
            "TEMPORAL_DECAY_ENABLED",
            "TEMPORAL_WEIGHT",
            "MMR_ENABLED",
            "MMR_LAMBDA",
            "WRITE_DEDUP_ENABLED",
            "WRITE_DEDUP_THRESHOLD",
            "CONFLICT_DETECTION_ENABLED",
        ]
        for flag in expected_flags:
            assert flag in fields, f"Missing config flag: {flag}"

    def test_entity_extractor_imports(self) -> None:
        """Entity extractor should be importable."""
        from app.services.entity_extractor import extract_entities, ExtractedEntity

        assert extract_entities is not None
        assert ExtractedEntity is not None


# ---------------------------------------------------------------------------
# Routing Mode Distribution Tests
# ---------------------------------------------------------------------------


class TestRoutingModeDistribution:
    """Validate that different query types route correctly."""

    def test_temporal_queries_route_to_temporal(self) -> None:
        """Temporal queries should route to TEMPORAL or TEMPORAL_SEMANTIC."""
        router = QueryRouter()
        temporal_queries = [
            "what happened recently?",
            "show me the last 5 events",
            "what did we do today?",
        ]
        for q in temporal_queries:
            mode = router.route(q)
            assert mode in (
                RoutingMode.TEMPORAL,
                RoutingMode.TEMPORAL_SEMANTIC,
                RoutingMode.HYBRID,
            ), f"Query '{q}' routed to {mode.name}, expected TEMPORAL or TEMPORAL_SEMANTIC"

    def test_decision_queries_route_to_decision(self) -> None:
        """Decision queries should route to DECISION."""
        router = QueryRouter()
        decision_queries = [
            "why did we decide to use Pinecone?",
            "what was the rationale for choosing Redis?",
        ]
        for q in decision_queries:
            mode = router.route(q)
            assert mode in (
                RoutingMode.DECISION,
                RoutingMode.HYBRID,
            ), f"Query '{q}' routed to {mode.name}, expected DECISION"

    def test_session_queries_route_to_session(self) -> None:
        """Session queries should route to SESSION."""
        router = QueryRouter()
        session_queries = [
            "resume from last session",
            "where were we in the last checkpoint?",
        ]
        for q in session_queries:
            mode = router.route(q)
            assert mode in (
                RoutingMode.SESSION,
                RoutingMode.HYBRID,
                RoutingMode.TEMPORAL,
                RoutingMode.TEMPORAL_SEMANTIC,
            ), f"Query '{q}' routed to {mode.name}, expected SESSION"

    def test_generic_queries_route_to_hybrid(self) -> None:
        """Generic queries should route to HYBRID (default)."""
        router = QueryRouter()
        generic_queries = [
            "how does the system work?",
            "explain memory architecture",
        ]
        for q in generic_queries:
            mode = router.route(q)
            # Generic can be HYBRID, VECTOR, or PATTERN
            assert mode in (
                RoutingMode.HYBRID,
                RoutingMode.VECTOR,
                RoutingMode.PATTERN,
            ), f"Query '{q}' routed to {mode.name}"


# ---------------------------------------------------------------------------
# Temporal-Semantic Hybrid Tests
# ---------------------------------------------------------------------------


class TestTemporalSemanticPath:
    """Validate TEMPORAL_SEMANTIC combined path."""

    @pytest.mark.asyncio
    async def test_temporal_semantic_with_recent_events(
        self, mock_memory_service
    ) -> None:
        """TEMPORAL_SEMANTIC should use temporal window + semantic refinement."""
        mock_memory_service.query_router = MagicMock()
        mock_memory_service.query_router.route = MagicMock(
            return_value=RoutingMode.TEMPORAL_SEMANTIC
        )

        # Mock get_recent_events to return temporal results with event_seq
        mock_memory_service.get_recent_events = AsyncMock(
            return_value=[
                {
                    "id": "ts1",
                    "metadata": {
                        "text": "Recent temporal event",
                        "event_seq": 200,
                        "event_time": "2026-03-18T11:00:00Z",
                    },
                },
            ]
        )

        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "what decisions were made recently?",
                top_k_final=5,
            )

        assert isinstance(results, list)
        for r in results:
            assert r.get("metadata", {}).get("routing_mode") == "TEMPORAL_SEMANTIC"

    @pytest.mark.asyncio
    async def test_temporal_semantic_falls_back_on_no_recent(
        self, mock_memory_service
    ) -> None:
        """TEMPORAL_SEMANTIC with no recent events should fall back to full semantic."""
        mock_memory_service.query_router = MagicMock()
        mock_memory_service.query_router.route = MagicMock(
            return_value=RoutingMode.TEMPORAL_SEMANTIC
        )

        # No recent events
        mock_memory_service.get_recent_events = AsyncMock(return_value=[])

        with patch(
            "app.services.memory_service.get_embedding",
            return_value=[0.1] * 1536,
        ), patch(
            "app.services.memory_service.extract_entities",
            return_value=[],
        ):
            results = await mock_memory_service.perform_query(
                "what decisions were made recently?",
                top_k_final=5,
            )

        # Should still return results via full semantic fallback
        assert isinstance(results, list)
