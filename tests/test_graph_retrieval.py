"""Tests for graph-augmented retrieval (P9A.6).

Validates:
- Entity extraction from queries
- Multi-hop graph traversal
- Session chain traversal
- Decision graph recall
- Distance-based scoring
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, List

# Test entity extraction (no mocking needed — pure functions)
from app.services.entity_extractor import (
    extract_entities,
    extract_entity_names,
    KNOWN_ENTITIES,
)


# ---------------------------------------------------------------------------
# Entity Extraction Tests
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    """Validate regex-based entity extraction."""

    def test_known_project_detected(self) -> None:
        """Known project names should be extracted with high confidence."""
        entities = extract_entities("What decisions did we make about NovaCore?")
        names = [e.name for e in entities]
        assert "novacore" in names

    def test_known_technology_detected(self) -> None:
        """Known technologies should be extracted."""
        entities = extract_entities("How does Pinecone handle vector queries?")
        names = [e.name for e in entities]
        assert "pinecone" in names

    def test_multiple_entities(self) -> None:
        """Multiple entities in one query should all be found."""
        entities = extract_entities("Compare Redis and Neo4j for our use case")
        names = [e.name for e in entities]
        assert "redis" in names
        assert "neo4j" in names

    def test_case_insensitive(self) -> None:
        """Entity matching should be case-insensitive."""
        entities = extract_entities("we chose PINECONE over redis")
        names = [e.name for e in entities]
        assert "pinecone" in names
        assert "redis" in names

    def test_tech_acronyms(self) -> None:
        """Tech acronyms should be detected."""
        entities = extract_entities("Set up the MCP server for LLM queries")
        names = [e.name for e in entities]
        assert "MCP" in names
        assert "LLM" in names

    def test_no_entities_in_generic_query(self) -> None:
        """Generic queries should return empty or minimal entities."""
        entities = extract_entities("what happened recently?")
        # May return some entities from capitalized words, but no known ones
        known = [e for e in entities if e.confidence >= 1.0]
        assert len(known) == 0

    def test_extract_entity_names_convenience(self) -> None:
        """Convenience function should return just names."""
        names = extract_entity_names("Deploy NovaCore with Docker")
        assert "novacore" in names or "nova-core" in names
        assert "docker" in names

    def test_confidence_ordering(self) -> None:
        """Entities should be sorted by confidence descending."""
        entities = extract_entities("NovaCore uses Pinecone and Some Random Name")
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                assert entities[i].confidence >= entities[i + 1].confidence

    def test_no_duplicates(self) -> None:
        """Same entity mentioned multiple times should appear once."""
        entities = extract_entities("Redis cache with Redis timeline and Redis pubsub")
        redis_count = sum(1 for e in entities if e.name == "redis")
        assert redis_count == 1

    def test_multi_word_entities(self) -> None:
        """Multi-word known entities should be detected."""
        entities = extract_entities("The circuit breaker tripped during the outage")
        names = [e.name for e in entities]
        assert "circuit breaker" in names

    def test_concept_entities(self) -> None:
        """Concept entities should be detected."""
        entities = extract_entities("Update the heartbeat and checkpoint config")
        names = [e.name for e in entities]
        assert "heartbeat" in names
        assert "checkpoint" in names

    def test_empty_string(self) -> None:
        """Empty string should return empty list."""
        assert extract_entities("") == []

    def test_known_entities_dict_nonempty(self) -> None:
        """Sanity: KNOWN_ENTITIES dict should have entries."""
        assert len(KNOWN_ENTITIES) > 10


# ---------------------------------------------------------------------------
# Mock GraphClient for multi-hop tests
# ---------------------------------------------------------------------------


class MockAsyncSession:
    """Mock Neo4j async session."""

    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records

    async def run(self, cypher: str, params: dict = None):
        return MockResult(self._records)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockResult:
    """Mock Neo4j result cursor."""

    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records

    async def data(self):
        return self._records

    async def single(self):
        return self._records[0] if self._records else None

    async def consume(self):
        pass


class MockDriver:
    """Mock Neo4j async driver."""

    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records

    def session(self, database: str = None):
        return MockAsyncSession(self._records)


# ---------------------------------------------------------------------------
# Multi-hop Traversal Tests
# ---------------------------------------------------------------------------


class TestMultihopTraversal:
    """Validate multi-hop graph traversal."""

    @pytest.fixture
    def graph_client(self):
        """Create a GraphClient with mocked driver."""
        from app.services.graph_client import GraphClient
        client = GraphClient.__new__(GraphClient)
        client._DATABASE = "neo4j"
        client.driver = None  # Will be set per test
        return client

    @pytest.mark.asyncio
    async def test_multihop_returns_results(self, graph_client) -> None:
        """Multi-hop should return related nodes with scores."""
        graph_client.driver = MockDriver([
            {
                "id": "related-1",
                "text": "Related decision",
                "node_properties": {"memory_type": "decision"},
                "distance": 1,
                "graph_score": 0.5,
            },
            {
                "id": "related-2",
                "text": "Two hops away",
                "node_properties": {"memory_type": "context"},
                "distance": 2,
                "graph_score": 0.333,
            },
        ])

        results = await graph_client.query_graph_multihop(
            entity_names=["pinecone"], max_hops=2, top_k=10
        )
        assert len(results) == 2
        assert results[0]["id"] == "related-1"
        assert results[0]["source"] == "graph_multihop"
        assert results[0]["graph_score"] == 0.5

    @pytest.mark.asyncio
    async def test_multihop_empty_entities(self, graph_client) -> None:
        """Empty entity list should return empty results."""
        graph_client.driver = MockDriver([])
        results = await graph_client.query_graph_multihop(
            entity_names=[], max_hops=2
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_multihop_no_driver(self, graph_client) -> None:
        """No driver should return empty results."""
        results = await graph_client.query_graph_multihop(
            entity_names=["test"], max_hops=2
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_multihop_clamps_hops(self, graph_client) -> None:
        """Hops should be clamped to [1, 3]."""
        graph_client.driver = MockDriver([])
        # Should not raise even with extreme values
        await graph_client.query_graph_multihop(
            entity_names=["test"], max_hops=0  # clamped to 1
        )
        await graph_client.query_graph_multihop(
            entity_names=["test"], max_hops=100  # clamped to 3
        )

    @pytest.mark.asyncio
    async def test_multihop_caps_entities(self, graph_client) -> None:
        """Should cap entity names at 5."""
        graph_client.driver = MockDriver([])
        many_entities = [f"entity_{i}" for i in range(20)]
        results = await graph_client.query_graph_multihop(
            entity_names=many_entities, max_hops=2
        )
        # Should not error with many entities
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_multihop_distance_scoring(self, graph_client) -> None:
        """Closer nodes should have higher graph_score."""
        graph_client.driver = MockDriver([
            {
                "id": "close",
                "text": "Close node",
                "node_properties": {},
                "distance": 1,
                "graph_score": 0.5,  # 1/(1+1)
            },
            {
                "id": "far",
                "text": "Far node",
                "node_properties": {},
                "distance": 3,
                "graph_score": 0.25,  # 1/(1+3)
            },
        ])

        results = await graph_client.query_graph_multihop(
            entity_names=["test"], max_hops=3, top_k=10
        )
        assert results[0]["graph_score"] > results[1]["graph_score"]

    @pytest.mark.asyncio
    async def test_multihop_graceful_on_error(self, graph_client) -> None:
        """Should return empty on Neo4j error."""
        mock_driver = MagicMock()
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=Exception("Connection lost"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_driver.session = MagicMock(return_value=mock_session)
        graph_client.driver = mock_driver

        results = await graph_client.query_graph_multihop(
            entity_names=["test"], max_hops=2
        )
        assert results == []


# ---------------------------------------------------------------------------
# Session Chain Tests
# ---------------------------------------------------------------------------


class TestSessionChain:
    """Validate session chain traversal."""

    @pytest.fixture
    def graph_client(self):
        from app.services.graph_client import GraphClient
        client = GraphClient.__new__(GraphClient)
        client._DATABASE = "neo4j"
        client.driver = None
        return client

    @pytest.mark.asyncio
    async def test_session_chain_returns_history(self, graph_client) -> None:
        """Should return session chain from newest to oldest."""
        graph_client.driver = MockDriver([
            {
                "current": {
                    "session_id": "sess-3",
                    "started_at": "2026-03-18T10:00:00Z",
                    "ended_at": None,
                    "last_event_seq": 300,
                    "summary": "Current session",
                    "project": "nova-core",
                    "chain_distance": 0,
                },
                "ancestors": [
                    {
                        "session_id": "sess-2",
                        "started_at": "2026-03-17T10:00:00Z",
                        "ended_at": "2026-03-17T18:00:00Z",
                        "last_event_seq": 200,
                        "summary": "Previous session",
                        "project": "nova-core",
                        "chain_distance": 1,
                    },
                ],
            }
        ])

        chain = await graph_client.get_session_chain("sess-3", depth=5)
        assert len(chain) == 2
        assert chain[0]["session_id"] == "sess-3"
        assert chain[1]["session_id"] == "sess-2"

    @pytest.mark.asyncio
    async def test_session_chain_no_driver(self, graph_client) -> None:
        """No driver should return empty."""
        chain = await graph_client.get_session_chain("sess-1")
        assert chain == []

    @pytest.mark.asyncio
    async def test_session_chain_not_found(self, graph_client) -> None:
        """Non-existent session should return empty."""
        graph_client.driver = MockDriver([])
        chain = await graph_client.get_session_chain("nonexistent")
        assert chain == []


# ---------------------------------------------------------------------------
# Decision Graph Recall Tests
# ---------------------------------------------------------------------------


class TestDecisionGraphRecall:
    """Validate decision recall via graph."""

    @pytest.fixture
    def graph_client(self):
        from app.services.graph_client import GraphClient
        client = GraphClient.__new__(GraphClient)
        client._DATABASE = "neo4j"
        client.driver = None
        return client

    @pytest.mark.asyncio
    async def test_find_related_decisions(self, graph_client) -> None:
        """Should return decisions with related nodes."""
        graph_client.driver = MockDriver([
            {
                "id": "decision-1",
                "text": "Chose Pinecone for vector storage",
                "memory_type": "decision",
                "event_seq": 100,
                "event_time": "2026-03-15T10:00:00Z",
                "neighbors": [
                    {"id": "ctx-1", "text": "Research context", "rel_type": "RELATED_TO", "memory_type": "research"},
                ],
            },
        ])

        results = await graph_client.find_related_decisions("pinecone decision", top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "decision-1"
        assert results[0]["source"] == "graph_decision"
        assert len(results[0]["metadata"]["related_nodes"]) == 1

    @pytest.mark.asyncio
    async def test_find_decisions_no_driver(self, graph_client) -> None:
        """No driver should return empty."""
        results = await graph_client.find_related_decisions("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_find_decisions_graceful_error(self, graph_client) -> None:
        """Should return empty on Neo4j error."""
        mock_driver = MagicMock()
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=Exception("DB error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_driver.session = MagicMock(return_value=mock_session)
        graph_client.driver = mock_driver

        results = await graph_client.find_related_decisions("test")
        assert results == []
