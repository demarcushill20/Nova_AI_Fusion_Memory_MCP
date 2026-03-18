"""Tests for write-time deduplication and conflict detection (P9A.5).

Validates:
- Exact duplicate detection (same content → skip)
- Semantic duplicate detection (paraphrased → detected)
- Non-duplicate passes normally
- Conflict detection for decision-category memories
- on_duplicate behavior modes (skip/update/append/conflict)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.conflict_detector import (
    CONFLICT_THRESHOLD_HIGH,
    CONFLICT_THRESHOLD_LOW,
    DUPLICATE_THRESHOLD,
    check_semantic_duplicate,
    detect_conflicts,
    resolve_duplicate_action,
)


# ---------------------------------------------------------------------------
# Mock Pinecone client
# ---------------------------------------------------------------------------


class MockPineconeClient:
    """Fake Pinecone client that returns configurable results."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        self._results = results or []

    def query_vector(
        self,
        query_vector: List[float],
        top_k: int = 3,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self._results


# ---------------------------------------------------------------------------
# Semantic duplicate detection tests
# ---------------------------------------------------------------------------


class TestSemanticDuplicateCheck:
    """Validate write-time semantic dedup."""

    @pytest.mark.asyncio
    async def test_exact_duplicate_detected(self) -> None:
        """Content with similarity >= 0.92 should be flagged as duplicate."""
        client = MockPineconeClient(results=[
            {
                "id": "existing-123",
                "score": 0.95,
                "metadata": {
                    "text": "We decided to use Pinecone for vector storage",
                    "event_time": "2026-03-18T10:00:00Z",
                    "memory_type": "decision",
                },
            }
        ])
        result = await check_semantic_duplicate(
            client, "We decided to use Pinecone for vectors", [0.1] * 10
        )
        assert result is not None
        assert result["id"] == "existing-123"
        assert result["score"] == 0.95

    @pytest.mark.asyncio
    async def test_semantic_duplicate_detected(self) -> None:
        """Paraphrased content above threshold should be detected."""
        client = MockPineconeClient(results=[
            {
                "id": "existing-456",
                "score": 0.93,
                "metadata": {"text": "Chose Pinecone as our vector DB", "event_time": "2026-03-17T08:00:00Z"},
            }
        ])
        result = await check_semantic_duplicate(
            client, "Selected Pinecone for the vector database", [0.1] * 10
        )
        assert result is not None
        assert result["id"] == "existing-456"

    @pytest.mark.asyncio
    async def test_non_duplicate_passes(self) -> None:
        """Content below threshold should not be flagged."""
        client = MockPineconeClient(results=[
            {
                "id": "different-789",
                "score": 0.65,
                "metadata": {"text": "Set up Redis for caching"},
            }
        ])
        result = await check_semantic_duplicate(
            client, "We use Pinecone for vectors", [0.1] * 10
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_results_returns_none(self) -> None:
        """Empty results should not flag duplicate."""
        client = MockPineconeClient(results=[])
        result = await check_semantic_duplicate(client, "New content", [0.1] * 10)
        assert result is None

    @pytest.mark.asyncio
    async def test_exclude_self_on_update(self) -> None:
        """When updating, exclude the item's own ID from duplicate check."""
        client = MockPineconeClient(results=[
            {
                "id": "self-id",
                "score": 0.99,
                "metadata": {"text": "Same content"},
            }
        ])
        result = await check_semantic_duplicate(
            client, "Same content", [0.1] * 10, exclude_id="self-id"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_custom_threshold(self) -> None:
        """Custom threshold should be respected."""
        client = MockPineconeClient(results=[
            {"id": "close-match", "score": 0.85, "metadata": {"text": "Close match"}},
        ])
        # Default threshold (0.92) → not duplicate
        result = await check_semantic_duplicate(client, "Test", [0.1] * 10)
        assert result is None

        # Lower threshold (0.80) → duplicate
        result = await check_semantic_duplicate(
            client, "Test", [0.1] * 10, threshold=0.80
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_graceful_on_error(self) -> None:
        """If Pinecone query fails, return None (don't block upsert)."""
        client = MockPineconeClient()
        client.query_vector = MagicMock(side_effect=Exception("Connection failed"))
        result = await check_semantic_duplicate(client, "Test", [0.1] * 10)
        assert result is None


# ---------------------------------------------------------------------------
# Conflict detection tests
# ---------------------------------------------------------------------------


class TestConflictDetection:
    """Validate decision conflict detection."""

    @pytest.mark.asyncio
    async def test_conflict_detected(self) -> None:
        """Similar (0.70-0.92) decision should be flagged as conflict."""
        client = MockPineconeClient(results=[
            {
                "id": "old-decision",
                "score": 0.80,
                "metadata": {
                    "text": "We chose Memcached for caching",
                    "event_time": "2026-03-10T12:00:00Z",
                    "category": "decision",
                },
            }
        ])
        conflicts = await detect_conflicts(
            client, "We now use Redis for caching", [0.1] * 10
        )
        assert len(conflicts) == 1
        assert conflicts[0]["id"] == "old-decision"
        assert conflicts[0]["similarity"] == 0.80

    @pytest.mark.asyncio
    async def test_duplicate_not_conflict(self) -> None:
        """Very similar (>= 0.92) should not be flagged as conflict (it's a duplicate)."""
        client = MockPineconeClient(results=[
            {"id": "dup", "score": 0.95, "metadata": {"text": "Same decision"}},
        ])
        conflicts = await detect_conflicts(client, "Same decision", [0.1] * 10)
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_dissimilar_not_conflict(self) -> None:
        """Low similarity (<= 0.70) should not be flagged."""
        client = MockPineconeClient(results=[
            {"id": "unrelated", "score": 0.40, "metadata": {"text": "Unrelated"}},
        ])
        conflicts = await detect_conflicts(client, "New decision", [0.1] * 10)
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_multiple_conflicts(self) -> None:
        """Multiple conflicting memories should all be returned."""
        client = MockPineconeClient(results=[
            {"id": "conflict-1", "score": 0.75, "metadata": {"text": "Option A"}},
            {"id": "conflict-2", "score": 0.82, "metadata": {"text": "Option B"}},
            {"id": "unrelated", "score": 0.30, "metadata": {"text": "Irrelevant"}},
        ])
        conflicts = await detect_conflicts(client, "New choice", [0.1] * 10)
        assert len(conflicts) == 2

    @pytest.mark.asyncio
    async def test_graceful_on_error(self) -> None:
        """If detection fails, return empty list (don't block upsert)."""
        client = MockPineconeClient()
        client.query_vector = MagicMock(side_effect=Exception("API error"))
        conflicts = await detect_conflicts(client, "Test", [0.1] * 10)
        assert conflicts == []


# ---------------------------------------------------------------------------
# Duplicate action resolution tests
# ---------------------------------------------------------------------------


class TestResolveDuplicateAction:
    """Validate on_duplicate behavior resolution."""

    def test_no_duplicate_returns_append(self) -> None:
        assert resolve_duplicate_action(None) == "append"

    def test_explicit_skip(self) -> None:
        dup = {"id": "x", "score": 0.95}
        assert resolve_duplicate_action(dup, on_duplicate="skip") == "skip"

    def test_explicit_update(self) -> None:
        dup = {"id": "x", "score": 0.95}
        assert resolve_duplicate_action(dup, on_duplicate="update") == "update"

    def test_explicit_append(self) -> None:
        dup = {"id": "x", "score": 0.95}
        assert resolve_duplicate_action(dup, on_duplicate="append") == "append"

    def test_explicit_conflict(self) -> None:
        dup = {"id": "x", "score": 0.95}
        assert resolve_duplicate_action(dup, on_duplicate="conflict") == "conflict"

    def test_auto_debug_skips(self) -> None:
        dup = {"id": "x", "score": 0.95}
        assert resolve_duplicate_action(dup, on_duplicate="auto", category="debug") == "skip"

    def test_auto_decision_updates(self) -> None:
        dup = {"id": "x", "score": 0.95}
        assert resolve_duplicate_action(dup, on_duplicate="auto", category="decision") == "update"

    def test_auto_research_appends(self) -> None:
        dup = {"id": "x", "score": 0.95}
        assert resolve_duplicate_action(dup, on_duplicate="auto", category="research") == "append"

    def test_auto_unknown_category_skips(self) -> None:
        dup = {"id": "x", "score": 0.95}
        assert resolve_duplicate_action(dup, on_duplicate="auto", category="unknown") == "skip"


# ---------------------------------------------------------------------------
# Threshold boundary tests
# ---------------------------------------------------------------------------


class TestThresholdBoundaries:
    """Validate threshold constants are consistent."""

    def test_duplicate_threshold_value(self) -> None:
        assert DUPLICATE_THRESHOLD == 0.92

    def test_conflict_range(self) -> None:
        assert CONFLICT_THRESHOLD_LOW < CONFLICT_THRESHOLD_HIGH
        assert CONFLICT_THRESHOLD_LOW == 0.70
        assert CONFLICT_THRESHOLD_HIGH == 0.92

    def test_no_overlap_gap(self) -> None:
        """Conflict range should end where duplicate range begins."""
        assert CONFLICT_THRESHOLD_HIGH == DUPLICATE_THRESHOLD
