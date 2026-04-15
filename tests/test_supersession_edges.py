"""Unit tests for Phase 5a supersession edge wiring.

Covers:
1. ``on_memory_supersede`` extended signature (reason, run_id, backward compat)
2. Supersession hook integration in ``perform_upsert`` (flag guard, non-fatal)

All tests are fully hermetic — no live Neo4j, Pinecone, or Redis required.
Every dependency is replaced with ``unittest.mock`` fakes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Make the ``app`` package importable without pulling in app.config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

from app.services.associations.memory_edges import MemoryEdge
from app.services.associations.edge_service import MemoryEdgeService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edge_service() -> tuple[MemoryEdgeService, AsyncMock]:
    """Return (service, create_edge_mock) with a mocked driver."""
    driver = MagicMock(name="AsyncDriver")
    svc = MemoryEdgeService(driver)
    svc.create_edge = AsyncMock(return_value=True)
    return svc, svc.create_edge


# ---------------------------------------------------------------------------
# 1. on_memory_supersede with reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_with_reason() -> None:
    """When ``reason`` is provided, it appears in edge metadata."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_supersede("new-1", "old-1", reason="decision changed")

    create_mock.assert_awaited_once()
    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.source_id == "new-1"
    assert edge.target_id == "old-1"
    assert edge.edge_type == "SUPERSEDES"
    assert edge.created_by == "edge_service.on_memory_supersede"
    assert edge.metadata == {"reason": "decision changed"}


# ---------------------------------------------------------------------------
# 2. on_memory_supersede with custom run_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_with_custom_run_id() -> None:
    """When ``run_id`` is provided, it overrides the default."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_supersede("new-2", "old-2", run_id="my-custom-run")

    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.run_id == "my-custom-run"


# ---------------------------------------------------------------------------
# 2b. Empty-string run_id is preserved, not replaced by default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_empty_string_run_id_rejected() -> None:
    """Empty-string run_id reaches MemoryEdge (not silently replaced)
    and is properly rejected by __post_init__ validation."""
    svc, create_mock = _make_edge_service()

    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        await svc.on_memory_supersede("new-r", "old-r", run_id="")


# ---------------------------------------------------------------------------
# 3. Backward compatibility — positional (new_id, old_id) only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_backward_compat() -> None:
    """Calling with just (new_id, old_id) still works — defaults apply."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_supersede("new-3", "old-3")

    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.run_id == "supersession_hook"
    assert edge.metadata is None
    assert edge.edge_version == 1


# ---------------------------------------------------------------------------
# 3b. Self-loop returns None without creating an edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_self_loop_returns_none() -> None:
    """Self-loop (same-id for both endpoints) returns None without creating an edge."""
    svc, create_mock = _make_edge_service()

    result = await svc.on_memory_supersede("same-id", "same-id")

    assert result is None
    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Hook skipped when feature flag is off
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersession_hook_skipped_when_flag_off() -> None:
    """When ASSOC_PROVENANCE_WRITE_ENABLED is False, no edge service
    is constructed and no supersession edges are created."""

    mock_persist = AsyncMock(return_value=True)
    mock_embedding = MagicMock(return_value=[0.1] * 8)
    mock_inject = AsyncMock(side_effect=lambda m: m)

    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", mock_embedding), \
         patch("app.services.memory_service.detect_conflicts", new_callable=AsyncMock) as mock_detect, \
         patch("app.services.memory_service.check_semantic_duplicate", new_callable=AsyncMock, return_value=None):

        mock_settings.ASSOC_PROVENANCE_WRITE_ENABLED = False
        mock_settings.ASSOC_SIMILARITY_WRITE_ENABLED = False
        mock_settings.ASSOC_ENTITY_WRITE_ENABLED = False
        mock_settings.ASSOC_TEMPORAL_WRITE_ENABLED = False
        mock_settings.WRITE_DEDUP_ENABLED = False
        mock_settings.CONFLICT_DETECTION_ENABLED = True
        mock_settings.EMBEDDING_MODEL = "test"

        from app.services.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)
        svc._initialized = True
        svc._similarity_linker = None
        svc._entity_linker = None
        svc._temporal_linker = None
        svc._provenance_edge_service = None
        svc.redis_timeline = None
        svc.embedding_model_name = "test"
        svc.pinecone_client = MagicMock()
        svc.graph_client = MagicMock()
        svc.sequence_service = MagicMock()
        svc.sequence_service.next_seq = AsyncMock(return_value=42)
        svc._persist_memory_item = mock_persist
        svc._inject_chronology = mock_inject
        svc._resolve_memory_id = MagicMock(return_value="test-id")

        # conflict_info would be populated but flag is off
        mock_detect.return_value = [
            {"id": "conflict-1", "similarity": 0.85, "text": "old decision"}
        ]

        result = await svc.perform_upsert(
            content="new decision",
            metadata={"category": "decision"},
        )

        assert result == "test-id"
        # The provenance edge service should never have been constructed
        assert svc._provenance_edge_service is None


# ---------------------------------------------------------------------------
# 5. Hook is non-fatal — exceptions do not break perform_upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersession_hook_nonfatal() -> None:
    """If on_memory_supersede raises, perform_upsert still returns the
    item ID successfully — the hook is fail-open."""

    mock_persist = AsyncMock(return_value=True)
    mock_embedding = MagicMock(return_value=[0.1] * 8)
    mock_inject = AsyncMock(side_effect=lambda m: m)

    boom_edge_service = MagicMock()
    boom_edge_service.on_memory_supersede = AsyncMock(
        side_effect=RuntimeError("Neo4j exploded")
    )

    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", mock_embedding), \
         patch("app.services.memory_service.detect_conflicts", new_callable=AsyncMock) as mock_detect, \
         patch("app.services.memory_service.check_semantic_duplicate", new_callable=AsyncMock, return_value=None):

        mock_settings.ASSOC_PROVENANCE_WRITE_ENABLED = True
        mock_settings.ASSOC_SIMILARITY_WRITE_ENABLED = False
        mock_settings.ASSOC_ENTITY_WRITE_ENABLED = False
        mock_settings.ASSOC_TEMPORAL_WRITE_ENABLED = False
        mock_settings.WRITE_DEDUP_ENABLED = False
        mock_settings.CONFLICT_DETECTION_ENABLED = True
        mock_settings.EMBEDDING_MODEL = "test"

        from app.services.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)
        svc._initialized = True
        svc._similarity_linker = None
        svc._entity_linker = None
        svc._temporal_linker = None
        svc._provenance_edge_service = boom_edge_service
        svc.redis_timeline = None
        svc.embedding_model_name = "test"
        svc.pinecone_client = MagicMock()
        svc.graph_client = MagicMock()
        svc.sequence_service = MagicMock()
        svc.sequence_service.next_seq = AsyncMock(return_value=42)
        svc._persist_memory_item = mock_persist
        svc._inject_chronology = mock_inject
        svc._resolve_memory_id = MagicMock(return_value="test-id")

        mock_detect.return_value = [
            {"id": "conflict-1", "similarity": 0.85, "text": "old decision"}
        ]

        result = await svc.perform_upsert(
            content="new decision",
            metadata={"category": "decision"},
        )

        # Despite the exception, upsert succeeds
        assert result == "test-id"
        # The edge service WAS called (flag was on, conflicts existed)
        boom_edge_service.on_memory_supersede.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. Happy-path integration — flag ON, two conflicts, edges created
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersession_hook_happy_path() -> None:
    """When the flag is ON and detect_conflicts returns two conflicts,
    on_memory_supersede is called exactly twice with correct arguments."""

    mock_persist = AsyncMock(return_value=True)
    mock_embedding = MagicMock(return_value=[0.1] * 8)
    mock_inject = AsyncMock(side_effect=lambda m: m)

    happy_edge_service = MagicMock()
    happy_edge_service.on_memory_supersede = AsyncMock(return_value=None)

    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", mock_embedding), \
         patch("app.services.memory_service.detect_conflicts", new_callable=AsyncMock) as mock_detect, \
         patch("app.services.memory_service.check_semantic_duplicate", new_callable=AsyncMock, return_value=None):

        mock_settings.ASSOC_PROVENANCE_WRITE_ENABLED = True
        mock_settings.ASSOC_SIMILARITY_WRITE_ENABLED = False
        mock_settings.ASSOC_ENTITY_WRITE_ENABLED = False
        mock_settings.ASSOC_TEMPORAL_WRITE_ENABLED = False
        mock_settings.WRITE_DEDUP_ENABLED = False
        mock_settings.CONFLICT_DETECTION_ENABLED = True
        mock_settings.EMBEDDING_MODEL = "test"

        from app.services.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)
        svc._initialized = True
        svc._similarity_linker = None
        svc._entity_linker = None
        svc._temporal_linker = None
        svc._provenance_edge_service = happy_edge_service
        svc.redis_timeline = None
        svc.embedding_model_name = "test"
        svc.pinecone_client = MagicMock()
        svc.graph_client = MagicMock()
        svc.sequence_service = MagicMock()
        svc.sequence_service.next_seq = AsyncMock(return_value=42)
        svc._persist_memory_item = mock_persist
        svc._inject_chronology = mock_inject
        svc._resolve_memory_id = MagicMock(return_value="test-id")

        mock_detect.return_value = [
            {"id": "conflict-A", "similarity": 0.91, "text": "old decision A"},
            {"id": "conflict-B", "similarity": 0.87, "text": "old decision B"},
        ]

        result = await svc.perform_upsert(
            content="new decision",
            metadata={"category": "decision", "session_id": "sess-42"},
        )

        assert result == "test-id"

        # Exactly two calls — one per conflict
        assert happy_edge_service.on_memory_supersede.await_count == 2

        calls = happy_edge_service.on_memory_supersede.call_args_list
        # First conflict
        assert calls[0].kwargs["new_id"] == "test-id"
        assert calls[0].kwargs["old_id"] == "conflict-A"
        assert "0.910" in calls[0].kwargs["reason"]
        assert calls[0].kwargs["run_id"] == "sess-42"
        # Second conflict
        assert calls[1].kwargs["new_id"] == "test-id"
        assert calls[1].kwargs["old_id"] == "conflict-B"
        assert "0.870" in calls[1].kwargs["reason"]
        assert calls[1].kwargs["run_id"] == "sess-42"
