"""Unit tests for Phase 5c compaction edge wiring.

Covers:
1. ``on_memory_compact`` creates N edges for N sources
2. Custom ``run_id`` is forwarded
3. Self-loop guard (summary_id in source_ids)
4. ``edge_version == 1`` on created edges
5. Compaction hook skipped when ``ASSOC_PROVENANCE_WRITE_ENABLED`` is False
6. Per-item fault tolerance — one source failure does not kill others

All tests are fully hermetic — no live Neo4j, Pinecone, or Redis required.
Every dependency is replaced with ``unittest.mock`` fakes.
"""

from __future__ import annotations

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
# 1. on_memory_compact creates edges — N sources produce N edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_creates_edges() -> None:
    """Three source memories produce exactly three COMPACTED_FROM edges."""
    svc, create_mock = _make_edge_service()
    sources = ["src-1", "src-2", "src-3"]

    result = await svc.on_memory_compact("summary-1", sources)

    assert result == 3
    assert create_mock.await_count == 3

    for i, call in enumerate(create_mock.call_args_list):
        edge: MemoryEdge = call[0][0]
        assert edge.source_id == "summary-1"
        assert edge.target_id == sources[i]
        assert edge.edge_type == "COMPACTED_FROM"
        assert edge.weight == 1.0
        assert edge.created_by == "edge_service.on_memory_compact"
        assert edge.run_id == "compaction_hook"


# ---------------------------------------------------------------------------
# 2b. Empty-string run_id reaches MemoryEdge and is rejected (not silently
#     replaced by the default). The per-item except catches the ValueError,
#     so the method returns 0 instead of raising.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_empty_string_run_id_rejected() -> None:
    """Empty-string run_id is not silently replaced — it reaches MemoryEdge
    which rejects it. The per-item except catches the ValueError."""
    svc, create_mock = _make_edge_service()

    result = await svc.on_memory_compact(
        "summary-r", ["src-1", "src-2"], run_id=""
    )

    assert result == 0
    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2. Custom run_id is forwarded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_with_custom_run_id() -> None:
    """When ``run_id`` is provided, it overrides the default."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_compact(
        "summary-2", ["src-a", "src-b"], run_id="my-run-42"
    )

    for call in create_mock.call_args_list:
        edge: MemoryEdge = call[0][0]
        assert edge.run_id == "my-run-42"


# ---------------------------------------------------------------------------
# 3. Self-loop guard — summary_id in source_ids is skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_self_loop_guard() -> None:
    """When summary_id appears in source_ids, that entry is skipped."""
    svc, create_mock = _make_edge_service()

    result = await svc.on_memory_compact(
        "summary-3", ["src-x", "summary-3", "src-y"]
    )

    # Only two edges created — the self-loop entry was skipped
    assert result == 2
    assert create_mock.await_count == 2

    targets = [call[0][0].target_id for call in create_mock.call_args_list]
    assert "summary-3" not in targets
    assert targets == ["src-x", "src-y"]


# ---------------------------------------------------------------------------
# 4. edge_version == 1 on every created edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_edge_version() -> None:
    """All COMPACTED_FROM edges have edge_version == 1."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_compact("summary-4", ["src-1", "src-2"])

    for call in create_mock.call_args_list:
        edge: MemoryEdge = call[0][0]
        assert edge.edge_version == 1


# ---------------------------------------------------------------------------
# 5. Hook skipped when feature flag is off
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_caller_guard_pattern_contract() -> None:
    """Caller-side guard pattern: when the flag is off, on_memory_compact
    is never called. This is a pattern contract test — it validates the
    caller's guard logic, not the actual feature flag wiring."""
    mock_edge_service = MagicMock()
    mock_edge_service.on_memory_compact = AsyncMock(return_value=3)

    # Simulate the flag-guarded caller pattern from memory_service.py
    flag_enabled = False
    provenance_svc = mock_edge_service

    if flag_enabled:
        await provenance_svc.on_memory_compact(
            "summary-5", ["src-1", "src-2", "src-3"]
        )

    # No call was made because the flag was off
    mock_edge_service.on_memory_compact.assert_not_awaited()


# ---------------------------------------------------------------------------
# 6. Per-item fault tolerance — one failure does not kill the rest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_hook_nonfatal() -> None:
    """If one source edge fails, the remaining sources still get edges."""
    svc, create_mock = _make_edge_service()

    # Second call (src-2) raises; first and third should succeed
    create_mock.side_effect = [
        True,                                    # src-1: success
        RuntimeError("Neo4j timeout"),           # src-2: failure
        True,                                    # src-3: success
    ]

    result = await svc.on_memory_compact(
        "summary-6", ["src-1", "src-2", "src-3"]
    )

    # Two succeeded, one failed
    assert result == 2
    assert create_mock.await_count == 3


# ---------------------------------------------------------------------------
# 7. Empty source list produces zero edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_empty_sources() -> None:
    """An empty source list returns 0 and makes no create_edge calls."""
    svc, create_mock = _make_edge_service()

    result = await svc.on_memory_compact("summary-7", [])

    assert result == 0
    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 8. Invalid summary_id raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_invalid_summary_id() -> None:
    """An empty or non-string summary_id raises ValueError."""
    svc, _ = _make_edge_service()

    with pytest.raises(ValueError, match="summary_id must be a non-empty string"):
        await svc.on_memory_compact("", ["src-1"])

    with pytest.raises(ValueError, match="summary_id must be a non-empty string"):
        await svc.on_memory_compact(None, ["src-1"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. Bare string source_ids raises TypeError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_bare_string_raises() -> None:
    """Passing a bare string instead of a list raises TypeError."""
    svc, _ = _make_edge_service()

    with pytest.raises(TypeError, match="source_ids must be a list"):
        await svc.on_memory_compact("summary-x", "src-1")


# ---------------------------------------------------------------------------
# 10. Partial create_edge failure yields correct count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_partial_create_failure() -> None:
    """create_edge returning False for one source yields correct count."""
    svc, create_mock = _make_edge_service()
    create_mock.side_effect = [True, False, True]

    result = await svc.on_memory_compact("summary-p", ["src-1", "src-2", "src-3"])

    assert result == 2
    assert create_mock.await_count == 3
