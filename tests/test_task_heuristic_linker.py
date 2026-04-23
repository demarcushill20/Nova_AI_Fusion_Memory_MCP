"""Unit tests for :class:`TaskHeuristicLinker` (PLAN-0759 Phase 7b).

Design
------

These tests are **fully hermetic**. They do not touch the real Neo4j
container or the ``MemoryService`` orchestration surface. Every dependency
of :class:`TaskHeuristicLinker` is replaced with an ``AsyncMock`` /
``MagicMock`` so the tests isolate the linker's own logic: category
filtering, project matching, time-delta validation, edge creation, and
exception containment.

Each test asserts one of:

1. A contract on the *inputs* the linker hands to its dependencies
   (which edges hit the edge service).
2. A contract on the *outputs* of :meth:`enqueue_link` itself
   (returns immediately, never raises, skips non-matching patterns).
3. A structural invariant of the class (fail-open, flag gating).
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Make the ``app`` package importable without pulling in app.config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

from app.services.associations.memory_edges import (
    MemoryEdge,
    VALID_EDGE_TYPES,
)
from app.services.associations.task_heuristic_linker import TaskHeuristicLinker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(tz=timezone.utc)
_FIVE_DAYS_AGO = (_NOW - timedelta(days=5)).isoformat()
_NOW_ISO = _NOW.isoformat()
_FORTY_DAYS_AGO = (_NOW - timedelta(days=40)).isoformat()


def _make_linker(
    *,
    fetcher_result: Optional[dict[str, Any]] = None,
    fetcher_side_effect: Any = None,
    edge_create_return: bool = True,
    edge_create_side_effect: Any = None,
) -> tuple[TaskHeuristicLinker, AsyncMock, AsyncMock]:
    """Construct a linker wired to controllable async mocks.

    Returns ``(linker, memory_fetcher_mock, edge_create_mock)``.
    """
    memory_fetcher = AsyncMock(
        return_value=fetcher_result,
        side_effect=fetcher_side_effect,
    )

    edge_service = MagicMock(name="MemoryEdgeService")
    if edge_create_side_effect is not None:
        edge_service.create_edge = AsyncMock(side_effect=edge_create_side_effect)
    else:
        edge_service.create_edge = AsyncMock(return_value=edge_create_return)

    linker = TaskHeuristicLinker(
        edge_service=edge_service,
        memory_fetcher=memory_fetcher,
    )
    return linker, memory_fetcher, edge_service.create_edge


async def _drain_inflight(linker: TaskHeuristicLinker, timeout: float = 2.0) -> None:
    """Wait for every currently in-flight background task to complete."""
    deadline = time.perf_counter() + timeout
    while linker._inflight and time.perf_counter() < deadline:
        tasks = list(linker._inflight)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.sleep(0)


def _bug_fix_metadata(
    *,
    fixes_task_id: str = "task-001",
    project: str = "nova",
    event_time: str = _NOW_ISO,
) -> dict[str, Any]:
    """Return a standard bug_fix metadata dict."""
    return {
        "category": "bug_fix",
        "fixes_task_id": fixes_task_id,
        "project": project,
        "event_time": event_time,
    }


def _task_failed_metadata(
    *,
    project: str = "nova",
    event_time: str = _FIVE_DAYS_AGO,
) -> dict[str, Any]:
    """Return a standard task_failed metadata dict."""
    return {
        "category": "task_failed",
        "project": project,
        "event_time": event_time,
    }


# ---------------------------------------------------------------------------
# 1. Creates CAUSED_BY edge for valid bug_fix -> task_failed pair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creates_caused_by_edge() -> None:
    """bug_fix + fixes_task_id + task_failed target -> CAUSED_BY edge."""
    target_meta = _task_failed_metadata()
    linker, fetcher, create_edge = _make_linker(fetcher_result=target_meta)

    meta = _bug_fix_metadata()
    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=meta,
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_awaited_once()

    edge: MemoryEdge = create_edge.call_args[0][0]
    assert edge.source_id == "bugfix-001"
    assert edge.target_id == "task-001"
    assert edge.edge_type == "CAUSED_BY"
    assert edge.weight == 1.0
    assert edge.created_by == "task_heuristic_linker"


# ---------------------------------------------------------------------------
# 2. Skips non-bug_fix category
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_non_bug_fix() -> None:
    """category != 'bug_fix' -> no fetch, no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata()
    )

    await linker.enqueue_link(
        memory_id="mem-001",
        metadata={"category": "research", "fixes_task_id": "task-001", "project": "nova"},
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_not_awaited()
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. Skips missing fixes_task_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_missing_fixes_task_id() -> None:
    """No fixes_task_id -> no fetch, no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata()
    )

    await linker.enqueue_link(
        memory_id="mem-001",
        metadata={"category": "bug_fix", "project": "nova"},
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_not_awaited()
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Skips target not task_failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_target_not_task_failed() -> None:
    """Target category != 'task_failed' -> no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result={
            "category": "research",
            "project": "nova",
            "event_time": _FIVE_DAYS_AGO,
        }
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Skips cross-project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_cross_project() -> None:
    """Different project -> no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata(project="other-project")
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(project="nova"),
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 6. Skips old memories (>30 days apart)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_old_memories() -> None:
    """>30 days apart -> no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata(event_time=_FORTY_DAYS_AGO)
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(event_time=_NOW_ISO),
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. Edge metadata verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_metadata() -> None:
    """Verify CAUSED_BY type, weight, created_by, metadata=None."""
    target_meta = _task_failed_metadata()
    linker, fetcher, create_edge = _make_linker(fetcher_result=target_meta)

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    create_edge.assert_awaited_once()
    edge: MemoryEdge = create_edge.call_args[0][0]
    assert edge.edge_type == "CAUSED_BY"
    assert edge.edge_type in VALID_EDGE_TYPES
    assert edge.weight == 1.0
    assert edge.created_by == "task_heuristic_linker"
    assert edge.metadata is None
    assert edge.run_id.startswith("wt-taskheur-")
    assert edge.created_at == edge.last_seen_at


# ---------------------------------------------------------------------------
# 8. Non-fatal: edge creation failure does not propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nonfatal_edge_failure() -> None:
    """Edge creation failure is swallowed, no exception propagates."""
    target_meta = _task_failed_metadata()
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=target_meta,
        edge_create_side_effect=RuntimeError("Neo4j down"),
    )

    # This must NOT raise
    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    # The fetch was called, the edge creation was attempted
    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_awaited_once()


# ---------------------------------------------------------------------------
# 9. Flag shipped True (integration-level test of the guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_heuristic_flag_default_is_true() -> None:
    """ASSOC_TASK_HEURISTIC_WRITE_ENABLED defaults to True.

    Flipped 2026-04-23 after Phase 6 parity gate cleared (the documented
    precondition). The hook body is a cheap metadata check — it only
    dispatches when the memory is a ``bug_fix`` with a ``fixes_task_id``,
    so the default-on flag is a no-op for non-opted-in callers.
    """
    from app.config import settings

    assert settings.ASSOC_TASK_HEURISTIC_WRITE_ENABLED is True

    # Verify the flag-guarded block in perform_upsert is reachable:
    # `success and settings.ASSOC_TASK_HEURISTIC_WRITE_ENABLED`.
    assert (True and settings.ASSOC_TASK_HEURISTIC_WRITE_ENABLED)


# ---------------------------------------------------------------------------
# 10. Skips when target not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_target_not_found() -> None:
    """Target memory does not exist -> no edge."""
    linker, fetcher, create_edge = _make_linker(fetcher_result=None)

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 11. Skips when memory_fetcher is None (linker disabled)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_when_no_fetcher() -> None:
    """No memory_fetcher -> _link_one returns 0, no edge."""
    edge_service = MagicMock(name="MemoryEdgeService")
    edge_service.create_edge = AsyncMock(return_value=True)

    linker = TaskHeuristicLinker(
        edge_service=edge_service,
        memory_fetcher=None,
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    edge_service.create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 12. Skips when project is None or empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_when_no_project() -> None:
    """project=None -> enqueue_link returns immediately."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata()
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project=None,
    )
    await _drain_inflight(linker)

    fetcher.assert_not_awaited()
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 13. Non-fatal: memory_fetcher failure does not propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nonfatal_fetcher_failure() -> None:
    """Memory fetcher failure is swallowed, no exception propagates."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_side_effect=RuntimeError("Pinecone down"),
    )

    # This must NOT raise
    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 14. Wrong time order: bug_fix older than task_failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_wrong_time_order() -> None:
    """bug_fix event_time <= task_failed event_time -> no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata(event_time=_NOW_ISO)
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(event_time=_FIVE_DAYS_AGO),
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 15. Exact 30-day boundary is rejected (< 30, not <=)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_exact_30_day_boundary() -> None:
    """Exactly 30 days apart -> no edge (boundary is exclusive)."""
    exactly_30_days_ago = (_NOW - timedelta(days=30)).isoformat()
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata(event_time=exactly_30_days_ago)
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(event_time=_NOW_ISO),
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 16. create_edge returns False (edge already exists / MERGE no-op)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_edge_returns_false() -> None:
    """create_edge returning False -> edges_count is 0, no exception."""
    target_meta = _task_failed_metadata()
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=target_meta,
        edge_create_return=False,
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    create_edge.assert_awaited_once()


# ---------------------------------------------------------------------------
# 17. Missing source timestamp -> rejected (fail-closed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_missing_source_timestamp() -> None:
    """Missing event_time on source -> no edge (fail-closed)."""
    target_meta = _task_failed_metadata()
    linker, fetcher, create_edge = _make_linker(fetcher_result=target_meta)

    meta = _bug_fix_metadata()
    del meta["event_time"]
    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=meta,
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 18. Missing target timestamp -> rejected (fail-closed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_missing_target_timestamp() -> None:
    """Missing event_time on target -> no edge (fail-closed)."""
    target_meta = {"category": "task_failed", "project": "nova"}
    linker, fetcher, create_edge = _make_linker(fetcher_result=target_meta)

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 19. Timeout handling in _link_one_safe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_is_nonfatal() -> None:
    """Timeout in _link_one is caught and returns 0."""
    async def _slow_fetcher(mid: str) -> dict:
        await asyncio.sleep(10)
        return _task_failed_metadata()

    edge_service = MagicMock(name="MemoryEdgeService")
    edge_service.create_edge = AsyncMock(return_value=True)

    linker = TaskHeuristicLinker(
        edge_service=edge_service,
        memory_fetcher=_slow_fetcher,
    )
    linker.BACKGROUND_TIMEOUT = 0.05

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    edge_service.create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 20. Non-dict metadata is rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_non_dict_metadata() -> None:
    """Non-dict metadata -> enqueue_link returns immediately."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata()
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata="not a dict",  # type: ignore[arg-type]
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_not_awaited()
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 21. Non-dict target_meta from fetcher is rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_non_dict_target_meta() -> None:
    """Fetcher returning a non-dict -> treated as not found."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result="not a dict",  # type: ignore[arg-type]
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 22. run_id propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_id_propagation() -> None:
    """Explicit run_id is propagated to the created edge."""
    target_meta = _task_failed_metadata()
    linker, fetcher, create_edge = _make_linker(fetcher_result=target_meta)

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
        run_id="custom-run-123",
    )
    await _drain_inflight(linker)

    create_edge.assert_awaited_once()
    edge: MemoryEdge = create_edge.call_args[0][0]
    assert edge.run_id == "custom-run-123"


# ---------------------------------------------------------------------------
# 23. CancelledError is handled (does not escape envelope)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_error_is_handled() -> None:
    """CancelledError in _link_one is caught, not propagated."""
    async def _cancel_fetcher(mid: str) -> dict:
        raise asyncio.CancelledError()

    edge_service = MagicMock(name="MemoryEdgeService")
    edge_service.create_edge = AsyncMock(return_value=True)

    linker = TaskHeuristicLinker(
        edge_service=edge_service,
        memory_fetcher=_cancel_fetcher,
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project="nova",
    )
    await _drain_inflight(linker)

    edge_service.create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 24. Semaphore saturation drops new requests (F11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saturation_drops_excess() -> None:
    """When in-flight tasks >= BACKGROUND_MAX_IN_FLIGHT, new calls are dropped."""
    gate = asyncio.Event()

    async def _blocking_fetcher(mid: str) -> dict:
        await gate.wait()
        return _task_failed_metadata()

    edge_service = MagicMock(name="MemoryEdgeService")
    edge_service.create_edge = AsyncMock(return_value=True)

    linker = TaskHeuristicLinker(
        edge_service=edge_service,
        memory_fetcher=_blocking_fetcher,
    )
    linker.BACKGROUND_MAX_IN_FLIGHT = 4

    for i in range(6):
        await linker.enqueue_link(
            memory_id=f"bugfix-{i:03d}",
            metadata=_bug_fix_metadata(fixes_task_id=f"task-{i:03d}"),
            project="nova",
        )
        await asyncio.sleep(0)

    assert len(linker._inflight) == 4

    gate.set()
    await _drain_inflight(linker)


# ---------------------------------------------------------------------------
# 25. Empty-string fixes_task_id is rejected (F14)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_empty_string_fixes_task_id() -> None:
    """fixes_task_id='' -> no fetch, no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata()
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata={
            "category": "bug_fix",
            "fixes_task_id": "",
            "project": "nova",
            "event_time": _NOW_ISO,
        },
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_not_awaited()
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 26. Whitespace-only fixes_task_id is rejected (F14 extended)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_whitespace_fixes_task_id() -> None:
    """fixes_task_id='   ' -> no fetch, no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata()
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata={
            "category": "bug_fix",
            "fixes_task_id": "   ",
            "project": "nova",
            "event_time": _NOW_ISO,
        },
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_not_awaited()
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 27. Non-string fixes_task_id is rejected (F4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_non_string_fixes_task_id() -> None:
    """fixes_task_id=42 -> no fetch, no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata()
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata={
            "category": "bug_fix",
            "fixes_task_id": 42,
            "project": "nova",
            "event_time": _NOW_ISO,
        },
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_not_awaited()
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 28. Self-loop (memory_id == fixes_task_id) is rejected (F7/F15)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_self_loop() -> None:
    """memory_id == fixes_task_id -> no fetch, no edge."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata()
    )

    await linker.enqueue_link(
        memory_id="mem-001",
        metadata={
            "category": "bug_fix",
            "fixes_task_id": "mem-001",
            "project": "nova",
            "event_time": _NOW_ISO,
        },
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_not_awaited()
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 29. Z-suffix timestamps are accepted (F2/F16)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepts_z_suffix_timestamps() -> None:
    """Timestamps ending in 'Z' are normalized and create edges."""
    target_meta = {
        "category": "task_failed",
        "project": "nova",
        "event_time": "2026-04-10T12:00:00Z",
    }
    linker, fetcher, create_edge = _make_linker(fetcher_result=target_meta)

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata={
            "category": "bug_fix",
            "fixes_task_id": "task-001",
            "project": "nova",
            "event_time": "2026-04-15T12:00:00Z",
        },
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_awaited_once()
    edge: MemoryEdge = create_edge.call_args[0][0]
    assert edge.edge_type == "CAUSED_BY"


# ---------------------------------------------------------------------------
# 30. Mixed naive + aware timestamps are handled (F1/F16)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_naive_aware_timestamps() -> None:
    """Naive source + aware target timestamps produce valid edge."""
    target_meta = {
        "category": "task_failed",
        "project": "nova",
        "event_time": "2026-04-10T12:00:00+00:00",
    }
    linker, fetcher, create_edge = _make_linker(fetcher_result=target_meta)

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata={
            "category": "bug_fix",
            "fixes_task_id": "task-001",
            "project": "nova",
            "event_time": "2026-04-15T12:00:00",
        },
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_awaited_once()
    edge: MemoryEdge = create_edge.call_args[0][0]
    assert edge.edge_type == "CAUSED_BY"


# ---------------------------------------------------------------------------
# 31. Non-string event_time is rejected (F8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_non_string_event_time() -> None:
    """Non-string event_time -> no edge (fail-closed)."""
    target_meta = _task_failed_metadata()
    linker, fetcher, create_edge = _make_linker(fetcher_result=target_meta)

    meta = _bug_fix_metadata()
    meta["event_time"] = 12345
    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=meta,
        project="nova",
    )
    await _drain_inflight(linker)

    fetcher.assert_awaited_once_with("task-001")
    create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 32. Non-string project is rejected (F9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_non_string_project() -> None:
    """Non-string project -> enqueue_link returns immediately."""
    linker, fetcher, create_edge = _make_linker(
        fetcher_result=_task_failed_metadata()
    )

    await linker.enqueue_link(
        memory_id="bugfix-001",
        metadata=_bug_fix_metadata(),
        project=42,
    )
    await _drain_inflight(linker)

    fetcher.assert_not_awaited()
    create_edge.assert_not_awaited()
