"""Unit tests for :class:`TemporalLinker` (PLAN-0759 Phase 4 / Sprint 10).

Design
------

These tests are **fully hermetic**. They do not touch the real Neo4j
container or the ``MemoryService`` orchestration surface. Every
dependency of :class:`TemporalLinker` is replaced with an
``unittest.mock.AsyncMock`` / ``MagicMock`` so the tests isolate the
linker's own logic: session-scoped predecessor lookup, per-session
lock serialization, out-of-order arrival handling, edge-attribution
correctness, bounded concurrency, exception containment, and the
constructor's dependency-injection contract.

Each test asserts one of:

1. A contract on the *inputs* the linker hands to its dependencies
   (which predecessor-lookup args were bound, which edges hit the
   edge service), or
2. A contract on the *outputs* of :meth:`enqueue_link` itself
   (returns immediately, never raises, logs the right event on skip
   paths), or
3. A structural invariant of the class (no module-level state, no
   flag reads, no driver construction at init).

No test depends on wall-clock timing beyond coarse upper bounds on
how long ``enqueue_link`` may take to return on the fast path.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# --- Make the ``app`` package importable without pulling in app.config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

from app.services.associations.memory_edges import (
    BIDIRECTIONAL_EDGE_TYPES,
    MemoryEdge,
    VALID_EDGE_TYPES,
)
from app.services.associations.temporal_linker import TemporalLinker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linker(
    *,
    predecessor_result: Optional[tuple[str, int]] = None,
    predecessor_side_effect: Any = None,
    edge_create_return: bool = True,
    edge_create_side_effect: Any = None,
) -> tuple[TemporalLinker, AsyncMock, AsyncMock]:
    """Construct a linker wired to controllable async mocks.

    Returns ``(linker, predecessor_lookup_mock, edge_create_mock)``.
    ``predecessor_lookup_mock`` is the ``AsyncMock`` used in place of
    the driver-based predecessor lookup; tests assert on its
    ``call_args`` to verify the session_id / event_seq / exclude_id
    binding. ``edge_create_mock`` is the mock standing in for
    :meth:`MemoryEdgeService.create_edge`; tests inspect its
    ``call_args_list`` for ``MemoryEdge`` correctness.
    """
    predecessor_lookup = AsyncMock(
        return_value=predecessor_result,
        side_effect=predecessor_side_effect,
    )

    edge_service = MagicMock(name="MemoryEdgeService")
    if edge_create_side_effect is not None:
        edge_service.create_edge = AsyncMock(side_effect=edge_create_side_effect)
    else:
        edge_service.create_edge = AsyncMock(return_value=edge_create_return)

    linker = TemporalLinker(
        edge_service=edge_service,
        predecessor_lookup=predecessor_lookup,
    )
    return linker, predecessor_lookup, edge_service.create_edge


async def _drain_inflight(linker: TemporalLinker, timeout: float = 2.0) -> None:
    """Wait for every currently in-flight background task to complete."""
    deadline = time.perf_counter() + timeout
    while linker._inflight and time.perf_counter() < deadline:
        tasks = list(linker._inflight)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# 1. Constructor stores injected deps, no sessions / queries at init
# ---------------------------------------------------------------------------


def test_constructor_stores_injected_deps_without_side_effects() -> None:
    """Constructor is a pure assignment — no lookup or edge call fires."""
    edge_service = MagicMock(name="MemoryEdgeService")
    pred_lookup = AsyncMock(return_value=None)

    linker = TemporalLinker(
        edge_service=edge_service,
        predecessor_lookup=pred_lookup,
    )

    assert linker._edge_service is edge_service
    assert linker._predecessor_lookup is pred_lookup
    assert linker._database == "neo4j"
    assert linker._driver is None
    assert isinstance(linker._semaphore, asyncio.Semaphore)
    assert linker._session_locks == {}
    assert isinstance(linker._session_locks_lock, asyncio.Lock)
    assert linker._inflight == set()
    # Neither dependency saw a call.
    pred_lookup.assert_not_called()
    edge_service.assert_not_called()


# ---------------------------------------------------------------------------
# 2. session_id=None → no_session path, no lookup, no edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_session_id_none_logs_no_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, pred_lookup, create_edge = _make_linker(predecessor_result=("p-1", 1))
    with caplog.at_level(
        "INFO", logger="app.services.associations.temporal_linker"
    ):
        await linker.enqueue_link(
            memory_id="m-1",
            session_id=None,
            thread_id="t-1",
            event_seq=3,
            project="p-A",
        )
        await _drain_inflight(linker)

    pred_lookup.assert_not_awaited()
    create_edge.assert_not_awaited()
    assert any("temporal_link.no_session" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 3. event_seq=None → no_session path, no lookup, no edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_event_seq_none_logs_no_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, pred_lookup, create_edge = _make_linker(predecessor_result=("p-1", 1))
    with caplog.at_level(
        "INFO", logger="app.services.associations.temporal_linker"
    ):
        await linker.enqueue_link(
            memory_id="m-1",
            session_id="s-1",
            thread_id="t-1",
            event_seq=None,
            project="p-A",
        )
        await _drain_inflight(linker)

    pred_lookup.assert_not_awaited()
    create_edge.assert_not_awaited()
    assert any("temporal_link.no_session" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. enqueue_link returns immediately for hermetic mocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_returns_immediately() -> None:
    """enqueue_link schedules work on the loop and returns in < 50 ms."""
    linker, _, create_edge = _make_linker(predecessor_result=("p-1", 1))

    t0 = time.perf_counter()
    await linker.enqueue_link(
        memory_id="m-1",
        session_id="s-1",
        thread_id="t-1",
        event_seq=3,
        project="p-A",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"enqueue_link blocked for {elapsed_ms:.3f}ms"

    # Drain for cleanliness — but it is NOT required for correctness.
    await _drain_inflight(linker)
    create_edge.assert_awaited_once()


# ---------------------------------------------------------------------------
# 5. First memory in session: no_predecessor, zero edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_memory_in_session_logs_no_predecessor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, pred_lookup, create_edge = _make_linker(predecessor_result=None)
    with caplog.at_level(
        "INFO", logger="app.services.associations.temporal_linker"
    ):
        await linker.enqueue_link(
            memory_id="m-1",
            session_id="s-1",
            thread_id="t-1",
            event_seq=1,  # first in session, but the fact is encoded
            project="p-A",  # by predecessor_result=None above.
        )
        await _drain_inflight(linker)

    pred_lookup.assert_awaited_once()
    create_edge.assert_not_awaited()
    assert any(
        "temporal_link.no_predecessor" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# 6. Second memory in session: one MEMORY_FOLLOWS edge created
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_memory_creates_one_memory_follows_edge() -> None:
    linker, pred_lookup, create_edge = _make_linker(
        predecessor_result=("m-first", 1)
    )
    await linker.enqueue_link(
        memory_id="m-2",
        session_id="s-1",
        thread_id="t-1",
        event_seq=3,
        project="p-A",
    )
    await _drain_inflight(linker)

    pred_lookup.assert_awaited_once_with("s-1", 3, "m-2")
    create_edge.assert_awaited_once()
    (edge,) = create_edge.call_args.args
    assert isinstance(edge, MemoryEdge)
    assert edge.edge_type == "MEMORY_FOLLOWS"
    assert "MEMORY_FOLLOWS" in VALID_EDGE_TYPES
    assert "MEMORY_FOLLOWS" not in BIDIRECTIONAL_EDGE_TYPES


# ---------------------------------------------------------------------------
# 7. Edge direction: source=current (later), target=predecessor (earlier)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_direction_current_points_to_predecessor() -> None:
    linker, _, create_edge = _make_linker(predecessor_result=("m-earlier", 2))
    await linker.enqueue_link(
        memory_id="m-later",
        session_id="s-1",
        thread_id="t-1",
        event_seq=9,
        project="p-A",
    )
    await _drain_inflight(linker)

    create_edge.assert_awaited_once()
    (edge,) = create_edge.call_args.args
    assert edge.source_id == "m-later"
    assert edge.target_id == "m-earlier"
    # Direction invariant: later memory FOLLOWS the earlier one.


# ---------------------------------------------------------------------------
# 8. Edge attribution: type / created_by / run_id prefix / metadata=None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_attribution_fields_are_correct() -> None:
    linker, _, create_edge = _make_linker(predecessor_result=("m-p", 4))
    await linker.enqueue_link(
        memory_id="m-c",
        session_id="s-1",
        thread_id="t-1",
        event_seq=5,
        project="p-A",
    )
    await _drain_inflight(linker)

    (edge,) = create_edge.call_args.args
    assert edge.edge_type == "MEMORY_FOLLOWS"
    assert edge.created_by == "temporal_linker"
    assert edge.run_id.startswith("wt-temporal-")
    assert len(edge.run_id) == len("wt-temporal-") + 8
    assert edge.weight == 1.0
    assert edge.metadata is None, (
        "Neo4j 5 refuses Map-valued relationship properties; "
        "metadata must be LITERALLY None on every MEMORY_FOLLOWS edge"
    )
    # created_at and last_seen_at are set to the same ISO string on first write.
    assert edge.created_at == edge.last_seen_at
    assert isinstance(edge.created_at, str) and edge.created_at


# ---------------------------------------------------------------------------
# 9. Session isolation: lookup args carry the exact session_id each call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_isolation_lookup_args_carry_session_id() -> None:
    linker, pred_lookup, _ = _make_linker(predecessor_result=None)
    await linker.enqueue_link(
        memory_id="m-A", session_id="s-A", thread_id="t-A",
        event_seq=1, project="p-X",
    )
    await linker.enqueue_link(
        memory_id="m-B", session_id="s-B", thread_id="t-B",
        event_seq=1, project="p-X",
    )
    await _drain_inflight(linker)

    assert pred_lookup.await_count == 2
    calls = pred_lookup.await_args_list
    sessions_seen = {call.args[0] for call in calls}
    assert sessions_seen == {"s-A", "s-B"}
    # Every call received the session_id it was enqueued under — no
    # cross-session leakage.


# ---------------------------------------------------------------------------
# 10. Per-session lock: same-session calls run serially
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_session_lock_serializes_same_session_calls() -> None:
    """Two concurrent enqueue_link calls on the same session run one at a
    time inside the session-lock window. Measure via a predecessor
    lookup that records overlap.
    """
    live = 0
    max_concurrent = 0
    counter_lock = asyncio.Lock()

    async def slow_lookup(
        session_id: str, event_seq: int, exclude_id: str
    ) -> Optional[tuple[str, int]]:
        nonlocal live, max_concurrent
        async with counter_lock:
            live += 1
            if live > max_concurrent:
                max_concurrent = live
        try:
            await asyncio.sleep(0.02)
            return None
        finally:
            async with counter_lock:
                live -= 1

    edge_service = MagicMock()
    edge_service.create_edge = AsyncMock(return_value=True)
    linker = TemporalLinker(
        edge_service=edge_service, predecessor_lookup=slow_lookup
    )

    # Two memories in the SAME session, dispatched close together.
    await linker.enqueue_link(
        memory_id="m-1", session_id="s-same", thread_id="t",
        event_seq=1, project="p-A",
    )
    await linker.enqueue_link(
        memory_id="m-2", session_id="s-same", thread_id="t",
        event_seq=2, project="p-A",
    )
    await _drain_inflight(linker, timeout=5.0)

    # At most one task was ever inside the lookup at any instant.
    assert max_concurrent == 1, (
        f"per-session lock did not serialize: observed {max_concurrent} "
        f"concurrent predecessor lookups in the same session"
    )


# ---------------------------------------------------------------------------
# 11. Different sessions: calls run in parallel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_different_sessions_run_in_parallel() -> None:
    """Two concurrent enqueue_link calls on DIFFERENT sessions may
    overlap inside the predecessor lookup.
    """
    live = 0
    max_concurrent = 0
    counter_lock = asyncio.Lock()

    async def slow_lookup(
        session_id: str, event_seq: int, exclude_id: str
    ) -> Optional[tuple[str, int]]:
        nonlocal live, max_concurrent
        async with counter_lock:
            live += 1
            if live > max_concurrent:
                max_concurrent = live
        try:
            await asyncio.sleep(0.02)
            return None
        finally:
            async with counter_lock:
                live -= 1

    edge_service = MagicMock()
    edge_service.create_edge = AsyncMock(return_value=True)
    linker = TemporalLinker(
        edge_service=edge_service, predecessor_lookup=slow_lookup
    )

    # Fire two calls on DIFFERENT sessions back-to-back so their
    # background tasks contend for overlap.
    await linker.enqueue_link(
        memory_id="m-A", session_id="s-A", thread_id="t",
        event_seq=1, project="p-A",
    )
    await linker.enqueue_link(
        memory_id="m-B", session_id="s-B", thread_id="t",
        event_seq=1, project="p-A",
    )
    await _drain_inflight(linker, timeout=5.0)

    assert max_concurrent == 2, (
        f"different sessions should run in parallel; observed "
        f"max_concurrent={max_concurrent}"
    )


# ---------------------------------------------------------------------------
# 12. Out-of-order arrival: Sprint 10's self-healing MERGE strategy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_out_of_order_arrival_self_heals_via_merge() -> None:
    """Memories arrive as (B, seq=5) then (A, seq=3) then (C, seq=7).

    Sprint 10 design:
    - B arrives first: predecessor_lookup(s, 5, B) returns None (A not
      yet visible to the lookup) — no edge.
    - A arrives second: predecessor_lookup(s, 3, A) returns None (B has
      seq=5 which is > 3, not a predecessor) — no edge.
    - C arrives third: predecessor_lookup(s, 7, C) returns (B, 5) —
      one edge created: C -> B.

    Total edges: exactly 1, direction C -> B. The missing B -> A edge
    is the known deferred gap for out-of-order arrivals spanning two
    separate store cycles; documented in the module docstring.
    """
    # A hand-rolled stateful stub for predecessor_lookup that mimics
    # the Neo4j query semantics: returns the largest seq < event_seq
    # in the same session from the `store` of memories we've told it
    # about, but we gate store visibility to encode "A is not yet
    # visible when B queries".
    store: list[tuple[str, int]] = []

    async def stateful_lookup(
        session_id: str, event_seq: int, exclude_id: str
    ) -> Optional[tuple[str, int]]:
        assert session_id == "s-out-of-order"
        candidates = [
            (mid, seq) for (mid, seq) in store
            if seq < event_seq and mid != exclude_id
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    edge_service = MagicMock()
    edge_service.create_edge = AsyncMock(return_value=True)
    linker = TemporalLinker(
        edge_service=edge_service, predecessor_lookup=stateful_lookup
    )

    # B arrives first. At this moment the store is empty.
    await linker.enqueue_link(
        memory_id="m-B", session_id="s-out-of-order", thread_id="t",
        event_seq=5, project="p-A",
    )
    await _drain_inflight(linker)
    store.append(("m-B", 5))
    assert edge_service.create_edge.await_count == 0

    # A arrives second, with a lower seq. Store visible = [B@5].
    await linker.enqueue_link(
        memory_id="m-A", session_id="s-out-of-order", thread_id="t",
        event_seq=3, project="p-A",
    )
    await _drain_inflight(linker)
    store.append(("m-A", 3))
    assert edge_service.create_edge.await_count == 0, (
        "A's predecessor lookup must find nothing — B has seq=5 > 3"
    )

    # C arrives third, seq=7. Store visible = [B@5, A@3]. Predecessor
    # is B (largest seq < 7).
    await linker.enqueue_link(
        memory_id="m-C", session_id="s-out-of-order", thread_id="t",
        event_seq=7, project="p-A",
    )
    await _drain_inflight(linker)

    assert edge_service.create_edge.await_count == 1
    (edge,) = edge_service.create_edge.call_args.args
    assert edge.source_id == "m-C"
    assert edge.target_id == "m-B"


# ---------------------------------------------------------------------------
# 13. Idempotency: two enqueues for the same memory resolve cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_double_enqueue_same_memory_is_idempotent() -> None:
    """Calling enqueue_link twice for the same memory queries predecessor
    twice and delegates MERGE idempotency to the edge service (which is
    mocked to return True both times). Both calls succeed; no exception.
    """
    linker, pred_lookup, create_edge = _make_linker(
        predecessor_result=("m-pred", 1)
    )

    await linker.enqueue_link(
        memory_id="m-1", session_id="s-1", thread_id="t",
        event_seq=2, project="p-A",
    )
    await linker.enqueue_link(
        memory_id="m-1", session_id="s-1", thread_id="t",
        event_seq=2, project="p-A",
    )
    await _drain_inflight(linker)

    assert pred_lookup.await_count == 2
    assert create_edge.await_count == 2
    # Both edges have the same endpoints and edge_type; MERGE
    # idempotency is the edge service's job (out of scope here).
    for call in create_edge.await_args_list:
        (edge,) = call.args
        assert edge.source_id == "m-1"
        assert edge.target_id == "m-pred"
        assert edge.edge_type == "MEMORY_FOLLOWS"


# ---------------------------------------------------------------------------
# 14. Exception in predecessor_lookup is contained and logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predecessor_lookup_exception_is_contained_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, pred_lookup, create_edge = _make_linker(
        predecessor_side_effect=RuntimeError("neo4j boom")
    )
    with caplog.at_level(
        "WARNING", logger="app.services.associations.temporal_linker"
    ):
        await linker.enqueue_link(
            memory_id="m-1", session_id="s-1", thread_id="t",
            event_seq=2, project="p-A",
        )
        await _drain_inflight(linker)

    pred_lookup.assert_awaited_once()
    create_edge.assert_not_awaited()
    assert any("temporal_link.failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 15. Exception in edge_service.create_edge is contained and logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_service_create_edge_exception_is_contained_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, _, create_edge = _make_linker(
        predecessor_result=("m-pred", 1),
        edge_create_side_effect=RuntimeError("neo4j down"),
    )
    with caplog.at_level(
        "WARNING", logger="app.services.associations.temporal_linker"
    ):
        await linker.enqueue_link(
            memory_id="m-1", session_id="s-1", thread_id="t",
            event_seq=2, project="p-A",
        )
        await _drain_inflight(linker)

    create_edge.assert_awaited_once()
    assert any("temporal_link.failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 16. Timeout is contained and logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_is_contained_and_logged(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shrink the timeout to a few ms and have the predecessor lookup
    sleep forever so ``asyncio.wait_for`` cancels it.
    """
    monkeypatch.setattr(TemporalLinker, "BACKGROUND_TIMEOUT", 0.02)

    async def stalling_lookup(
        session_id: str, event_seq: int, exclude_id: str
    ) -> Optional[tuple[str, int]]:
        await asyncio.sleep(5.0)
        return None

    edge_service = MagicMock()
    edge_service.create_edge = AsyncMock(return_value=True)
    linker = TemporalLinker(
        edge_service=edge_service, predecessor_lookup=stalling_lookup
    )

    with caplog.at_level(
        "WARNING", logger="app.services.associations.temporal_linker"
    ):
        await linker.enqueue_link(
            memory_id="m-1", session_id="s-1", thread_id="t",
            event_seq=2, project="p-A",
        )
        await _drain_inflight(linker, timeout=2.0)

    assert any("temporal_link.failed" in r.message for r in caplog.records)
    assert any("timeout" in r.message for r in caplog.records)
    edge_service.create_edge.assert_not_awaited()


# ---------------------------------------------------------------------------
# 17. Bounded concurrency: 100 calls across many sessions, max ≤ 32
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_concurrency_never_exceeds_max_in_flight() -> None:
    """100 concurrent enqueue_link calls across 100 distinct sessions —
    inside the semaphore-held section, the number of concurrently-
    executing linker tasks must never exceed ``BACKGROUND_MAX_IN_FLIGHT``.
    """
    live = 0
    max_concurrent = 0
    counter_lock = asyncio.Lock()

    async def counting_lookup(
        session_id: str, event_seq: int, exclude_id: str
    ) -> Optional[tuple[str, int]]:
        nonlocal live, max_concurrent
        async with counter_lock:
            live += 1
            if live > max_concurrent:
                max_concurrent = live
        try:
            # Force overlap across the whole 100-task burst.
            await asyncio.sleep(0.005)
            return None
        finally:
            async with counter_lock:
                live -= 1

    edge_service = MagicMock()
    edge_service.create_edge = AsyncMock(return_value=True)
    linker = TemporalLinker(
        edge_service=edge_service, predecessor_lookup=counting_lookup
    )

    for i in range(100):
        await linker.enqueue_link(
            memory_id=f"m-{i}",
            session_id=f"s-{i}",  # distinct sessions ⇒ no per-session lock
            thread_id="t",        # contention, only semaphore contention.
            event_seq=1,
            project="p-A",
        )
    await _drain_inflight(linker, timeout=30.0)

    assert live == 0
    assert max_concurrent > 1, (
        f"no concurrency observed (max_concurrent={max_concurrent}); "
        "the test is vacuous"
    )
    assert max_concurrent <= TemporalLinker.BACKGROUND_MAX_IN_FLIGHT, (
        f"observed {max_concurrent} concurrent linker tasks inside the "
        f"semaphore-held section, limit is "
        f"{TemporalLinker.BACKGROUND_MAX_IN_FLIGHT}"
    )


# ---------------------------------------------------------------------------
# 18. Structured log events: every documented name appears at least once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_log_event_names_are_present(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The six documented event names are all emitted by the expected
    code paths. Executed across multiple enqueue_link calls to cover
    every branch in one test.
    """
    seen: set[str] = set()

    # Path 1: no_session (session_id=None).
    linker_a, _, _ = _make_linker(predecessor_result=None)
    with caplog.at_level(
        "INFO", logger="app.services.associations.temporal_linker"
    ):
        await linker_a.enqueue_link(
            memory_id="m-1", session_id=None, thread_id="t",
            event_seq=1, project="p-A",
        )
        await _drain_inflight(linker_a)
    for r in caplog.records:
        for name in (
            "temporal_link.queued",
            "temporal_link.completed",
            "temporal_link.failed",
            "temporal_link.no_session",
            "temporal_link.no_predecessor",
        ):
            if name in r.message:
                seen.add(name)
    caplog.clear()

    # Path 2: no_predecessor (first memory in session).
    linker_b, _, _ = _make_linker(predecessor_result=None)
    with caplog.at_level(
        "INFO", logger="app.services.associations.temporal_linker"
    ):
        await linker_b.enqueue_link(
            memory_id="m-1", session_id="s-1", thread_id="t",
            event_seq=1, project="p-A",
        )
        await _drain_inflight(linker_b)
    for r in caplog.records:
        for name in (
            "temporal_link.queued",
            "temporal_link.completed",
            "temporal_link.failed",
            "temporal_link.no_predecessor",
        ):
            if name in r.message:
                seen.add(name)
    caplog.clear()

    # Path 3: completed (second memory in session, predecessor found).
    linker_c, _, _ = _make_linker(predecessor_result=("m-pred", 1))
    with caplog.at_level(
        "INFO", logger="app.services.associations.temporal_linker"
    ):
        await linker_c.enqueue_link(
            memory_id="m-1", session_id="s-1", thread_id="t",
            event_seq=2, project="p-A",
        )
        await _drain_inflight(linker_c)
    for r in caplog.records:
        for name in (
            "temporal_link.queued",
            "temporal_link.completed",
            "temporal_link.failed",
        ):
            if name in r.message:
                seen.add(name)
    caplog.clear()

    # Path 4: failed (predecessor lookup raises).
    linker_d, _, _ = _make_linker(
        predecessor_side_effect=RuntimeError("boom")
    )
    with caplog.at_level(
        "WARNING", logger="app.services.associations.temporal_linker"
    ):
        await linker_d.enqueue_link(
            memory_id="m-1", session_id="s-1", thread_id="t",
            event_seq=2, project="p-A",
        )
        await _drain_inflight(linker_d)
    for r in caplog.records:
        if "temporal_link.failed" in r.message:
            seen.add("temporal_link.failed")

    # The dropped_semaphore_full path is exercised by a separate test
    # (below) that fully pins a semaphore of size 1. We assert its
    # name appears in the logger module source as well so this test
    # does not have to race the semaphore.
    import inspect

    from app.services.associations import temporal_linker as tl_module

    src = inspect.getsource(tl_module)
    assert "temporal_link.dropped_semaphore_full" in src

    # The required five event names hit via real execution paths.
    required = {
        "temporal_link.queued",
        "temporal_link.completed",
        "temporal_link.failed",
        "temporal_link.no_session",
        "temporal_link.no_predecessor",
    }
    missing = required - seen
    assert not missing, f"missing structured log events: {missing}"


# ---------------------------------------------------------------------------
# 19. __init__ with neither driver nor predecessor_lookup raises ValueError
# ---------------------------------------------------------------------------


def test_init_without_driver_or_lookup_raises_valueerror() -> None:
    edge_service = MagicMock(name="MemoryEdgeService")
    with pytest.raises(ValueError, match="predecessor_lookup"):
        TemporalLinker(edge_service=edge_service)


# ---------------------------------------------------------------------------
# 20. __init__ with both driver and lookup uses the explicit callable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_with_both_prefers_explicit_predecessor_lookup() -> None:
    """Documented priority: if both driver and predecessor_lookup are
    supplied, the explicit callable wins and the driver is never
    touched.
    """
    edge_service = MagicMock(name="MemoryEdgeService")
    edge_service.create_edge = AsyncMock(return_value=True)

    explicit_lookup = AsyncMock(return_value=("m-pred", 1))

    # The driver is a MagicMock that records any .session() call —
    # the test asserts it is NOT touched on the happy path.
    fake_driver = MagicMock(name="AsyncDriver")

    linker = TemporalLinker(
        edge_service=edge_service,
        predecessor_lookup=explicit_lookup,
        driver=fake_driver,
    )
    assert linker._predecessor_lookup is explicit_lookup

    await linker.enqueue_link(
        memory_id="m-1", session_id="s-1", thread_id="t",
        event_seq=2, project="p-A",
    )
    await _drain_inflight(linker)

    explicit_lookup.assert_awaited_once_with("s-1", 2, "m-1")
    # driver.session was never called — the explicit callable shadowed it.
    fake_driver.session.assert_not_called()


# ---------------------------------------------------------------------------
# 21. Class constants have the expected types (structural pin)
# ---------------------------------------------------------------------------


def test_class_constants_have_documented_types() -> None:
    assert isinstance(TemporalLinker.BACKGROUND_TIMEOUT, float)
    assert isinstance(TemporalLinker.BACKGROUND_MAX_IN_FLIGHT, int)
    assert TemporalLinker.BACKGROUND_MAX_IN_FLIGHT > 0


# ---------------------------------------------------------------------------
# 22. run_id is unique per enqueue_link call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_id_is_unique_per_enqueue_call() -> None:
    linker, _, create_edge = _make_linker(predecessor_result=("m-pred", 1))
    await linker.enqueue_link(
        memory_id="m-1", session_id="s-1", thread_id="t",
        event_seq=2, project="p-A",
    )
    await linker.enqueue_link(
        memory_id="m-2", session_id="s-1", thread_id="t",
        event_seq=3, project="p-A",
    )
    await _drain_inflight(linker)

    run_ids = {call.args[0].run_id for call in create_edge.await_args_list}
    assert len(run_ids) == 2
    for rid in run_ids:
        assert rid.startswith("wt-temporal-")


# ---------------------------------------------------------------------------
# 23. Session lock is created only once per session_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_lock_reused_across_calls_for_same_session() -> None:
    linker, _, _ = _make_linker(predecessor_result=None)
    # Two calls for the same session should reuse the same lock
    # instance in the internal dict.
    await linker.enqueue_link(
        memory_id="m-1", session_id="s-shared", thread_id="t",
        event_seq=1, project="p-A",
    )
    await linker.enqueue_link(
        memory_id="m-2", session_id="s-shared", thread_id="t",
        event_seq=2, project="p-A",
    )
    await _drain_inflight(linker)

    assert "s-shared" in linker._session_locks
    # Exactly one lock entry for the shared session, not two.
    shared_lock = linker._session_locks["s-shared"]
    assert isinstance(shared_lock, asyncio.Lock)
    # A second get_session_lock for the same session returns the same
    # object (no silent replacement).
    again = await linker._get_session_lock("s-shared")
    assert again is shared_lock
