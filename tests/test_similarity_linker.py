"""Unit tests for :class:`SimilarityLinker` (PLAN-0759 Phase 2 / Sprint 6).

Design
------

These tests are **fully hermetic**. They do not touch the real Pinecone
index, the real Neo4j container, or the ``MemoryService`` orchestration
surface. Every dependency of :class:`SimilarityLinker` is replaced with a
``unittest.mock.MagicMock`` or a tiny hand-rolled fake so that the tests
isolate the linker's own logic: fan-out math, threshold filtering,
self-exclusion, project scoping, exception containment, and the
semaphore-bounded concurrency contract.

Every test asserts either:

1. A contract on the *inputs* the linker hands to its dependencies
   (what it asks Pinecone for, what edges it hands to the edge service), or
2. A contract on the *outputs* of :meth:`enqueue_link` itself (returns
   immediately, never raises, logs the right event on the empty-candidate
   path), or
3. A structural invariant of the class (constant values, no module-level
   state).

No test depends on wall-clock timing beyond a single coarse upper bound
on how long ``enqueue_link`` may take to return on the fast path (< 50ms
for mocks — well above any real CI machine's scheduling jitter but still
orders of magnitude below any real Pinecone / Neo4j round-trip).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any
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
from app.services.associations.similarity_linker import SimilarityLinker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match(mid: str, score: float, metadata: dict | None = None) -> dict:
    """Return a Pinecone-compatible match dict."""
    return {"id": mid, "score": score, "metadata": metadata or {}}


def _embedding(seed: float = 0.1, dim: int = 8) -> list[float]:
    """Return a tiny deterministic embedding. Dimension is irrelevant to
    the linker — we only ever pass it through to Pinecone which is mocked.
    """
    return [seed * (i + 1) for i in range(dim)]


def _make_linker(
    *,
    pinecone_matches: list[dict] | Exception | None = None,
    edge_service_batch: AsyncMock | None = None,
    cross_project_enabled: bool = False,
) -> tuple[SimilarityLinker, MagicMock, AsyncMock]:
    """Construct a linker wired to controllable mocks.

    Returns ``(linker, pinecone_mock, create_edges_batch_mock)``.
    """
    pinecone = MagicMock(name="PineconeClient")
    if isinstance(pinecone_matches, Exception):
        pinecone.query_vector.side_effect = pinecone_matches
    else:
        pinecone.query_vector.return_value = pinecone_matches or []

    edge_service = MagicMock(name="MemoryEdgeService")
    if edge_service_batch is None:
        edge_service_batch = AsyncMock(return_value=0)
    edge_service.create_edges_batch = edge_service_batch

    linker = SimilarityLinker(
        pinecone_client=pinecone,
        edge_service=edge_service,
        cross_project_enabled=cross_project_enabled,
    )
    return linker, pinecone, edge_service_batch


async def _drain_inflight(linker: SimilarityLinker, timeout: float = 2.0) -> None:
    """Wait for every currently in-flight background task to complete."""
    deadline = time.perf_counter() + timeout
    while linker._inflight and time.perf_counter() < deadline:
        tasks = list(linker._inflight)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# 1. Constructor
# ---------------------------------------------------------------------------


def test_constructor_stores_injected_dependencies_without_instantiation() -> None:
    """The constructor is a pure assignment — no side effects on deps."""
    pinecone = MagicMock(name="PineconeClient")
    edge_service = MagicMock(name="MemoryEdgeService")

    linker = SimilarityLinker(
        pinecone_client=pinecone,
        edge_service=edge_service,
        cross_project_enabled=True,
    )

    assert linker._pinecone is pinecone
    assert linker._edge_service is edge_service
    assert linker._cross_project_enabled is True
    # No calls on either dep at construction time.
    pinecone.assert_not_called()
    edge_service.assert_not_called()
    # Semaphore is created fresh, sized correctly.
    assert isinstance(linker._semaphore, asyncio.Semaphore)
    assert linker._inflight == set()


# ---------------------------------------------------------------------------
# 2. Default threshold pinning
# ---------------------------------------------------------------------------


def test_default_threshold_is_exactly_0_82() -> None:
    """Operator decision A — ship conservative, no calibration work.

    This test exists to catch silent drift: any future refactor that tweaks
    the threshold up or down will fail this test loudly, forcing the
    change to be a deliberate operator decision.
    """
    assert SimilarityLinker.SIMILARITY_THRESHOLD == 0.82


# ---------------------------------------------------------------------------
# 3. enqueue_link is non-blocking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_returns_immediately() -> None:
    """enqueue_link schedules work on the loop and returns in < 50 ms."""
    # Give Pinecone a big candidate pool so _link_one does real work —
    # the whole point of this test is that enqueue_link returns before
    # any of that real work starts.
    matches = [_match(f"n-{i}", 0.9 - 0.01 * i) for i in range(20)]
    linker, _, batch_mock = _make_linker(
        pinecone_matches=matches, cross_project_enabled=True
    )

    t0 = time.perf_counter()
    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={},
        project=None,  # cross_project_enabled=True so None is fine
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"enqueue_link blocked for {elapsed_ms:.3f}ms"

    # Drain for cleanliness — but it is NOT required for correctness.
    await _drain_inflight(linker)
    batch_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# 4. Pinecone request shape (top_k, filter, project scoping)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_requests_candidate_pool_plus_one() -> None:
    """top_k = CANDIDATE_POOL + 1 (over-fetch for self-exclusion)."""
    linker, pinecone, _ = _make_linker(
        pinecone_matches=[], cross_project_enabled=True
    )
    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={},
        project="p-A",
    )
    await _drain_inflight(linker)

    pinecone.query_vector.assert_called_once()
    kwargs = pinecone.query_vector.call_args.kwargs
    assert kwargs["top_k"] == SimilarityLinker.CANDIDATE_POOL + 1


@pytest.mark.asyncio
async def test_enqueue_link_scopes_by_project_when_cross_project_disabled() -> None:
    """cross_project_enabled=False ⇒ filter={'project': project}."""
    linker, pinecone, _ = _make_linker(
        pinecone_matches=[], cross_project_enabled=False
    )
    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={},
        project="p-A",
    )
    await _drain_inflight(linker)

    kwargs = pinecone.query_vector.call_args.kwargs
    assert kwargs["filter"] == {"project": "p-A"}


@pytest.mark.asyncio
async def test_enqueue_link_no_filter_when_cross_project_enabled() -> None:
    """cross_project_enabled=True ⇒ filter=None."""
    linker, pinecone, _ = _make_linker(
        pinecone_matches=[], cross_project_enabled=True
    )
    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={},
        project="p-A",
    )
    await _drain_inflight(linker)

    kwargs = pinecone.query_vector.call_args.kwargs
    assert kwargs["filter"] is None


@pytest.mark.asyncio
async def test_enqueue_link_skips_entirely_when_project_none_and_scoped() -> None:
    """project=None + cross_project_enabled=False ⇒ no Pinecone call."""
    linker, pinecone, batch_mock = _make_linker(
        pinecone_matches=[_match("n-1", 0.99)], cross_project_enabled=False
    )
    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={},
        project=None,
    )
    await _drain_inflight(linker)

    pinecone.query_vector.assert_not_called()
    batch_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Threshold filtering + top-K enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_five_candidates_all_above_threshold_produces_five_edges() -> None:
    matches = [_match(f"n-{i}", 0.9 - 0.01 * i) for i in range(5)]
    linker, _, batch_mock = _make_linker(
        pinecone_matches=matches, cross_project_enabled=True
    )

    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={},
        project="p-A",
    )
    await _drain_inflight(linker)

    batch_mock.assert_awaited_once()
    (edges,) = batch_mock.call_args.args
    assert len(edges) == 5
    assert all(e.edge_type == "SIMILAR_TO" for e in edges)
    assert {e.target_id for e in edges} == {f"n-{i}" for i in range(5)}


@pytest.mark.asyncio
async def test_mixed_candidates_only_above_threshold_are_linked() -> None:
    """5 candidates, 3 above 0.82 → 3 edges."""
    matches = [
        _match("n-hi-1", 0.95),
        _match("n-lo-1", 0.80),
        _match("n-hi-2", 0.88),
        _match("n-hi-3", 0.82),  # exactly at threshold ⇒ kept
        _match("n-lo-2", 0.50),
    ]
    linker, _, batch_mock = _make_linker(
        pinecone_matches=matches, cross_project_enabled=True
    )
    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={},
        project="p-A",
    )
    await _drain_inflight(linker)

    batch_mock.assert_awaited_once()
    (edges,) = batch_mock.call_args.args
    assert {e.target_id for e in edges} == {"n-hi-1", "n-hi-2", "n-hi-3"}


@pytest.mark.asyncio
async def test_self_is_excluded_from_pinecone_candidates() -> None:
    """If Pinecone returns the memory's own id, it is filtered out."""
    matches = [
        _match("m-self", 0.99),  # the memory itself
        _match("n-1", 0.95),
        _match("n-2", 0.88),
    ]
    linker, _, batch_mock = _make_linker(
        pinecone_matches=matches, cross_project_enabled=True
    )
    await linker.enqueue_link(
        memory_id="m-self",
        embedding=_embedding(),
        metadata={},
        project="p-A",
    )
    await _drain_inflight(linker)

    batch_mock.assert_awaited_once()
    (edges,) = batch_mock.call_args.args
    target_ids = {e.target_id for e in edges}
    assert "m-self" not in target_ids
    assert target_ids == {"n-1", "n-2"}


@pytest.mark.asyncio
async def test_more_than_max_neighbors_uses_only_top_k() -> None:
    """MAX_NEIGHBORS + 5 candidates above threshold → only top MAX_NEIGHBORS."""
    # 15 candidates, all above 0.82, with strictly descending scores so
    # the expected top-K is deterministic: n-00..n-09 (highest scores).
    matches = [_match(f"n-{i:02d}", 0.99 - 0.005 * i) for i in range(15)]
    linker, _, batch_mock = _make_linker(
        pinecone_matches=matches, cross_project_enabled=True
    )
    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={},
        project="p-A",
    )
    await _drain_inflight(linker)

    batch_mock.assert_awaited_once()
    (edges,) = batch_mock.call_args.args
    assert len(edges) == SimilarityLinker.MAX_NEIGHBORS
    assert {e.target_id for e in edges} == {
        f"n-{i:02d}" for i in range(SimilarityLinker.MAX_NEIGHBORS)
    }


# ---------------------------------------------------------------------------
# 6. Empty / error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_pinecone_candidates_produces_zero_edges_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, _, batch_mock = _make_linker(
        pinecone_matches=[], cross_project_enabled=True
    )
    with caplog.at_level("INFO", logger="app.services.associations.similarity_linker"):
        await linker.enqueue_link(
            memory_id="m-1",
            embedding=_embedding(),
            metadata={},
            project="p-A",
        )
        await _drain_inflight(linker)

    batch_mock.assert_not_awaited()
    assert any("similarity_link.no_candidates" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_pinecone_exception_is_contained_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, _, batch_mock = _make_linker(
        pinecone_matches=RuntimeError("pinecone down"),
        cross_project_enabled=True,
    )
    with caplog.at_level("WARNING", logger="app.services.associations.similarity_linker"):
        # enqueue_link must not raise.
        await linker.enqueue_link(
            memory_id="m-1",
            embedding=_embedding(),
            metadata={},
            project="p-A",
        )
        await _drain_inflight(linker)

    batch_mock.assert_not_awaited()
    assert any("similarity_link.failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_edge_service_batch_exception_is_contained_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failing_batch = AsyncMock(side_effect=RuntimeError("neo4j down"))
    linker, _, _ = _make_linker(
        pinecone_matches=[_match("n-1", 0.95)],
        edge_service_batch=failing_batch,
        cross_project_enabled=True,
    )
    with caplog.at_level("WARNING", logger="app.services.associations.similarity_linker"):
        await linker.enqueue_link(
            memory_id="m-1",
            embedding=_embedding(),
            metadata={},
            project="p-A",
        )
        await _drain_inflight(linker)

    failing_batch.assert_awaited_once()
    assert any("similarity_link.failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 7. Edge property correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edges_carry_correct_attribution_and_metadata() -> None:
    matches = [_match("n-1", 0.95), _match("n-2", 0.87)]
    linker, _, batch_mock = _make_linker(
        pinecone_matches=matches, cross_project_enabled=True
    )
    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={"layer": "episodic"},
        project="p-A",
    )
    await _drain_inflight(linker)

    (edges,) = batch_mock.call_args.args
    assert len(edges) == 2
    # All edges for a single enqueue_link call share the same run_id.
    run_ids = {e.run_id for e in edges}
    assert len(run_ids) == 1
    (run_id,) = run_ids
    assert run_id.startswith("wt-link-")
    assert len(run_id) == len("wt-link-") + 8

    for e in edges:
        assert e.edge_type == "SIMILAR_TO"
        assert "SIMILAR_TO" in VALID_EDGE_TYPES  # sanity
        assert "SIMILAR_TO" in BIDIRECTIONAL_EDGE_TYPES  # canonicalized downstream
        assert e.created_by == "similarity_linker"
        assert e.source_id == "m-1"
        assert e.target_id in {"n-1", "n-2"}
        assert 0.0 <= e.weight <= 1.0
        # metadata is None because Sprint 4's edge_cypher template
        # cannot persist Map-valued properties to Neo4j. Attribution
        # lives on created_by + run_id instead. This pins that decision.
        assert e.metadata is None
        # MemoryEdge dataclass property sanity:
        assert isinstance(e, MemoryEdge)


@pytest.mark.asyncio
async def test_edge_weight_matches_pinecone_score() -> None:
    matches = [_match("n-1", 0.9275), _match("n-2", 0.8423)]
    linker, _, batch_mock = _make_linker(
        pinecone_matches=matches, cross_project_enabled=True
    )
    await linker.enqueue_link(
        memory_id="m-1",
        embedding=_embedding(),
        metadata={},
        project="p-A",
    )
    await _drain_inflight(linker)

    (edges,) = batch_mock.call_args.args
    by_id = {e.target_id: e for e in edges}
    assert by_id["n-1"].weight == pytest.approx(0.9275)
    assert by_id["n-2"].weight == pytest.approx(0.8423)


# ---------------------------------------------------------------------------
# 8. Bounded concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_concurrency_never_exceeds_max_in_flight() -> None:
    """100 concurrent enqueue_link calls — inside the semaphore-held section
    of ``_link_one``, the number of concurrently-executing linker tasks
    must never exceed :data:`BACKGROUND_MAX_IN_FLIGHT`.

    Instrumentation: we count *inside* Pinecone's blocking ``query_vector``
    call, which is invoked via ``asyncio.to_thread`` inside the linker's
    ``async with self._semaphore`` block. The Pinecone side is the
    narrowest window where we can observe "real" concurrency — by the
    time a task has crossed into ``query_vector`` it has already
    acquired the semaphore, and it will not release until the entire
    ``_link_one`` call completes.
    """
    import threading

    live = 0
    max_concurrent = 0
    counter_lock = threading.Lock()

    def blocking_query(*args: Any, **kwargs: Any) -> list[dict]:
        nonlocal live, max_concurrent
        with counter_lock:
            live += 1
            if live > max_concurrent:
                max_concurrent = live
        try:
            # Sleep long enough to force overlap across the entire
            # 100-task burst: with 100 tasks, 32 slots, and 5 ms sleep,
            # every task will share the semaphore-held window with at
            # least some others. Busy-sleep in the thread pool; the
            # asyncio loop stays responsive.
            time.sleep(0.005)
            return []
        finally:
            with counter_lock:
                live -= 1

    pinecone = MagicMock()
    pinecone.query_vector.side_effect = blocking_query

    edge_service = MagicMock()
    edge_service.create_edges_batch = AsyncMock(return_value=0)

    linker = SimilarityLinker(
        pinecone_client=pinecone,
        edge_service=edge_service,
        cross_project_enabled=True,
    )

    # Fire 100 enqueues rapid-fire.
    for i in range(100):
        await linker.enqueue_link(
            memory_id=f"m-{i}",
            embedding=_embedding(seed=float(i) * 0.001),
            metadata={},
            project="p-A",
        )
    await _drain_inflight(linker, timeout=30.0)

    # Every task released its counter slot.
    assert live == 0
    # Semaphore bound holds. We also require *some* concurrency actually
    # happened (otherwise we might be single-threading through and the
    # test would be vacuous).
    assert max_concurrent > 1, (
        f"no concurrency observed (max_concurrent={max_concurrent}); "
        "the test is vacuous"
    )
    assert max_concurrent <= SimilarityLinker.BACKGROUND_MAX_IN_FLIGHT, (
        f"observed {max_concurrent} concurrent linker tasks inside the "
        f"semaphore-held section, limit is "
        f"{SimilarityLinker.BACKGROUND_MAX_IN_FLIGHT}"
    )


# ---------------------------------------------------------------------------
# 9. Secondary structural invariants
# ---------------------------------------------------------------------------


def test_class_constants_are_integers_or_floats_as_documented() -> None:
    """Sanity pin on the class-level configuration."""
    assert isinstance(SimilarityLinker.SIMILARITY_THRESHOLD, float)
    assert isinstance(SimilarityLinker.MAX_NEIGHBORS, int)
    assert isinstance(SimilarityLinker.CANDIDATE_POOL, int)
    assert isinstance(SimilarityLinker.BACKGROUND_TIMEOUT, float)
    assert isinstance(SimilarityLinker.BACKGROUND_MAX_IN_FLIGHT, int)
    assert SimilarityLinker.CANDIDATE_POOL >= SimilarityLinker.MAX_NEIGHBORS
