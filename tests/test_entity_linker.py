"""Unit tests for :class:`EntityLinker` (PLAN-0759 Phase 3 / Sprint 9).

Design
------

These tests are **fully hermetic**. They do not touch the real Neo4j
container or the ``MemoryService`` orchestration surface. Every
dependency of :class:`EntityLinker` is replaced with an
``unittest.mock.MagicMock`` / ``AsyncMock`` so the tests isolate the
linker's own logic: tier-A-vs-tier-B entity resolution, canonicalization,
deduplication, edge-property correctness, exception containment, bounded
concurrency, and the read-only lookup helpers.

Each test asserts one of:

1. A contract on the *inputs* the linker hands to the Neo4j driver
   (which Cypher params were bound, how many statements ran), or
2. A contract on the *outputs* of :meth:`enqueue_link` itself (returns
   immediately, never raises, logs the right event on skip paths), or
3. A structural invariant of the class (no module-level state, no
   flag reads, no driver construction at init).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# --- Make the ``app`` package importable ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

from app.services.associations.entity_linker import EntityLinker
from app.services.associations.memory_edges import EDGE_VERSION, VALID_EDGE_TYPES


# ---------------------------------------------------------------------------
# Fake async driver / session / result
# ---------------------------------------------------------------------------


class _FakeAsyncResult:
    """Fake ``neo4j.AsyncResult``. Iterates over a pre-supplied list."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)

    def __aiter__(self) -> "_FakeAsyncResult":
        self._iter = iter(self._rows)
        return self

    async def __anext__(self) -> dict:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def consume(self) -> None:
        return None

    async def single(self) -> dict | None:
        if self._rows:
            return self._rows[0]
        return None


class _FakeAsyncSession:
    """Fake ``neo4j.AsyncSession``. Records every run call."""

    def __init__(
        self,
        run_log: list[tuple[str, dict]],
        default_rows: list[dict] | None = None,
        run_side_effect: Exception | None = None,
    ) -> None:
        self._run_log = run_log
        self._default_rows = default_rows or [{"name": "stub"}]
        self._run_side_effect = run_side_effect

    async def __aenter__(self) -> "_FakeAsyncSession":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def run(self, query: str, params: dict | None = None) -> _FakeAsyncResult:
        if self._run_side_effect is not None:
            raise self._run_side_effect
        self._run_log.append((query, dict(params or {})))
        return _FakeAsyncResult(list(self._default_rows))


class _FakeAsyncDriver:
    """Fake ``neo4j.AsyncDriver``. Hands out a fresh session per call."""

    def __init__(
        self,
        run_log: list[tuple[str, dict]],
        default_rows: list[dict] | None = None,
        run_side_effect: Exception | None = None,
        on_session_open: callable | None = None,
    ) -> None:
        self._run_log = run_log
        self._default_rows = default_rows
        self._run_side_effect = run_side_effect
        self._on_session_open = on_session_open
        self.session_calls: list[dict] = []

    def session(self, database: str | None = None) -> _FakeAsyncSession:
        self.session_calls.append({"database": database})
        if self._on_session_open is not None:
            self._on_session_open()
        return _FakeAsyncSession(
            run_log=self._run_log,
            default_rows=self._default_rows,
            run_side_effect=self._run_side_effect,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linker(
    *,
    default_rows: list[dict] | None = None,
    run_side_effect: Exception | None = None,
    on_session_open: callable | None = None,
) -> tuple[EntityLinker, _FakeAsyncDriver, list[tuple[str, dict]]]:
    """Construct an :class:`EntityLinker` wired to a fake async driver."""
    run_log: list[tuple[str, dict]] = []
    driver = _FakeAsyncDriver(
        run_log=run_log,
        default_rows=default_rows,
        run_side_effect=run_side_effect,
        on_session_open=on_session_open,
    )
    linker = EntityLinker(driver=driver)
    return linker, driver, run_log


async def _drain_inflight(linker: EntityLinker, timeout: float = 2.0) -> None:
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


def test_constructor_stores_injected_driver_without_opening_sessions() -> None:
    """Constructor is a pure assignment — no driver.session() called."""
    driver = MagicMock(name="AsyncDriver")
    linker = EntityLinker(driver=driver)
    assert linker._driver is driver
    assert linker._database == "neo4j"
    assert isinstance(linker._semaphore, asyncio.Semaphore)
    assert linker._inflight == set()
    driver.session.assert_not_called()


def test_constructor_respects_database_override() -> None:
    driver = MagicMock(name="AsyncDriver")
    linker = EntityLinker(driver=driver, database="custom_db")
    assert linker._database == "custom_db"


# ---------------------------------------------------------------------------
# 2. enqueue_link returns immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_returns_immediately_for_mock_driver() -> None:
    linker, _, _ = _make_linker()
    t0 = time.perf_counter()
    await linker.enqueue_link(
        memory_id="m-1",
        content="see agents/memory_router.py and MemoryEdge class",
        metadata={},
        project="p-A",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"enqueue_link blocked for {elapsed_ms:.3f}ms"
    await _drain_inflight(linker)


# ---------------------------------------------------------------------------
# 3. Tier B heuristic extraction (no caller-provided entities)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_runs_extractor_when_entities_missing() -> None:
    linker, driver, run_log = _make_linker()
    await linker.enqueue_link(
        memory_id="m-1",
        content="see agents/memory_router.py using MemoryEdge dataclass",
        metadata={},
        project="p-A",
    )
    await _drain_inflight(linker)

    # At least one session opened and at least one Cypher run invoked.
    assert driver.session_calls, "expected at least one session to open"
    # Two distinct canonicals should have been upserted.
    canon_names = {params["canon_name"] for _, params in run_log}
    assert "agents/memory_router.py" in canon_names
    assert "memoryedge" in canon_names


# ---------------------------------------------------------------------------
# 4. Tier A caller-provided bypasses extractor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_uses_caller_provided_entities_verbatim() -> None:
    linker, _, run_log = _make_linker()
    await linker.enqueue_link(
        memory_id="m-1",
        # Content that would trigger the extractor if Tier B ran.
        content="see agents/memory_router.py and MemoryEdge class",
        metadata={"entities": ["Neo4j", "Pinecone", "FusionMemory"]},
        project="p-A",
    )
    await _drain_inflight(linker)

    canon_names = {params["canon_name"] for _, params in run_log}
    assert canon_names == {"neo4j", "pinecone", "fusionmemory"}
    # The Tier B hits (``agents/memory_router.py``, ``memoryedge``) must
    # NOT appear — Tier A is exclusive when non-empty.
    assert "agents/memory_router.py" not in canon_names
    assert "memoryedge" not in canon_names


# ---------------------------------------------------------------------------
# 5. project=None is a no-op + log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_project_none_does_not_open_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, driver, run_log = _make_linker()
    with caplog.at_level("WARNING", logger="app.services.associations.entity_linker"):
        await linker.enqueue_link(
            memory_id="m-1",
            content="stuff",
            metadata={"entities": ["neo4j"]},
            project=None,
        )
    await _drain_inflight(linker)
    assert driver.session_calls == []
    assert run_log == []
    assert any("entity_link.no_project" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 6. Empty entities is a no-op + log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_no_entities_does_not_open_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, driver, run_log = _make_linker()
    with caplog.at_level("INFO", logger="app.services.associations.entity_linker"):
        await linker.enqueue_link(
            memory_id="m-1",
            content="",  # empty content, empty Tier A
            metadata={},
            project="p-A",
        )
    await _drain_inflight(linker)
    assert driver.session_calls == []
    assert run_log == []
    assert any("entity_link.no_entities" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 7. Canonicalization dedups case/whitespace variants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_canonicalization_dedups_case_variants() -> None:
    linker, _, run_log = _make_linker()
    await linker.enqueue_link(
        memory_id="m-1",
        content="",
        metadata={"entities": ["Neo4j", "neo4j", "NEO4J", " Neo4j "]},
        project="p-A",
    )
    await _drain_inflight(linker)
    canon_names = [params["canon_name"] for _, params in run_log]
    assert canon_names.count("neo4j") == 1
    assert len(run_log) == 1


# ---------------------------------------------------------------------------
# 8. Ranking / truncation applies when > 20 entities supplied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_link_truncates_over_max_entities() -> None:
    linker, _, run_log = _make_linker()
    raw_entities = [f"Entity{i:02d}" for i in range(25)]
    await linker.enqueue_link(
        memory_id="m-1",
        content="",
        metadata={"entities": raw_entities},
        project="p-A",
    )
    await _drain_inflight(linker)
    # Tier A truncates to MAX_ENTITIES_PER_MEMORY (20) in _resolve_entities,
    # then canonicalization preserves 20 distinct canonicals.
    assert len(run_log) == EntityLinker.MAX_ENTITIES_PER_MEMORY


# ---------------------------------------------------------------------------
# 9. Three canonical entities → three MERGE calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_entities_produce_three_cypher_runs() -> None:
    linker, _, run_log = _make_linker()
    await linker.enqueue_link(
        memory_id="m-1",
        content="",
        metadata={"entities": ["alpha", "beta", "gamma"]},
        project="p-A",
    )
    await _drain_inflight(linker)
    assert len(run_log) == 3
    canons = {params["canon_name"] for _, params in run_log}
    assert canons == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# 10. Edge attribution fields on every MERGE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_merge_has_correct_attribution() -> None:
    linker, _, run_log = _make_linker()
    await linker.enqueue_link(
        memory_id="m-1",
        content="",
        metadata={"entities": ["alpha", "beta"]},
        project="p-A",
    )
    await _drain_inflight(linker)
    run_ids = {params["run_id"] for _, params in run_log}
    assert len(run_ids) == 1  # single enqueue_link → single run_id
    (run_id,) = run_ids
    assert run_id.startswith("wt-entity-")
    assert len(run_id) == len("wt-entity-") + 8

    for _, params in run_log:
        assert params["created_by"] == "entity_linker"
        assert params["edge_version"] == EDGE_VERSION
        assert params["weight"] == 1.0
        assert params["project"] == "p-A"
        assert params["memory_id"] == "m-1"
        assert "now" in params and params["now"] is not None


# ---------------------------------------------------------------------------
# 11. metadata=None on every MENTIONS edge (Neo4j 5 rejects map props)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_mentions_edge_has_metadata_none() -> None:
    linker, _, run_log = _make_linker()
    await linker.enqueue_link(
        memory_id="m-1",
        content="",
        metadata={"entities": ["alpha", "beta"], "other": {"a": 1}},
        project="p-A",
    )
    await _drain_inflight(linker)
    assert run_log
    for _, params in run_log:
        assert params["metadata"] is None, (
            "Neo4j 5 refuses Map-valued relationship properties; "
            "metadata must be LITERALLY None on every MENTIONS edge"
        )


# ---------------------------------------------------------------------------
# 12. Neo4j session.run exception is contained and logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_neo4j_session_exception_is_contained_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    linker, _, _ = _make_linker(run_side_effect=RuntimeError("neo4j boom"))
    with caplog.at_level("WARNING", logger="app.services.associations.entity_linker"):
        await linker.enqueue_link(
            memory_id="m-1",
            content="",
            metadata={"entities": ["alpha"]},
            project="p-A",
        )
        await _drain_inflight(linker)
    assert any("entity_link.failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 13. Timeout is contained and logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_is_contained_and_logged(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Shrink timeout to a few ms so the test doesn't hang for 30 s.
    monkeypatch.setattr(EntityLinker, "BACKGROUND_TIMEOUT", 0.02)

    # Build a linker whose session.run sleeps forever — wait_for should
    # cancel it after ~20 ms.
    class _StallingSession:
        async def __aenter__(self) -> "_StallingSession":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def run(self, query: str, params: dict | None = None) -> Any:
            await asyncio.sleep(5.0)
            return _FakeAsyncResult([{"name": "never"}])

    class _StallingDriver:
        def session(self, database: str | None = None) -> _StallingSession:
            return _StallingSession()

    linker = EntityLinker(driver=_StallingDriver())
    with caplog.at_level("WARNING", logger="app.services.associations.entity_linker"):
        await linker.enqueue_link(
            memory_id="m-1",
            content="",
            metadata={"entities": ["alpha"]},
            project="p-A",
        )
        await _drain_inflight(linker, timeout=2.0)
    assert any("entity_link.failed" in r.message for r in caplog.records)
    assert any("timeout" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 14. Bounded concurrency never exceeds BACKGROUND_MAX_IN_FLIGHT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_concurrency_never_exceeds_max_in_flight() -> None:
    """100 concurrent enqueue_link → max observed concurrent sessions
    ≤ BACKGROUND_MAX_IN_FLIGHT (32)."""
    live = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    # Every session-open increments; every session-close decrements.
    class _CountingSession:
        async def __aenter__(self) -> "_CountingSession":
            nonlocal live, max_concurrent
            async with lock:
                live += 1
                if live > max_concurrent:
                    max_concurrent = live
            # Yield long enough that many tasks overlap.
            await asyncio.sleep(0.005)
            return self

        async def __aexit__(self, *exc: Any) -> None:
            nonlocal live
            async with lock:
                live -= 1
            return None

        async def run(self, query: str, params: dict | None = None) -> _FakeAsyncResult:
            return _FakeAsyncResult([{"name": params.get("canon_name", "x")}])

    class _CountingDriver:
        def session(self, database: str | None = None) -> _CountingSession:
            return _CountingSession()

    linker = EntityLinker(driver=_CountingDriver())

    for i in range(100):
        await linker.enqueue_link(
            memory_id=f"m-{i}",
            content="",
            metadata={"entities": ["alpha"]},
            project="p-A",
        )
    await _drain_inflight(linker, timeout=15.0)

    assert live == 0
    assert max_concurrent > 1, f"no concurrency observed ({max_concurrent})"
    assert max_concurrent <= EntityLinker.BACKGROUND_MAX_IN_FLIGHT, (
        f"observed {max_concurrent} concurrent entity-linker sessions, "
        f"limit is {EntityLinker.BACKGROUND_MAX_IN_FLIGHT}"
    )


# ---------------------------------------------------------------------------
# 15. get_memories_for_entity is read-only and returns shaped rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_memories_for_entity_read_only_shape() -> None:
    stub_rows = [
        {"memory_id": "m-a", "created_at": "2026-04-13T10:00:00+00:00",
         "last_seen_at": "2026-04-13T11:00:00+00:00"},
        {"memory_id": "m-b", "created_at": "2026-04-13T09:00:00+00:00",
         "last_seen_at": "2026-04-13T10:30:00+00:00"},
    ]
    linker, _, run_log = _make_linker(default_rows=stub_rows)
    rows = await linker.get_memories_for_entity(
        project="p-A", entity_name="Neo4j", limit=5
    )
    assert isinstance(rows, list)
    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == {"memory_id", "created_at", "last_seen_at"}

    # The lookup canonicalized the entity name.
    assert len(run_log) == 1
    _, params = run_log[0]
    assert params["canon_name"] == "neo4j"
    assert params["project"] == "p-A"
    assert params["limit"] == 5

    # The query is read-only: no MERGE / CREATE / SET / DELETE keywords.
    query, _ = run_log[0]
    upper = query.upper()
    assert "MERGE" not in upper
    assert "CREATE " not in upper
    assert " SET " not in upper
    assert "DELETE" not in upper


# ---------------------------------------------------------------------------
# 16. get_entities_for_memory derives mention_count via COUNT / degree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_entities_for_memory_derives_mention_count_via_degree() -> None:
    stub_rows = [
        {"project": "p-A", "name": "neo4j", "mention_count": 3},
        {"project": "p-A", "name": "pinecone", "mention_count": 1},
    ]
    linker, _, run_log = _make_linker(default_rows=stub_rows)
    rows = await linker.get_entities_for_memory("m-1")

    assert len(rows) == 2
    assert rows[0]["mention_count"] == 3
    assert rows[1]["mention_count"] == 1

    # The query MUST derive mention_count from the graph, not read a
    # stored property. We verify structurally: the query contains an
    # aggregation pattern (``size([...])``) and does NOT read
    # ``e.mention_count``.
    assert len(run_log) == 1
    query, _ = run_log[0]
    assert "size(" in query, (
        "mention_count must be derived via a size()/COUNT subquery, "
        "not from a stored e.mention_count property"
    )
    assert "e.mention_count" not in query, (
        "there is no stored e.mention_count property; mention_count must "
        "be computed from :MENTIONS degree"
    )


# ---------------------------------------------------------------------------
# 17. Validation on lookup helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_memories_for_entity_validates_inputs() -> None:
    linker, _, _ = _make_linker()
    with pytest.raises(ValueError):
        await linker.get_memories_for_entity(project="", entity_name="x", limit=5)
    with pytest.raises(ValueError):
        await linker.get_memories_for_entity(project="p", entity_name="", limit=5)
    with pytest.raises(ValueError):
        await linker.get_memories_for_entity(project="p", entity_name="x", limit=-1)


@pytest.mark.asyncio
async def test_get_entities_for_memory_validates_memory_id() -> None:
    linker, _, _ = _make_linker()
    with pytest.raises(ValueError):
        await linker.get_entities_for_memory("")


# ---------------------------------------------------------------------------
# 18. Structural: MENTIONS is in VALID_EDGE_TYPES
# ---------------------------------------------------------------------------


def test_mentions_edge_type_is_in_whitelist() -> None:
    assert "MENTIONS" in VALID_EDGE_TYPES


# ---------------------------------------------------------------------------
# 19. Class-level constants sanity
# ---------------------------------------------------------------------------


def test_class_constants_sanity() -> None:
    assert isinstance(EntityLinker.BACKGROUND_TIMEOUT, float)
    assert isinstance(EntityLinker.BACKGROUND_MAX_IN_FLIGHT, int)
    assert isinstance(EntityLinker.MAX_ENTITIES_PER_MEMORY, int)
    assert EntityLinker.MAX_ENTITIES_PER_MEMORY == 20


# ---------------------------------------------------------------------------
# 20. Every structured log event name is reachable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_log_event_names_all_emitted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sanity: exercise each skip-path / success-path and check the
    expected event-name substring appears at least once across the
    whole run."""
    caplog.set_level("INFO", logger="app.services.associations.entity_linker")

    # no_project
    linker_np, _, _ = _make_linker()
    await linker_np.enqueue_link(
        memory_id="m-a", content="", metadata={"entities": ["x"]}, project=None,
    )

    # no_entities
    linker_ne, _, _ = _make_linker()
    await linker_ne.enqueue_link(
        memory_id="m-b", content="", metadata={}, project="p-A",
    )

    # queued + completed
    linker_ok, _, _ = _make_linker()
    await linker_ok.enqueue_link(
        memory_id="m-c", content="", metadata={"entities": ["alpha"]}, project="p-A",
    )
    await _drain_inflight(linker_ok)

    # failed
    linker_fail, _, _ = _make_linker(run_side_effect=RuntimeError("neo boom"))
    await linker_fail.enqueue_link(
        memory_id="m-d", content="", metadata={"entities": ["alpha"]}, project="p-A",
    )
    await _drain_inflight(linker_fail)

    messages = " ".join(r.message for r in caplog.records)
    assert "entity_link.no_project" in messages
    assert "entity_link.no_entities" in messages
    assert "entity_link.queued" in messages
    assert "entity_link.completed" in messages
    assert "entity_link.failed" in messages
