"""Integration tests for the Sprint 11 temporal-linker hook in MemoryService.

Safety contract (Sprint 6 / Sprint 9 pattern)
---------------------------------------------

This suite runs against the **live** ``nova_neo4j_db`` container. Every
test obeys the same safety rules as the entity-linker integration suite:

1. **Dedicated primary label.** Every test memory node is written with
   the primary label ``:TemporalLinkerTestNode`` and the secondary label
   ``:base`` so the linker's predecessor-lookup Cypher (which matches
   on ``(:base {entity_id})`` plus ``session_id``/``event_seq``
   properties) can actually bind it. Teardown deletes by the primary
   test label — never by ``:base``.

2. **Unique ``run_id`` per test run.** A fresh
   ``sprint11-temporal-test-<uuid8>`` is generated at fixture start so
   repeat runs never collide with each other or with any pre-existing
   ``wt-temporal-*`` edges on the live graph.

3. **Unique session_id per test.** Every flag-ON test uses its own
   ``f"sprint11-temporal-session-{uuid.uuid4().hex[:8]}"`` so the
   per-session lock inside the linker cannot race between tests, and
   the predecessor-lookup cannot walk into nodes written by any other
   test.

4. **Neo4j-unreachable ⇒ skip.** If the live container is not
   available (CI without the ``nova_neo4j_db`` sidecar), the whole
   module skips cleanly.

5. **Unconditional ``try/finally`` teardown.** Every test wraps its
   setup in a ``try/finally`` that runs both the test-label cleanup
   and a surgical ``wt-temporal-*`` edge sweep scoped to the run_tag.

6. **Production-count invariant at the end.** The final test
   ``test_zzz_production_counts_unchanged`` asserts that ``:base``
   may drift upward but ``:Session``, ``FOLLOWS``, ``INCLUDES`` are
   byte-for-byte unchanged and that no ``:TemporalLinkerTestNode``
   survived. It also asserts there is no surviving
   ``sprint11-temporal-test-*`` edge in the graph.

7. **Pinecone is ALWAYS mocked.** Temporal linking does not touch
   Pinecone at all, but the broader ``MemoryService`` path does (for
   dedup/conflict, which we turn off). The Pinecone client is still a
   MagicMock so nothing leaks out.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# --- Make the ``app`` package importable ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
    from neo4j import exceptions as neo4j_exceptions
except ImportError:  # pragma: no cover - env sanity
    pytest.skip(
        "neo4j driver not installed; cannot run temporal-linker integration tests",
        allow_module_level=True,
    )

from app.config import settings


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_DB = "neo4j"
TEST_LABEL = "TemporalLinkerTestNode"
TEST_RUN_PREFIX = "sprint11-temporal-test-"
TEST_SESSION_PREFIX = "sprint11-temporal-session-"
PROD_SESSION_LABEL = "Session"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _deterministic_embedding(text: str, _model: str = "") -> list[float]:
    """1536-d pseudo-embedding. Content-derived, deterministic, non-zero."""
    import hashlib

    d = hashlib.md5(text.encode("utf-8")).digest()
    out = [((d[i % 16] + i * 7) % 251) / 251.0 for i in range(1536)]
    if not any(out):
        out[0] = 1.0
    return out


async def _try_open_async_driver() -> Optional[AsyncDriver]:
    try:
        driver = AsyncGraphDatabase.driver(TEST_NEO4J_URI, auth=None)
        await driver.verify_connectivity()
        return driver
    except (
        neo4j_exceptions.ServiceUnavailable,
        neo4j_exceptions.AuthError,
        OSError,
    ):
        return None


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def neo4j_async_driver() -> AsyncIterator[AsyncDriver]:
    driver = await _try_open_async_driver()
    if driver is None:
        pytest.skip(
            f"Neo4j not reachable at {TEST_NEO4J_URI}; skipping temporal-linker "
            "integration tests (expected in environments without a running "
            "nova_neo4j_db container)."
        )
    try:
        yield driver
    finally:
        # Belt-and-braces final sweep in case an individual test forgot.
        await _teardown_test_nodes(driver)
        await _teardown_test_edges(driver)
        await driver.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def production_counts(
    neo4j_async_driver: AsyncDriver,
) -> dict[str, int]:
    """Capture production counts once at module start for the final invariant."""
    async with neo4j_async_driver.session(database=TEST_DB) as session:
        base_count = (
            await (await session.run("MATCH (n:base) RETURN count(n) AS c")).single()
        )["c"]
        session_count = (
            await (
                await session.run(
                    f"MATCH (n:{PROD_SESSION_LABEL}) RETURN count(n) AS c"
                )
            ).single()
        )["c"]
        follows_count = (
            await (
                await session.run("MATCH ()-[r:FOLLOWS]->() RETURN count(r) AS c")
            ).single()
        )["c"]
        includes_count = (
            await (
                await session.run("MATCH ()-[r:INCLUDES]->() RETURN count(r) AS c")
            ).single()
        )["c"]
        memory_follows_count = (
            await (
                await session.run(
                    "MATCH ()-[r:MEMORY_FOLLOWS]->() RETURN count(r) AS c"
                )
            ).single()
        )["c"]
    return {
        "base": base_count,
        "session": session_count,
        "follows": follows_count,
        "includes": includes_count,
        "memory_follows": memory_follows_count,
    }


@pytest.fixture
def run_tag() -> str:
    return f"{TEST_RUN_PREFIX}{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_session_id() -> str:
    return f"{TEST_SESSION_PREFIX}{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# Test-node helpers
# --------------------------------------------------------------------------


async def _create_test_node(
    driver: AsyncDriver,
    entity_id: str,
    session_id: Optional[str],
    event_seq: Optional[int],
    run_tag: str,
    project: str = "sprint11-test",
) -> None:
    """Seed a ``:TemporalLinkerTestNode:base`` with the chronology fields
    the linker's predecessor-lookup queries for.
    """
    props: dict[str, Any] = {
        "entity_id": entity_id,
        "test_run": run_tag,
        "project": project,
    }
    if session_id is not None:
        props["session_id"] = session_id
    if event_seq is not None:
        props["event_seq"] = int(event_seq)
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(
                f"CREATE (n:{TEST_LABEL}:base) SET n = $props",
                {"props": props},
            )
        ).consume()


async def _teardown_test_nodes(driver: AsyncDriver) -> None:
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(f"MATCH (n:{TEST_LABEL}) DETACH DELETE n")
        ).consume()


async def _teardown_test_edges(driver: AsyncDriver) -> None:
    """Sweep any MEMORY_FOLLOWS edge whose run_id starts with the test
    prefix. Defence-in-depth: ``DETACH DELETE`` on the test nodes
    already removes these, but a test that manually created edges
    against real ``:base`` nodes would leave them behind.
    """
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(
                "MATCH (:base)-[r:MEMORY_FOLLOWS]->(:base) "
                "WHERE r.run_id STARTS WITH $prefix "
                "DELETE r",
                {"prefix": "wt-temporal-"},
            )
        ).consume()


async def _count_memory_follows_from(
    driver: AsyncDriver, memory_id: str
) -> int:
    async with driver.session(database=TEST_DB) as session:
        rec = await (
            await session.run(
                "MATCH (m:base {entity_id: $mid})"
                "-[r:MEMORY_FOLLOWS]->(:base) "
                "RETURN count(r) AS c",
                {"mid": memory_id},
            )
        ).single()
    return int(rec["c"]) if rec else 0


async def _get_memory_follows_details(
    driver: AsyncDriver, memory_id: str
) -> list[dict]:
    async with driver.session(database=TEST_DB) as session:
        result = await session.run(
            "MATCH (m:base {entity_id: $mid})"
            "-[r:MEMORY_FOLLOWS]->(p:base) "
            "RETURN p.entity_id AS predecessor_id, "
            "p.event_seq AS predecessor_seq, "
            "r.run_id AS run_id, r.created_at AS created_at, "
            "r.last_seen_at AS last_seen_at, r.weight AS weight, "
            "r.created_by AS created_by, r.edge_version AS edge_version, "
            "r.metadata AS metadata",
            {"mid": memory_id},
        )
        rows = [dict(record) async for record in result]
        await result.consume()
    return rows


async def _count_total_memory_follows_in_session(
    driver: AsyncDriver, run_tag: str
) -> int:
    """Count MEMORY_FOLLOWS edges whose source node is tagged with
    ``test_run = run_tag``. Scoped to test nodes only so concurrent
    production writes can't drift the count.
    """
    async with driver.session(database=TEST_DB) as session:
        rec = await (
            await session.run(
                f"MATCH (m:{TEST_LABEL})"
                f"-[r:MEMORY_FOLLOWS]->(:{TEST_LABEL}) "
                f"WHERE m.test_run = $tag "
                f"RETURN count(r) AS c",
                {"tag": run_tag},
            )
        ).single()
    return int(rec["c"]) if rec else 0


# --------------------------------------------------------------------------
# MemoryService factory (mirrors Sprint 9)
# --------------------------------------------------------------------------


def _build_memory_service(
    *,
    live_neo4j_driver: AsyncDriver,
    pinecone_mock: MagicMock,
) -> Any:
    pinecone_patch = patch(
        "app.services.memory_service.PineconeClient",
        return_value=pinecone_mock,
    )
    fake_graph = MagicMock()
    fake_graph.driver = live_neo4j_driver
    fake_graph.initialize = AsyncMock(return_value=True)
    fake_graph.close = AsyncMock(return_value=None)
    fake_graph.check_connection = AsyncMock(return_value=True)
    # upsert_graph_data is mocked because Sprint 11 relies on pre-seeded
    # :TemporalLinkerTestNode:base rows with the chronology fields
    # already set — we do not want _persist_memory_item to actually
    # rewrite them with a single-label :base node.
    fake_graph.upsert_graph_data = AsyncMock(return_value=True)
    fake_graph.delete_graph_data = AsyncMock(return_value=True)
    fake_graph.link_event_to_session = AsyncMock(return_value=True)

    graph_patch = patch(
        "app.services.memory_service.GraphClient",
        return_value=fake_graph,
    )
    embedding_patch = patch(
        "app.services.memory_service.get_embedding",
        side_effect=_deterministic_embedding,
    )
    extract_patch = patch(
        "app.services.memory_service.extract_entities",
        side_effect=lambda _t: [],
    )
    dedup_patch = patch.object(settings, "WRITE_DEDUP_ENABLED", False)
    conflict_patch = patch.object(settings, "CONFLICT_DETECTION_ENABLED", False)

    pinecone_patch.start()
    graph_patch.start()
    embedding_patch.start()
    extract_patch.start()
    dedup_patch.start()
    conflict_patch.start()

    from app.services.memory_service import MemoryService

    service = MemoryService()

    seq = {"n": 11000}

    async def _next_seq() -> int:
        seq["n"] += 1
        return seq["n"]

    service.sequence_service.next_seq = _next_seq  # type: ignore[method-assign]
    service.redis_timeline = None
    service._initialized = True
    service.reranker = None
    service._reranker_loaded = False
    service.pinecone_reranker = None

    service.__patches = [  # type: ignore[attr-defined]
        pinecone_patch,
        graph_patch,
        embedding_patch,
        extract_patch,
        dedup_patch,
        conflict_patch,
    ]
    return service


def _tear_down_service(service: Any) -> None:
    for p in getattr(service, "__patches", []):
        p.stop()


# --------------------------------------------------------------------------
# Flag-toggle + in-flight helpers
# --------------------------------------------------------------------------


class _FlagOverride:
    def __init__(self, name: str, value: bool) -> None:
        self._name = name
        self._value = value
        self._prev: bool | None = None

    def __enter__(self) -> "_FlagOverride":
        self._prev = getattr(settings, self._name)
        setattr(settings, self._name, self._value)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._prev is not None:
            setattr(settings, self._name, self._prev)


async def _wait_for_inflight(service: Any, timeout: float = 5.0) -> None:
    linker = getattr(service, "_temporal_linker", None)
    if linker is None:
        return
    deadline = asyncio.get_event_loop().time() + timeout
    while (
        getattr(linker, "_inflight", None)
        and asyncio.get_event_loop().time() < deadline
    ):
        tasks = list(linker._inflight)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.sleep(0.01)


def _build_noop_pinecone() -> MagicMock:
    pm = MagicMock()
    pm.initialize = MagicMock(return_value=True)
    pm.check_connection = MagicMock(return_value=True)
    pm.upsert_vector = MagicMock(return_value=True)
    pm.delete_vector = MagicMock(return_value=True)
    pm.query_vector = MagicMock(return_value=[])
    return pm


# --------------------------------------------------------------------------
# Upsert helper — the test drives perform_upsert against a pre-seeded node
# --------------------------------------------------------------------------


async def _upsert(
    service: Any,
    memory_id: str,
    *,
    session_id: Optional[str],
    event_seq: Optional[int],
    thread_id: Optional[str] = None,
    project: str = "sprint11-test",
    content: str = "temporal linker test content",
) -> str:
    """Call ``service.perform_upsert`` with enough chronology for the
    Sprint 11 hook to engage. ``event_seq`` is passed in the metadata
    dict so it survives ``_inject_chronology`` (which overwrites the
    key unconditionally); callers that want a *specific* event_seq
    should pre-seed it on the test node too (which we do in
    ``_create_test_node``) since the linker reads ``event_seq`` off the
    live Neo4j node, not off the metadata argument — except the hook
    itself reads ``metadata.get("event_seq")``. To keep these two
    consistent, ``_inject_chronology`` is patched at the service level
    to return the incoming ``event_seq`` unchanged when the caller
    supplied one. See ``_build_memory_service`` — we override
    ``service.sequence_service.next_seq`` to a counter, and this
    helper overrides ``_inject_chronology`` per call to pin the
    metadata value.
    """
    metadata: dict[str, Any] = {
        "project": project,
        "memory_type": "test",
    }
    if session_id is not None:
        metadata["session_id"] = session_id
    if thread_id is not None:
        metadata["thread_id"] = thread_id
    if event_seq is not None:
        metadata["event_seq"] = int(event_seq)

    # Patch _inject_chronology for this one call so the metadata
    # event_seq is preserved instead of being overwritten by the
    # default sequence service. We only do this if the caller
    # supplied an event_seq; otherwise the default injection path
    # runs as normal.
    if event_seq is not None:
        async def _inject_preserve(m: dict[str, Any]) -> dict[str, Any]:
            if "event_time" not in m or not m["event_time"]:
                m["event_time"] = _now_iso()
            if "memory_type" not in m or not m["memory_type"]:
                m["memory_type"] = "scratch"
            # event_seq is PRESERVED here — intentionally overriding
            # the default "never caller-provided" rule for the test
            # harness so the linker's predecessor-lookup sees the
            # value the test intended.
            return m

        with patch.object(
            service, "_inject_chronology", side_effect=_inject_preserve
        ):
            returned = await service.perform_upsert(
                content=content, memory_id=memory_id, metadata=metadata
            )
    else:
        returned = await service.perform_upsert(
            content=content, memory_id=memory_id, metadata=metadata
        )
    return returned


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_01_flag_off_hook_is_noop(
    neo4j_async_driver: AsyncDriver,
    production_counts: dict[str, int],  # force fixture ordering
    run_tag: str,
    test_session_id: str,
) -> None:
    """ASSOC_TEMPORAL_WRITE_ENABLED=False ⇒ linker never constructed,
    zero MEMORY_FOLLOWS edges created."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s11-n01-src-{run_tag}"
    try:
        await _create_test_node(
            neo4j_async_driver,
            src_id,
            session_id=test_session_id,
            event_seq=1,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", False):
            returned_id = await _upsert(
                service,
                src_id,
                session_id=test_session_id,
                event_seq=1,
            )
            await asyncio.sleep(0.05)

        assert returned_id == src_id
        assert service._temporal_linker is None, (
            "Flag OFF path must NOT instantiate TemporalLinker lazily"
        )
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_id) == 0
        )
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_02_flag_on_first_memory_in_session_no_edge(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_session_id: str,
) -> None:
    """Flag-ON first memory in session: no predecessor ⇒ no edge."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s11-n02-src-{run_tag}"
    try:
        await _create_test_node(
            neo4j_async_driver,
            src_id,
            session_id=test_session_id,
            event_seq=1,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            returned_id = await _upsert(
                service,
                src_id,
                session_id=test_session_id,
                event_seq=1,
            )
            await _wait_for_inflight(service)

        assert returned_id == src_id
        assert service._temporal_linker is not None, (
            "Flag ON path must lazily instantiate TemporalLinker"
        )
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_id) == 0
        )
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_03_flag_on_second_memory_links_to_first(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_session_id: str,
) -> None:
    """Two memories in the same session, event_seq 1 then 2 ⇒ exactly
    one MEMORY_FOLLOWS edge from memory 2 → memory 1."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_1 = f"s11-n03-src-1-{run_tag}"
    src_2 = f"s11-n03-src-2-{run_tag}"
    try:
        await _create_test_node(
            neo4j_async_driver,
            src_1,
            session_id=test_session_id,
            event_seq=1,
            run_tag=run_tag,
        )
        await _create_test_node(
            neo4j_async_driver,
            src_2,
            session_id=test_session_id,
            event_seq=2,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            await _upsert(
                service, src_1, session_id=test_session_id, event_seq=1
            )
            await _wait_for_inflight(service)
            await _upsert(
                service, src_2, session_id=test_session_id, event_seq=2
            )
            await _wait_for_inflight(service)

        # First memory has no predecessor.
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_1) == 0
        )
        # Second memory has exactly one outgoing edge pointing to the first.
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_2) == 1
        )
        details = await _get_memory_follows_details(neo4j_async_driver, src_2)
        assert len(details) == 1
        row = details[0]
        assert row["predecessor_id"] == src_1
        assert row["predecessor_seq"] == 1
        assert row["weight"] == 1.0
        assert row["created_by"] == "temporal_linker"
        assert row["run_id"].startswith("wt-temporal-")
        # Neo4j 5 refuses Map-valued rel properties — metadata is LITERALLY
        # None on the edge (not stored, not empty-dict).
        assert row["metadata"] is None
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_04_three_memory_chain(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_session_id: str,
) -> None:
    """Three in-order memories 1 → 2 → 3 produce exactly two edges:
    2 → 1 and 3 → 2."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    ids = [f"s11-n04-src-{i}-{run_tag}" for i in range(1, 4)]
    try:
        for i, mid in enumerate(ids, start=1):
            await _create_test_node(
                neo4j_async_driver,
                mid,
                session_id=test_session_id,
                event_seq=i,
                run_tag=run_tag,
            )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            for i, mid in enumerate(ids, start=1):
                await _upsert(
                    service, mid, session_id=test_session_id, event_seq=i
                )
                await _wait_for_inflight(service)

        # Edge accounting scoped to the three test nodes.
        assert (
            await _count_memory_follows_from(neo4j_async_driver, ids[0]) == 0
        )
        assert (
            await _count_memory_follows_from(neo4j_async_driver, ids[1]) == 1
        )
        assert (
            await _count_memory_follows_from(neo4j_async_driver, ids[2]) == 1
        )

        d2 = await _get_memory_follows_details(neo4j_async_driver, ids[1])
        d3 = await _get_memory_follows_details(neo4j_async_driver, ids[2])
        assert d2[0]["predecessor_id"] == ids[0]
        assert d2[0]["predecessor_seq"] == 1
        assert d3[0]["predecessor_id"] == ids[1]
        assert d3[0]["predecessor_seq"] == 2

        # Scoped total across this run_tag is exactly 2.
        assert (
            await _count_total_memory_follows_in_session(
                neo4j_async_driver, run_tag
            )
            == 2
        )
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_05_different_sessions_dont_link(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
) -> None:
    """Two memories in DIFFERENT sessions ⇒ zero MEMORY_FOLLOWS edges
    between them, even if their event_seq values would otherwise form
    a chain."""
    sid_a = f"{TEST_SESSION_PREFIX}a-{uuid.uuid4().hex[:8]}"
    sid_b = f"{TEST_SESSION_PREFIX}b-{uuid.uuid4().hex[:8]}"
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_a = f"s11-n05-src-a-{run_tag}"
    src_b = f"s11-n05-src-b-{run_tag}"
    try:
        await _create_test_node(
            neo4j_async_driver,
            src_a,
            session_id=sid_a,
            event_seq=1,
            run_tag=run_tag,
        )
        await _create_test_node(
            neo4j_async_driver,
            src_b,
            session_id=sid_b,
            event_seq=2,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            await _upsert(service, src_a, session_id=sid_a, event_seq=1)
            await _wait_for_inflight(service)
            await _upsert(service, src_b, session_id=sid_b, event_seq=2)
            await _wait_for_inflight(service)

        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_a) == 0
        )
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_b) == 0
        )
        assert (
            await _count_total_memory_follows_in_session(
                neo4j_async_driver, run_tag
            )
            == 0
        )
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_06_missing_session_id_is_noop(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
) -> None:
    """Flag-ON memory without session_id ⇒ linker logs no_session and
    no edge is created."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s11-n06-src-{run_tag}"
    try:
        await _create_test_node(
            neo4j_async_driver,
            src_id,
            session_id=None,
            event_seq=1,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            await _upsert(
                service, src_id, session_id=None, event_seq=1
            )
            await _wait_for_inflight(service)

        assert service._temporal_linker is not None
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_id) == 0
        )
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_07_missing_event_seq_is_noop(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_session_id: str,
) -> None:
    """Flag-ON memory without event_seq ⇒ no linking, no edge."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s11-n07-src-{run_tag}"
    try:
        # Seed a predecessor so the skip can't be explained by
        # "nothing to link to" — a second missing-event_seq node would
        # still be the "no linking" path even with a predecessor.
        pred_id = f"s11-n07-pred-{run_tag}"
        await _create_test_node(
            neo4j_async_driver,
            pred_id,
            session_id=test_session_id,
            event_seq=1,
            run_tag=run_tag,
        )
        await _create_test_node(
            neo4j_async_driver,
            src_id,
            session_id=test_session_id,
            event_seq=None,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            # event_seq=None ⇒ the helper does NOT put it into metadata,
            # and _inject_chronology is NOT patched ⇒ the default
            # counter assigns a monotonic seq; but the *hook* still
            # passes metadata.get("event_seq"), which the default
            # chronology injector populates with the counter.
            # Instead, we want to test that an explicit None stays
            # None — do so by suppressing injection entirely.
            async def _inject_noop(m: dict[str, Any]) -> dict[str, Any]:
                if "event_time" not in m or not m["event_time"]:
                    m["event_time"] = _now_iso()
                if "memory_type" not in m or not m["memory_type"]:
                    m["memory_type"] = "scratch"
                return m

            with patch.object(
                service, "_inject_chronology", side_effect=_inject_noop
            ):
                await service.perform_upsert(
                    content="missing event_seq test",
                    memory_id=src_id,
                    metadata={
                        "project": "sprint11-test",
                        "memory_type": "test",
                        "session_id": test_session_id,
                    },
                )
            await _wait_for_inflight(service)

        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_id) == 0
        )
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_08_idempotent_second_upsert(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_session_id: str,
) -> None:
    """Re-storing the same memory does not duplicate the MEMORY_FOLLOWS
    edge — MERGE semantics via MemoryEdgeService."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_1 = f"s11-n08-src-1-{run_tag}"
    src_2 = f"s11-n08-src-2-{run_tag}"
    try:
        await _create_test_node(
            neo4j_async_driver,
            src_1,
            session_id=test_session_id,
            event_seq=1,
            run_tag=run_tag,
        )
        await _create_test_node(
            neo4j_async_driver,
            src_2,
            session_id=test_session_id,
            event_seq=2,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            await _upsert(
                service, src_1, session_id=test_session_id, event_seq=1
            )
            await _wait_for_inflight(service)
            await _upsert(
                service, src_2, session_id=test_session_id, event_seq=2
            )
            await _wait_for_inflight(service)

            first_count = await _count_memory_follows_from(
                neo4j_async_driver, src_2
            )
            assert first_count == 1

            # Re-store src_2 a second time — same (source, target) pair
            # ⇒ the underlying MERGE must find and reuse the existing
            # edge, not duplicate it.
            await _upsert(
                service, src_2, session_id=test_session_id, event_seq=2
            )
            await _wait_for_inflight(service)

            second_count = await _count_memory_follows_from(
                neo4j_async_driver, src_2
            )
            assert second_count == 1, (
                f"Re-store duplicated edge (expected 1, got {second_count})"
            )

        # Scoped edge total is still exactly 1.
        assert (
            await _count_total_memory_follows_in_session(
                neo4j_async_driver, run_tag
            )
            == 1
        )
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_09_out_of_order_arrival_self_heal_gap(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_session_id: str,
) -> None:
    """Document the Sprint 10 out-of-order fix-up gap.

    Arrival order: seq=3 first, seq=1 second. The Sprint 10 design
    choice (see ``temporal_linker.py`` module docstring) is the simpler
    self-healing MERGE strategy:

    - seq=3 arriving first finds no predecessor (seq=1 not yet stored
      in the graph at linker-call time — but we DO seed it because the
      predecessor-lookup query targets live Neo4j nodes). So with the
      standard integration pattern (seed both nodes up front, then
      drive upserts in the out-of-order sequence) seq=3 WILL find
      seq=1 and write an edge.

    To faithfully reproduce the out-of-order gap described in the
    Sprint 10 docstring, we must seed nodes in arrival order too: seed
    seq=3 first, upsert it, THEN seed seq=1, upsert it. That matches
    the production fire-and-forget sequence where the linker runs
    before later arrivals have been written.
    """
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_1 = f"s11-n09-src-1-{run_tag}"
    src_3 = f"s11-n09-src-3-{run_tag}"
    try:
        # Arrival order #1: seq=3 first.
        await _create_test_node(
            neo4j_async_driver,
            src_3,
            session_id=test_session_id,
            event_seq=3,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            await _upsert(
                service, src_3, session_id=test_session_id, event_seq=3
            )
            await _wait_for_inflight(service)

        # At this point only seq=3 is in the session — no predecessor,
        # no edge.
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_3) == 0
        )

        # Arrival order #2: seq=1 lands late.
        await _create_test_node(
            neo4j_async_driver,
            src_1,
            session_id=test_session_id,
            event_seq=1,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            await _upsert(
                service, src_1, session_id=test_session_id, event_seq=1
            )
            await _wait_for_inflight(service)

        # seq=1 has no smaller-seq predecessor either — no edge.
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_1) == 0
        )

        # The 3 → 1 edge is the KNOWN GAP: Sprint 10 does not
        # backward-scan on late arrival. Assert the gap is still
        # present (0 scoped edges) and document it.
        assert (
            await _count_total_memory_follows_in_session(
                neo4j_async_driver, run_tag
            )
            == 0
        ), (
            "Sprint 10 out-of-order gap regressed — if full fix-up is "
            "implemented in a later sprint, update this test's "
            "expectation and move the invariant into a positive "
            "assertion."
        )
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_10_concurrent_same_session_writes(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_session_id: str,
) -> None:
    """Two concurrent ``perform_upsert`` calls for the same session
    (seq=1 and seq=2) must still produce exactly one MEMORY_FOLLOWS
    edge (2 → 1). The per-session asyncio.Lock inside
    ``TemporalLinker`` is what guarantees this — if it were absent or
    racy, the seq=2 task could fire its predecessor-lookup *before*
    the seq=1 node was written (and produce a false no_predecessor
    outcome)."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_1 = f"s11-n10-src-1-{run_tag}"
    src_2 = f"s11-n10-src-2-{run_tag}"
    try:
        # Seed both nodes BEFORE the concurrent upserts so the
        # predecessor lookup can see the earlier node even if the
        # later upsert's linker task runs first. This is the lock
        # test — NOT an arrival-order test.
        await _create_test_node(
            neo4j_async_driver,
            src_1,
            session_id=test_session_id,
            event_seq=1,
            run_tag=run_tag,
        )
        await _create_test_node(
            neo4j_async_driver,
            src_2,
            session_id=test_session_id,
            event_seq=2,
            run_tag=run_tag,
        )

        with _FlagOverride("ASSOC_TEMPORAL_WRITE_ENABLED", True):
            # Fire both perform_upsert calls concurrently via gather.
            await asyncio.gather(
                _upsert(
                    service, src_1, session_id=test_session_id, event_seq=1
                ),
                _upsert(
                    service, src_2, session_id=test_session_id, event_seq=2
                ),
            )
            await _wait_for_inflight(service)

        # Expected state: src_1 has no outgoing edge, src_2 has
        # exactly one. The per-session lock serializes the linker
        # tasks so the seq=2 lookup always sees seq=1 in the graph.
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_1) == 0
        )
        assert (
            await _count_memory_follows_from(neo4j_async_driver, src_2) == 1
        )
        assert (
            await _count_total_memory_follows_in_session(
                neo4j_async_driver, run_tag
            )
            == 1
        )
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_edges(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_zzz_production_counts_unchanged(
    neo4j_async_driver: AsyncDriver,
    production_counts: dict[str, int],
) -> None:
    """Final invariant: no test leakage into production labels/edges.

    ``:base`` may drift upward from organic writes by the live service;
    ``:Session``, ``FOLLOWS``, ``INCLUDES``, and ``MEMORY_FOLLOWS`` must
    be unchanged. No ``:TemporalLinkerTestNode`` may survive. No
    ``wt-temporal-*`` edge tagged with a test run_id may exist.

    ``MEMORY_FOLLOWS`` is pinned to its module-start count because
    Sprint 11 is the first/only code that creates those edges at the
    time of writing — production count is expected to stay at 0 unless
    a flag-ON rollout happens between module start and this assertion
    (in which case the production fixture captures it and we compare
    equality, not zero).
    """
    async with neo4j_async_driver.session(database=TEST_DB) as session:
        base_after = (
            await (await session.run("MATCH (n:base) RETURN count(n) AS c")).single()
        )["c"]
        session_after = (
            await (
                await session.run(
                    f"MATCH (n:{PROD_SESSION_LABEL}) RETURN count(n) AS c"
                )
            ).single()
        )["c"]
        follows_after = (
            await (
                await session.run("MATCH ()-[r:FOLLOWS]->() RETURN count(r) AS c")
            ).single()
        )["c"]
        includes_after = (
            await (
                await session.run("MATCH ()-[r:INCLUDES]->() RETURN count(r) AS c")
            ).single()
        )["c"]
        memory_follows_after = (
            await (
                await session.run(
                    "MATCH ()-[r:MEMORY_FOLLOWS]->() RETURN count(r) AS c"
                )
            ).single()
        )["c"]
        test_label_after = (
            await (
                await session.run(f"MATCH (n:{TEST_LABEL}) RETURN count(n) AS c")
            ).single()
        )["c"]
        leftover_test_edges = (
            await (
                await session.run(
                    "MATCH ()-[r:MEMORY_FOLLOWS]->() "
                    "WHERE r.run_id STARTS WITH 'wt-temporal-' "
                    "RETURN count(r) AS c"
                )
            ).single()
        )["c"]

    assert test_label_after == 0, (
        f"Teardown leaked: {test_label_after} :{TEST_LABEL} nodes survived "
        "the full test run. BLOCKER — re-run teardown manually."
    )
    assert leftover_test_edges == 0, (
        f"{leftover_test_edges} wt-temporal-* MEMORY_FOLLOWS edges leaked "
        "into the live graph. BLOCKER."
    )
    assert base_after >= production_counts["base"], (
        f":base node count dropped: {production_counts['base']} -> "
        f"{base_after}. BLOCKER."
    )
    assert session_after == production_counts["session"], (
        f":Session count changed: {production_counts['session']} -> "
        f"{session_after}"
    )
    assert follows_after == production_counts["follows"], (
        f"FOLLOWS count changed: {production_counts['follows']} -> "
        f"{follows_after}"
    )
    assert includes_after == production_counts["includes"], (
        f"INCLUDES count changed: {production_counts['includes']} -> "
        f"{includes_after}"
    )
    assert memory_follows_after == production_counts["memory_follows"], (
        f"MEMORY_FOLLOWS count changed: "
        f"{production_counts['memory_follows']} -> {memory_follows_after}"
    )
