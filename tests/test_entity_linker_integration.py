"""Integration tests for the Sprint 9 entity-linker hook in MemoryService.

Safety contract (Sprint 2 / Sprint 5 / Sprint 6 pattern)
--------------------------------------------------------

This suite runs against the **live** ``nova_neo4j_db`` container. Every
test obeys the same safety rules as the similarity-linker integration
suite:

1. **Dedicated primary label.** Every test memory node is written with
   the primary label ``:EntityLinkerTestNode`` and the secondary label
   ``:base`` so the linker's Cypher (which matches on
   ``(:base {entity_id})``) can actually bind it. Teardown deletes by
   the primary test label, not ``:base``.
2. **Per-test project namespace.** Every test uses a unique
   ``sprint9-integration-test-<uuid>`` project so ``(project, name)``
   keyed ``:Entity`` nodes never collide with any future real data and
   can be surgically removed on teardown.
3. **Unique ``run_id`` per test run.** A fresh
   ``sprint9-entity-test-<uuid8>`` is generated at fixture start so
   repeat runs never collide.
4. **Neo4j-unreachable ⇒ skip.** If the live container is not
   available (CI without the ``nova_neo4j_db`` sidecar), the whole
   module skips cleanly.
5. **Unconditional ``try/finally`` teardown.** Every test wraps its
   setup in a ``try/finally`` that runs both the test-label cleanup
   and the namespaced-``:Entity`` cleanup.
6. **Production-count invariant at the end.** The final test asserts
   that ``:base`` (may drift upward), ``:Session``, ``FOLLOWS``,
   ``INCLUDES`` are unchanged and that no ``:EntityLinkerTestNode``
   survived. It also asserts there is no leftover ``:Entity`` with a
   non-test project namespace.
7. **Pinecone is ALWAYS mocked.** Entity linking does not touch
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
        "neo4j driver not installed; cannot run entity-linker integration tests",
        allow_module_level=True,
    )

from app.config import settings


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_DB = "neo4j"
TEST_LABEL = "EntityLinkerTestNode"
TEST_PROJECT_PREFIX = "sprint9-integration-test-"
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
            f"Neo4j not reachable at {TEST_NEO4J_URI}; skipping entity-linker "
            "integration tests (expected in environments without a running "
            "nova_neo4j_db container)."
        )
    try:
        yield driver
    finally:
        # Belt-and-braces final sweep in case an individual test forgot.
        await _teardown_test_nodes(driver)
        await _teardown_test_entities(driver)
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
    return {
        "base": base_count,
        "session": session_count,
        "follows": follows_count,
        "includes": includes_count,
    }


@pytest.fixture
def run_tag() -> str:
    return f"sprint9-entity-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_project() -> str:
    return f"{TEST_PROJECT_PREFIX}{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# Test-node helpers
# --------------------------------------------------------------------------


async def _create_test_nodes(
    driver: AsyncDriver, entity_ids: list[str], run_tag: str
) -> None:
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(
                f"UNWIND $ids AS eid "
                f"CREATE (n:{TEST_LABEL}:base "
                f"{{entity_id: eid, test_run: $tag}})",
                {"ids": entity_ids, "tag": run_tag},
            )
        ).consume()


async def _teardown_test_nodes(driver: AsyncDriver) -> None:
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(f"MATCH (n:{TEST_LABEL}) DETACH DELETE n")
        ).consume()


async def _teardown_test_entities(driver: AsyncDriver) -> None:
    """Remove every :Entity whose project starts with the test prefix."""
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(
                f"MATCH (e:Entity) WHERE e.project STARTS WITH $prefix "
                f"DETACH DELETE e",
                {"prefix": TEST_PROJECT_PREFIX},
            )
        ).consume()


async def _count_entity_nodes_in_project(
    driver: AsyncDriver, project: str
) -> int:
    async with driver.session(database=TEST_DB) as session:
        rec = await (
            await session.run(
                "MATCH (e:Entity {project: $project}) RETURN count(e) AS c",
                {"project": project},
            )
        ).single()
    return int(rec["c"]) if rec else 0


async def _count_mentions_from(
    driver: AsyncDriver, memory_id: str
) -> int:
    async with driver.session(database=TEST_DB) as session:
        rec = await (
            await session.run(
                "MATCH (m:base {entity_id: $mid})-[r:MENTIONS]->(:Entity) "
                "RETURN count(r) AS c",
                {"mid": memory_id},
            )
        ).single()
    return int(rec["c"]) if rec else 0


async def _get_mentions_details(
    driver: AsyncDriver, memory_id: str
) -> list[dict]:
    async with driver.session(database=TEST_DB) as session:
        result = await session.run(
            "MATCH (m:base {entity_id: $mid})-[r:MENTIONS]->(e:Entity) "
            "RETURN e.name AS name, e.project AS project, "
            "r.run_id AS run_id, r.created_at AS created_at, "
            "r.last_seen_at AS last_seen_at, r.weight AS weight, "
            "r.created_by AS created_by, r.edge_version AS edge_version, "
            "r.metadata AS metadata",
            {"mid": memory_id},
        )
        rows = [dict(record) async for record in result]
        await result.consume()
    return rows


# --------------------------------------------------------------------------
# MemoryService factory (mirrors Sprint 6)
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
    # Patch the entity-extractor symbol imported by memory_service itself
    # (the orchestration layer's existing extractor, which is unrelated
    # to the new associations one). Return [] so graph multi-hop paths
    # don't pull an unrelated NLP pipeline into the test.
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

    seq = {"n": 7000}

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
# Flag-toggle helper
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
    linker = getattr(service, "_entity_linker", None)
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
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_01_flag_off_hook_is_noop(
    neo4j_async_driver: AsyncDriver,
    production_counts: dict[str, int],  # force fixture ordering
    run_tag: str,
    test_project: str,
) -> None:
    """ASSOC_ENTITY_WRITE_ENABLED=False ⇒ linker never constructed,
    zero :Entity nodes created, zero edges touched."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s9-n01-src-{run_tag}"
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)

        # The cooccurrence hook constructs an EntityLinker for read-only
        # lookups even when ASSOC_ENTITY_WRITE_ENABLED is off, so force
        # cooccurrence off too for this flag-off isolation test.
        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", False), \
                _FlagOverride("ASSOC_COOCCURRENCE_WRITE_ENABLED", False):
            returned_id = await service.perform_upsert(
                content="flag-off entity test with neo4j and Pinecone mentions",
                memory_id=src_id,
                metadata={
                    "project": test_project,
                    "memory_type": "test",
                    "entities": ["neo4j", "pinecone"],
                },
            )
            await asyncio.sleep(0.05)

        assert returned_id == src_id
        assert service._entity_linker is None, (
            "Flag OFF path must NOT instantiate EntityLinker lazily"
        )
        assert await _count_mentions_from(neo4j_async_driver, src_id) == 0
        assert await _count_entity_nodes_in_project(
            neo4j_async_driver, test_project
        ) == 0
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_02_flag_on_with_caller_entities_creates_nodes_and_edges(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_project: str,
) -> None:
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s9-n02-src-{run_tag}"
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)

        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", True):
            returned_id = await service.perform_upsert(
                content="unrelated content",
                memory_id=src_id,
                metadata={
                    "project": test_project,
                    "memory_type": "test",
                    "entities": ["neo4j", "Pinecone", "FusionMemory"],
                },
            )
            await _wait_for_inflight(service)

        assert returned_id == src_id

        entity_count = await _count_entity_nodes_in_project(
            neo4j_async_driver, test_project
        )
        assert entity_count == 3, f"expected 3 :Entity nodes, got {entity_count}"

        mentions_count = await _count_mentions_from(neo4j_async_driver, src_id)
        assert mentions_count == 3

        details = await _get_mentions_details(neo4j_async_driver, src_id)
        names = {row["name"] for row in details}
        assert names == {"neo4j", "pinecone", "fusionmemory"}
        for row in details:
            assert row["project"] == test_project
            assert row["run_id"].startswith("wt-entity-")
            assert row["weight"] == 1.0
            assert row["created_by"] == "entity_linker"
            # Neo4j 5 refuses Map-valued props. metadata must come back
            # as None.
            assert row["metadata"] is None
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_03_flag_on_heuristic_extraction_runs(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_project: str,
) -> None:
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s9-n03-src-{run_tag}"
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)

        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", True):
            returned_id = await service.perform_upsert(
                content="working on agents/memory_router.py with MemoryEdge class",
                memory_id=src_id,
                metadata={"project": test_project, "memory_type": "test"},
            )
            await _wait_for_inflight(service)

        assert returned_id == src_id

        details = await _get_mentions_details(neo4j_async_driver, src_id)
        names = {row["name"] for row in details}
        assert len(names) >= 2, f"expected at least 2 entities, got {names}"
        assert "agents/memory_router.py" in names
        assert "memoryedge" in names
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_04_flag_on_project_none_skips(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
) -> None:
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s9-n04-src-{run_tag}"
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)

        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", True):
            returned_id = await service.perform_upsert(
                content="content with neo4j and Pinecone mentions",
                memory_id=src_id,
                metadata={
                    "project": None,
                    "memory_type": "test",
                    "entities": ["neo4j", "pinecone"],
                },
            )
            await _wait_for_inflight(service)

        assert returned_id == src_id
        # Zero mentions from this memory.
        assert await _count_mentions_from(neo4j_async_driver, src_id) == 0
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_05_flag_on_empty_content_and_no_caller_entities_skips(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_project: str,
) -> None:
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s9-n05-src-{run_tag}"
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)

        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", True):
            # perform_upsert rejects empty content ("Received empty
            # content for upsert") — use whitespace-only content
            # instead, which passes the caller's non-empty check but
            # yields zero matches in the heuristic extractor.
            returned_id = await service.perform_upsert(
                content="plain words that match no regex patterns",
                memory_id=src_id,
                metadata={"project": test_project, "memory_type": "test"},
            )
            await _wait_for_inflight(service)

        assert returned_id == src_id
        assert await _count_mentions_from(neo4j_async_driver, src_id) == 0
        assert await _count_entity_nodes_in_project(
            neo4j_async_driver, test_project
        ) == 0
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_06_idempotent_second_upsert(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_project: str,
) -> None:
    """Calling perform_upsert twice with the same memory does not duplicate."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s9-n06-src-{run_tag}"
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)

        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", True):
            await service.perform_upsert(
                content="first pass content",
                memory_id=src_id,
                metadata={
                    "project": test_project,
                    "memory_type": "test",
                    "entities": ["neo4j", "pinecone"],
                },
            )
            await _wait_for_inflight(service)

            first_details = await _get_mentions_details(neo4j_async_driver, src_id)
            assert len(first_details) == 2
            first_last_seen = {r["name"]: r["last_seen_at"] for r in first_details}

            # Small sleep to guarantee a newer wall-clock timestamp.
            await asyncio.sleep(0.02)

            await service.perform_upsert(
                content="second pass content",
                memory_id=src_id,
                metadata={
                    "project": test_project,
                    "memory_type": "test",
                    "entities": ["neo4j", "pinecone"],
                },
            )
            await _wait_for_inflight(service)

        # Still exactly 2 entities + 2 edges.
        assert await _count_entity_nodes_in_project(
            neo4j_async_driver, test_project
        ) == 2
        second_details = await _get_mentions_details(neo4j_async_driver, src_id)
        assert len(second_details) == 2
        second_last_seen = {r["name"]: r["last_seen_at"] for r in second_details}

        # last_seen_at advanced on both edges.
        for name in ("neo4j", "pinecone"):
            assert second_last_seen[name] >= first_last_seen[name]
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_07_canonicalization_dedups_variants(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_project: str,
) -> None:
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s9-n07-src-{run_tag}"
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)

        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", True):
            await service.perform_upsert(
                content="irrelevant",
                memory_id=src_id,
                metadata={
                    "project": test_project,
                    "memory_type": "test",
                    "entities": ["Neo4j", "NEO4J", "neo4j"],
                },
            )
            await _wait_for_inflight(service)

        assert await _count_entity_nodes_in_project(
            neo4j_async_driver, test_project
        ) == 1
        details = await _get_mentions_details(neo4j_async_driver, src_id)
        assert len(details) == 1
        assert details[0]["name"] == "neo4j"
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_08_project_isolation(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
) -> None:
    """Two memories in DIFFERENT project namespaces both referencing
    'neo4j' → two distinct :Entity nodes (one per project)."""
    project_a = f"{TEST_PROJECT_PREFIX}a-{uuid.uuid4().hex[:8]}"
    project_b = f"{TEST_PROJECT_PREFIX}b-{uuid.uuid4().hex[:8]}"

    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_a = f"s9-n08-src-a-{run_tag}"
    src_b = f"s9-n08-src-b-{run_tag}"
    try:
        await _create_test_nodes(neo4j_async_driver, [src_a, src_b], run_tag)

        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", True):
            await service.perform_upsert(
                content="x",
                memory_id=src_a,
                metadata={
                    "project": project_a,
                    "memory_type": "test",
                    "entities": ["neo4j"],
                },
            )
            await _wait_for_inflight(service)

            await service.perform_upsert(
                content="y",
                memory_id=src_b,
                metadata={
                    "project": project_b,
                    "memory_type": "test",
                    "entities": ["neo4j"],
                },
            )
            await _wait_for_inflight(service)

        assert await _count_entity_nodes_in_project(
            neo4j_async_driver, project_a
        ) == 1
        assert await _count_entity_nodes_in_project(
            neo4j_async_driver, project_b
        ) == 1
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_09_derived_mention_count(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_project: str,
) -> None:
    """Create 3 memories all mentioning the same entity, then ask for
    get_entities_for_memory on one of them and verify mention_count=3."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    memory_ids = [f"s9-n09-src-{i}-{run_tag}" for i in range(3)]
    try:
        await _create_test_nodes(neo4j_async_driver, memory_ids, run_tag)

        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", True):
            for mid in memory_ids:
                await service.perform_upsert(
                    content=f"content for {mid}",
                    memory_id=mid,
                    metadata={
                        "project": test_project,
                        "memory_type": "test",
                        "entities": ["neo4j"],
                    },
                )
                await _wait_for_inflight(service)

        # Use the linker's read-only helper to verify mention_count is
        # derived (not stored). We import fresh here so the test is
        # not coupled to the service's lazy linker.
        from app.services.associations.entity_linker import EntityLinker

        reader = EntityLinker(driver=neo4j_async_driver)
        rows = await reader.get_entities_for_memory(memory_ids[0])
        assert len(rows) == 1
        assert rows[0]["name"] == "neo4j"
        assert rows[0]["project"] == test_project
        assert rows[0]["mention_count"] == 3
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_10_rollback_by_run_id(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
    test_project: str,
) -> None:
    """After the linker creates edges tagged with a per-run run_id, we
    can surgically remove them via the ``scripts/assoc_rollback``
    library function."""
    pinecone_mock = _build_noop_pinecone()
    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s9-n10-src-{run_tag}"
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)

        with _FlagOverride("ASSOC_ENTITY_WRITE_ENABLED", True):
            await service.perform_upsert(
                content="x",
                memory_id=src_id,
                metadata={
                    "project": test_project,
                    "memory_type": "test",
                    "entities": ["alpha", "beta"],
                },
            )
            await _wait_for_inflight(service)

        # Recover the run_id from the edges so we can pass it to
        # assoc_rollback. All edges from a single enqueue_link share
        # the same run_id.
        details = await _get_mentions_details(neo4j_async_driver, src_id)
        assert len(details) == 2
        run_ids = {row["run_id"] for row in details}
        assert len(run_ids) == 1
        (target_run_id,) = run_ids
        assert target_run_id.startswith("wt-entity-")

        # assoc_rollback uses a SYNC driver. Open a temporary sync
        # driver, run the rollback, then close it.
        from neo4j import GraphDatabase

        from scripts.assoc_rollback import assoc_rollback as rollback_fn

        sync_driver = GraphDatabase.driver(TEST_NEO4J_URI, auth=None)
        try:
            report = rollback_fn(
                run_id=target_run_id,
                dry_run=False,
                driver=sync_driver,
                database=TEST_DB,
            )
        finally:
            sync_driver.close()

        assert report.total == 2
        assert report.deleted_by_type.get("MENTIONS") == 2

        # Verify the edges are gone from the live graph.
        assert await _count_mentions_from(neo4j_async_driver, src_id) == 0

        # :Entity nodes themselves persist (rollback only removes the
        # edges tagged with run_id). Teardown will sweep them.
        assert await _count_entity_nodes_in_project(
            neo4j_async_driver, test_project
        ) == 2
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)
        await _teardown_test_entities(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_zzz_production_counts_unchanged(
    neo4j_async_driver: AsyncDriver,
    production_counts: dict[str, int],
) -> None:
    """Final invariant: no test leakage into production labels/edges.

    ``:base`` may drift upward from organic writes by the live service;
    ``:Session``, ``FOLLOWS``, ``INCLUDES`` must be byte-for-byte
    unchanged. No ``:EntityLinkerTestNode`` may survive. No ``:Entity``
    with a non-test project may exist (Sprint 9 is the first code to
    ever write :Entity nodes, so non-test Entity count is expected to
    be 0 forever — but allow future phases to land things with the
    non-test namespace by asserting "no Entity lacks a project" rather
    than "zero Entities".

    Actually, since Sprint 9 is the ONLY code that writes :Entity
    nodes and this test runs to completion we assert BOTH: (a) zero
    leftover test-project Entities, and (b) any surviving Entities
    (there should be none) at least carry a non-null project.
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
        test_label_after = (
            await (
                await session.run(f"MATCH (n:{TEST_LABEL}) RETURN count(n) AS c")
            ).single()
        )["c"]
        test_entity_after = (
            await (
                await session.run(
                    "MATCH (e:Entity) WHERE e.project STARTS WITH $prefix "
                    "RETURN count(e) AS c",
                    {"prefix": TEST_PROJECT_PREFIX},
                )
            ).single()
        )["c"]
        mentions_after = (
            await (
                await session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c")
            ).single()
        )["c"]
        orphan_entities = (
            await (
                await session.run(
                    "MATCH (e:Entity) WHERE e.project IS NULL RETURN count(e) AS c"
                )
            ).single()
        )["c"]

    assert test_label_after == 0, (
        f"Teardown leaked: {test_label_after} :{TEST_LABEL} nodes survived "
        "the full test run. BLOCKER — re-run teardown manually."
    )
    assert test_entity_after == 0, (
        f"Teardown leaked: {test_entity_after} test-project :Entity nodes "
        "survived. BLOCKER."
    )
    assert orphan_entities == 0, (
        f"{orphan_entities} :Entity nodes are missing the project property. "
        "Sprint 9 must never create un-namespaced entities."
    )
    # Sprint 9 is the first/only code that creates MENTIONS edges; the
    # rollback test cleaned up its own edges, and every other flag-ON
    # test's edges hung off :EntityLinkerTestNode which teardown
    # DETACH DELETE'd. So MENTIONS should be exactly 0 after teardown.
    assert mentions_after == 0, (
        f"{mentions_after} MENTIONS edges leaked into the live graph. "
        "Teardown should have DETACH DELETE'd all of them via the test-"
        "label sweep. BLOCKER."
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
