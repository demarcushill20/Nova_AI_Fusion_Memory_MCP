"""Integration tests for the Sprint 6 similarity-link hook in MemoryService.

Safety contract (Sprint 2 / Sprint 5 pattern)
---------------------------------------------

This suite runs against the **live** ``nova_neo4j_db`` container. That means
production data (real ``:base`` nodes, real ``:Session`` chain, real
``FOLLOWS`` / ``INCLUDES`` edges) is one ``MATCH`` away, so every test obeys
the same rules as Sprint 2's ``test_assoc_rollback.py`` and Sprint 5's
``test_memory_edge_service.py``:

1. **Dedicated primary label.** Every test memory node is written with the
   primary label ``:SimilarityLinkerIntegrationTestNode`` and the
   secondary label ``:base`` so the linker's Cypher (which matches on
   ``(:base {entity_id})``) can actually bind it. Teardown deletes by the
   primary test label, not ``:base``.
2. **Unique ``run_id`` per test run.** A fresh
   ``sprint6-sim-link-test-<uuid8>`` is generated at module-scoped fixture
   start so repeat runs never collide.
3. **Neo4j-unreachable ⇒ skip.** If the live container is not available
   (CI without the ``nova_neo4j_db`` sidecar), the whole module skips.
4. **Unconditional ``try/finally`` teardown.** Every test wraps its setup
   in a ``try/finally`` that runs
   ``MATCH (n:SimilarityLinkerIntegrationTestNode) DETACH DELETE n``.
5. **Production-count invariant at the end.** The last test asserts
   ``:base``/``:Session``/``FOLLOWS``/``INCLUDES`` counts are unchanged
   (``:base`` may drift upward organically, like in Sprint 5's suite).
6. **Pinecone is ALWAYS mocked.** The test never writes to the real
   Pinecone index. Every integration test builds a ``MagicMock`` pinecone
   client and passes it into a ``MemoryService`` fixture that patches
   ``PineconeClient`` at import time.

What these tests cover
----------------------

Sprint 6's acceptance criteria call for 6 integration-level scenarios.
Each is implemented as its own ``test_NN_*`` function so failure isolation
is surgical:

1. Flag OFF ⇒ hook is a no-op: no linker created, no edges touched, no
   behavior change vs. the Phase 0 zero-regression baseline.
2. Flag ON, candidates found ⇒ ``SIMILAR_TO`` edges land in Neo4j.
3. Flag ON, zero candidates ⇒ zero edges, zero exceptions.
4. Flag ON, Pinecone raises ⇒ store still succeeds, zero edges.
5. Flag ON, project scoping ⇒ cross-project candidates are filtered out.
6. Production counts unchanged at teardown.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional
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
        "neo4j driver not installed; cannot run similarity-linker integration tests",
        allow_module_level=True,
    )

from app.config import settings


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_DB = "neo4j"
TEST_LABEL = "SimilarityLinkerIntegrationTestNode"
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
    """Open an async driver if Neo4j is reachable, else return None."""
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
            f"Neo4j not reachable at {TEST_NEO4J_URI}; skipping similarity-linker "
            "integration tests (expected in environments without a running "
            "nova_neo4j_db container)."
        )
    try:
        yield driver
    finally:
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
                await session.run(
                    "MATCH ()-[r:FOLLOWS]->() RETURN count(r) AS c"
                )
            ).single()
        )["c"]
        includes_count = (
            await (
                await session.run(
                    "MATCH ()-[r:INCLUDES]->() RETURN count(r) AS c"
                )
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
    """Per-test unique tag so teardown is surgical even on repeat runs."""
    return f"sprint6-sim-link-test-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# Test-node helpers (Sprint 5 pattern)
# --------------------------------------------------------------------------


async def _create_test_nodes(
    driver: AsyncDriver, entity_ids: list[str], run_tag: str
) -> None:
    """Create multi-label test nodes for Cypher matching compatibility."""
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
            await session.run(
                f"MATCH (n:{TEST_LABEL}) DETACH DELETE n"
            )
        ).consume()


async def _count_similar_to_edges_from(
    driver: AsyncDriver, source_id: str
) -> int:
    """Count ``SIMILAR_TO`` edges on a node, in either direction (bidir canon)."""
    async with driver.session(database=TEST_DB) as session:
        rec = await (
            await session.run(
                f"MATCH (a:{TEST_LABEL} {{entity_id: $src}})"
                f"-[r:SIMILAR_TO]-"
                f"(b:{TEST_LABEL}) "
                f"RETURN count(r) AS c",
                {"src": source_id},
            )
        ).single()
    return int(rec["c"]) if rec else 0


# --------------------------------------------------------------------------
# MemoryService factory
# --------------------------------------------------------------------------


def _build_memory_service(
    *,
    live_neo4j_driver: AsyncDriver,
    pinecone_mock: MagicMock,
) -> Any:
    """Construct a patched MemoryService wired to the live Neo4j + mock Pinecone.

    The returned service has:
      - the real ``GraphClient`` replaced so that ``graph_client.driver``
        points at the module-scoped live async driver (the one used for
        test-node creation and teardown). This lets the similarity linker
        reach the same test graph;
      - ``graph_client.upsert_graph_data`` mocked as a no-op AsyncMock so
        ``perform_upsert()`` doesn't actually touch production ``:base``
        nodes — the test pre-seeds any nodes it needs via
        ``_create_test_nodes``;
      - ``get_embedding`` replaced with a deterministic fake;
      - ``_initialized`` forced True and redis/rerankers disabled.
    """
    # Patch PineconeClient to return the mock.
    pinecone_patch = patch(
        "app.services.memory_service.PineconeClient",
        return_value=pinecone_mock,
    )
    # GraphClient replacement: a MagicMock that exposes ``driver`` pointing
    # at the live async driver. upsert_graph_data is a no-op so the test
    # fully controls which nodes exist.
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
    extract_patch = patch(
        "app.services.memory_service.extract_entities",
        side_effect=lambda _t: [],
    )

    # Disable the P9A.5 write-time dedup / conflict-detection path. Those
    # features run inside perform_upsert() BEFORE the hook and make their
    # own ``pinecone.query_vector`` call, which would (a) trip the test
    # mocks that expect zero query_vector calls in the flag-OFF case and
    # (b) return a stale "duplicate" on repeat runs. They are unrelated
    # to Sprint 6's similarity linker and the integration suite turns
    # them off for the duration of each test.
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

    # Make sequence deterministic (value is irrelevant beyond not drifting).
    seq = {"n": 5000}

    async def _next_seq() -> int:
        seq["n"] += 1
        return seq["n"]

    service.sequence_service.next_seq = _next_seq  # type: ignore[method-assign]
    service.redis_timeline = None
    service._initialized = True
    service.reranker = None
    service._reranker_loaded = False
    service.pinecone_reranker = None

    # Stash patches on the service instance so the test can stop them.
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
# Flag-toggle helper — restores the previous flag value in a try/finally.
# --------------------------------------------------------------------------


class _FlagOverride:
    """Temporarily set a Settings boolean, restoring the prior value."""

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


async def _wait_for_inflight(service: Any, timeout: float = 3.0) -> None:
    """Wait for the lazily-constructed similarity linker to drain its tasks."""
    linker = getattr(service, "_similarity_linker", None)
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


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_01_flag_off_hook_is_noop(
    neo4j_async_driver: AsyncDriver,
    production_counts: dict[str, int],  # force fixture ordering
    run_tag: str,
) -> None:
    """ASSOC_SIMILARITY_WRITE_ENABLED=False ⇒ linker never constructed,
    zero edges created, zero behavior change."""
    pinecone_mock = MagicMock()
    pinecone_mock.initialize = MagicMock(return_value=True)
    pinecone_mock.check_connection = MagicMock(return_value=True)
    pinecone_mock.upsert_vector = MagicMock(return_value=True)
    pinecone_mock.delete_vector = MagicMock(return_value=True)
    # query_vector should NEVER be called when the flag is off. We give
    # it a loud side-effect so the test fails immediately if it is.
    pinecone_mock.query_vector = MagicMock(
        side_effect=AssertionError(
            "query_vector called despite ASSOC_SIMILARITY_WRITE_ENABLED=False"
        )
    )

    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    src_id = f"s6-n01-src-{run_tag}"
    try:
        # Pre-seed the source node (and a couple of candidates, none of
        # which should ever be touched because the hook is off).
        await _create_test_nodes(
            neo4j_async_driver,
            [src_id, f"s6-n01-cand-a-{run_tag}", f"s6-n01-cand-b-{run_tag}"],
            run_tag,
        )

        with _FlagOverride("ASSOC_SIMILARITY_WRITE_ENABLED", False):
            returned_id = await service.perform_upsert(
                content="flag-off test memory content",
                memory_id=src_id,
                metadata={"project": "sprint6-test", "memory_type": "test"},
            )
            # Give any hypothetical (buggy) background task a chance.
            await asyncio.sleep(0.05)

        # Contract 1: the return value is exactly the src_id (store worked).
        assert returned_id == src_id

        # Contract 2: linker was never lazily constructed.
        assert service._similarity_linker is None, (
            "Flag OFF path must NOT instantiate SimilarityLinker lazily"
        )

        # Contract 3: zero SIMILAR_TO edges touch the test node.
        edge_count = await _count_similar_to_edges_from(
            neo4j_async_driver, src_id
        )
        assert edge_count == 0

        # Contract 4: Pinecone.query_vector was never called (the mock
        # would have raised). Just confirm the mock was still alive.
        pinecone_mock.query_vector.assert_not_called()
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_02_flag_on_with_candidates_creates_edges(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
) -> None:
    """ASSOC_SIMILARITY_WRITE_ENABLED=True + 3 above-threshold candidates ⇒
    3 ``SIMILAR_TO`` edges visible on the test subgraph."""
    src_id = f"s6-n02-src-{run_tag}"
    cand_ids = [
        f"s6-n02-cand-a-{run_tag}",
        f"s6-n02-cand-b-{run_tag}",
        f"s6-n02-cand-c-{run_tag}",
    ]

    def _candidates(*args: Any, **kwargs: Any) -> list[dict]:
        return [
            {"id": cand_ids[0], "score": 0.95, "metadata": {}},
            {"id": cand_ids[1], "score": 0.89, "metadata": {}},
            {"id": cand_ids[2], "score": 0.84, "metadata": {}},
        ]

    pinecone_mock = MagicMock()
    pinecone_mock.initialize = MagicMock(return_value=True)
    pinecone_mock.check_connection = MagicMock(return_value=True)
    pinecone_mock.upsert_vector = MagicMock(return_value=True)
    pinecone_mock.delete_vector = MagicMock(return_value=True)
    pinecone_mock.query_vector = MagicMock(side_effect=_candidates)

    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    try:
        # Pre-seed source + candidates.
        await _create_test_nodes(
            neo4j_async_driver, [src_id] + cand_ids, run_tag
        )

        with _FlagOverride("ASSOC_SIMILARITY_WRITE_ENABLED", True):
            returned_id = await service.perform_upsert(
                content="flag-on test memory with similar candidates",
                memory_id=src_id,
                metadata={"project": "sprint6-test", "memory_type": "test"},
            )
            await _wait_for_inflight(service)

        assert returned_id == src_id

        # Contract: exactly 3 SIMILAR_TO edges touching the source.
        edge_count = await _count_similar_to_edges_from(
            neo4j_async_driver, src_id
        )
        assert edge_count == 3, (
            f"expected 3 SIMILAR_TO edges, got {edge_count}"
        )

        # Contract: Pinecone was queried with top_k = CANDIDATE_POOL+1
        # and a project filter (cross_project default = False).
        pinecone_mock.query_vector.assert_called()
        kwargs = pinecone_mock.query_vector.call_args.kwargs
        assert kwargs["top_k"] == 31
        assert kwargs["filter"] == {"project": "sprint6-test"}
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_03_flag_on_with_no_candidates_creates_no_edges(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
) -> None:
    src_id = f"s6-n03-src-{run_tag}"

    pinecone_mock = MagicMock()
    pinecone_mock.initialize = MagicMock(return_value=True)
    pinecone_mock.check_connection = MagicMock(return_value=True)
    pinecone_mock.upsert_vector = MagicMock(return_value=True)
    pinecone_mock.delete_vector = MagicMock(return_value=True)
    pinecone_mock.query_vector = MagicMock(return_value=[])

    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)
        with _FlagOverride("ASSOC_SIMILARITY_WRITE_ENABLED", True):
            returned_id = await service.perform_upsert(
                content="no similar candidates in the graph",
                memory_id=src_id,
                metadata={"project": "sprint6-test", "memory_type": "test"},
            )
            await _wait_for_inflight(service)

        assert returned_id == src_id
        edge_count = await _count_similar_to_edges_from(
            neo4j_async_driver, src_id
        )
        assert edge_count == 0
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_04_flag_on_pinecone_raises_does_not_break_store(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
) -> None:
    src_id = f"s6-n04-src-{run_tag}"

    pinecone_mock = MagicMock()
    pinecone_mock.initialize = MagicMock(return_value=True)
    pinecone_mock.check_connection = MagicMock(return_value=True)
    pinecone_mock.upsert_vector = MagicMock(return_value=True)
    pinecone_mock.delete_vector = MagicMock(return_value=True)
    pinecone_mock.query_vector = MagicMock(
        side_effect=RuntimeError("pinecone exploded")
    )

    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    try:
        await _create_test_nodes(neo4j_async_driver, [src_id], run_tag)
        with _FlagOverride("ASSOC_SIMILARITY_WRITE_ENABLED", True):
            # perform_upsert must still return the id — fail-open.
            returned_id = await service.perform_upsert(
                content="pinecone is down — store must still succeed",
                memory_id=src_id,
                metadata={"project": "sprint6-test", "memory_type": "test"},
            )
            await _wait_for_inflight(service)

        assert returned_id == src_id
        edge_count = await _count_similar_to_edges_from(
            neo4j_async_driver, src_id
        )
        assert edge_count == 0
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_05_flag_on_project_scoping_filters_candidates(
    neo4j_async_driver: AsyncDriver,
    run_tag: str,
) -> None:
    """With cross_project_enabled=False (default), a project-mismatched
    candidate list (simulated via the mock returning [] when the filter
    is set) produces zero edges."""
    src_id = f"s6-n05-src-{run_tag}"
    other_project_id = f"s6-n05-other-{run_tag}"

    # The mock inspects the filter arg: if filter={"project": "A"}, return
    # an empty list (simulating project-scoped Pinecone). If no filter or
    # a different filter, it would return a candidate — but we never call
    # it that way.
    def _filtered(*args: Any, **kwargs: Any) -> list[dict]:
        f = kwargs.get("filter")
        if f == {"project": "sprint6-test-project-A"}:
            return []
        # Cross-project would see the candidate — but the default flag is
        # cross_project_enabled=False so the linker passes the filter
        # above. If we ever reach this branch, the test is broken.
        return [{"id": other_project_id, "score": 0.99, "metadata": {}}]

    pinecone_mock = MagicMock()
    pinecone_mock.initialize = MagicMock(return_value=True)
    pinecone_mock.check_connection = MagicMock(return_value=True)
    pinecone_mock.upsert_vector = MagicMock(return_value=True)
    pinecone_mock.delete_vector = MagicMock(return_value=True)
    pinecone_mock.query_vector = MagicMock(side_effect=_filtered)

    service = _build_memory_service(
        live_neo4j_driver=neo4j_async_driver,
        pinecone_mock=pinecone_mock,
    )
    try:
        # Seed source AND the other-project candidate, but we expect
        # zero edges because of the project filter.
        await _create_test_nodes(
            neo4j_async_driver, [src_id, other_project_id], run_tag
        )
        with _FlagOverride("ASSOC_SIMILARITY_WRITE_ENABLED", True):
            with _FlagOverride("ASSOC_CROSS_PROJECT_ENABLED", False):
                returned_id = await service.perform_upsert(
                    content="project-isolation test content",
                    memory_id=src_id,
                    metadata={
                        "project": "sprint6-test-project-A",
                        "memory_type": "test",
                    },
                )
                await _wait_for_inflight(service)

        assert returned_id == src_id
        edge_count = await _count_similar_to_edges_from(
            neo4j_async_driver, src_id
        )
        assert edge_count == 0

        # Also confirm the mock was actually called with the project filter.
        pinecone_mock.query_vector.assert_called()
        kwargs = pinecone_mock.query_vector.call_args.kwargs
        assert kwargs["filter"] == {"project": "sprint6-test-project-A"}
    finally:
        _tear_down_service(service)
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_zzz_production_counts_unchanged(
    neo4j_async_driver: AsyncDriver,
    production_counts: dict[str, int],
) -> None:
    """Final invariant: no test leakage into production labels/edges.

    ``:base`` may drift upward from organic writes by the live Fusion Memory
    service. ``:Session``, ``FOLLOWS``, ``INCLUDES`` must be byte-for-byte
    unchanged — Sprint 6 never writes to any of those.
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
                await session.run(
                    "MATCH ()-[r:FOLLOWS]->() RETURN count(r) AS c"
                )
            ).single()
        )["c"]
        includes_after = (
            await (
                await session.run(
                    "MATCH ()-[r:INCLUDES]->() RETURN count(r) AS c"
                )
            ).single()
        )["c"]
        test_label_after = (
            await (
                await session.run(
                    f"MATCH (n:{TEST_LABEL}) RETURN count(n) AS c"
                )
            ).single()
        )["c"]

    assert test_label_after == 0, (
        f"Teardown leaked: {test_label_after} :{TEST_LABEL} nodes survived "
        "the full test run. BLOCKER — re-run teardown manually."
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
