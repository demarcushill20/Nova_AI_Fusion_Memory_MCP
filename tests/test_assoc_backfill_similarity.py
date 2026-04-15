"""Integration tests for PLAN-0759 Step 2.6 — ``scripts/assoc_backfill_similarity``.

Safety model (Sprint 2 / Sprint 5 / Sprint 6 pattern)
-----------------------------------------------------

This suite runs against the **live** ``nova_neo4j_db`` container. Production
data is one ``MATCH`` away, so every test obeys the same safety rules as
``test_assoc_rollback.py``, ``test_memory_edge_service.py``, and
``test_similarity_linker_integration.py``:

1. **Dedicated multi-label.** Every test memory node is written with the
   primary label ``:BackfillTestNode`` and the secondary label ``:base``
   so the backfill's paginated ``MATCH (n:base)`` can bind it. Teardown
   deletes by the primary test label, not ``:base``.
2. **Unique ``run_id`` per test.** Each test generates a fresh
   ``sprint7-backfill-test-<uuid8>`` so repeat runs never collide. The
   library function's ``_allow_test_run_id=True`` kwarg bypasses the
   Sprint-prefix refusal for these identifiers only.
3. **Neo4j-unreachable ⇒ skip.** CI without the container must not fail.
4. **Unconditional ``try/finally`` teardown.** Every test wraps its setup
   in a ``try/finally`` that runs
   ``MATCH (n:BackfillTestNode) DETACH DELETE n``.
5. **Production-count invariant at the end.** The final test asserts
   ``:base``, ``:Session``, ``FOLLOWS``, ``INCLUDES`` counts are unchanged.
6. **Pinecone is ALWAYS mocked.** The test never writes to or reads from
   the real Pinecone index. Fetches are replaced via
   ``fetch_embedding_override`` and query_vector is a ``MagicMock``.

Coverage
--------
The Sprint 7 spec calls for ≥ 12 integration tests. This suite implements
14 and a final production-invariant check.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

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
except ImportError:  # pragma: no cover
    pytest.skip(
        "neo4j driver not installed; cannot run backfill integration tests",
        allow_module_level=True,
    )

from scripts.assoc_backfill_similarity import (
    BackfillError,
    BackfillReport,
    backfill_similarity_edges,
    _default_checkpoint_path,
    _validate_run_id,
)
from scripts.assoc_rollback import assoc_rollback


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_DB = "neo4j"
TEST_LABEL = "BackfillTestNode"
PROD_SESSION_LABEL = "Session"


def _mk_test_run_id() -> str:
    return f"sprint7-backfill-test-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# Async driver fixtures
# --------------------------------------------------------------------------


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


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def neo4j_async_driver() -> AsyncIterator[AsyncDriver]:
    driver = await _try_open_async_driver()
    if driver is None:
        pytest.skip(
            f"Neo4j not reachable at {TEST_NEO4J_URI}; skipping backfill "
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
) -> Dict[str, int]:
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


# --------------------------------------------------------------------------
# Test-node helpers
# --------------------------------------------------------------------------


async def _create_test_nodes(
    driver: AsyncDriver,
    nodes: List[Tuple[str, str]],  # list of (entity_id, project)
    run_tag: str,
) -> None:
    """Create :BackfillTestNode:base nodes with explicit project tags."""
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(
                f"UNWIND $rows AS row "
                f"CREATE (n:{TEST_LABEL}:base {{"
                f"  entity_id: row.eid, "
                f"  project: row.project, "
                f"  test_run: $tag"
                f"}})",
                {
                    "rows": [
                        {"eid": eid, "project": proj} for eid, proj in nodes
                    ],
                    "tag": run_tag,
                },
            )
        ).consume()


async def _teardown_test_nodes(driver: AsyncDriver) -> None:
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(f"MATCH (n:{TEST_LABEL}) DETACH DELETE n")
        ).consume()


async def _count_similar_edges_by_run_id(
    driver: AsyncDriver, run_id: str
) -> int:
    async with driver.session(database=TEST_DB) as session:
        rec = await (
            await session.run(
                "MATCH ()-[r:SIMILAR_TO]->() "
                "WHERE r.run_id = $run_id "
                "RETURN count(r) AS c",
                {"run_id": run_id},
            )
        ).single()
    return int(rec["c"]) if rec else 0


async def _count_test_label_nodes(driver: AsyncDriver) -> int:
    async with driver.session(database=TEST_DB) as session:
        rec = await (
            await session.run(f"MATCH (n:{TEST_LABEL}) RETURN count(n) AS c")
        ).single()
    return int(rec["c"]) if rec else 0


# --------------------------------------------------------------------------
# Mock Pinecone helpers
# --------------------------------------------------------------------------


def _make_candidates(ids_and_scores: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    return [
        {"id": mid, "score": score, "metadata": {}}
        for mid, score in ids_and_scores
    ]


def _build_pinecone_mock(
    query_side_effect: Any = None,
    query_return: Any = None,
) -> MagicMock:
    """Build a MagicMock Pinecone client suitable for the backfill path."""
    mock = MagicMock()
    mock.initialize = MagicMock(return_value=True)
    mock.check_connection = MagicMock(return_value=True)
    mock.upsert_vector = MagicMock(return_value=True)
    if query_side_effect is not None:
        mock.query_vector = MagicMock(side_effect=query_side_effect)
    else:
        mock.query_vector = MagicMock(return_value=query_return or [])
    # Give it a dummy index attr so _fetch_embedding (if called) would
    # traverse it — but every test provides fetch_embedding_override so
    # this should never be touched.
    mock.index = MagicMock()
    mock.index.fetch = MagicMock(
        side_effect=AssertionError(
            "Real pinecone.fetch called in integration test; override missing"
        )
    )
    return mock


async def _noop_fetch_embedding(
    _pinecone: Any, _memory_id: str
) -> Optional[List[float]]:
    # Deterministic dummy 1536-d vector — the backfill passes it straight to
    # the mock query_vector, which ignores its content.
    return [0.01] * 1536


async def _fetch_embedding_missing_for(
    missing_ids: List[str],
) -> Any:
    """Return a fetch_embedding_override that returns None for specified ids."""

    async def _fn(_pinecone: Any, memory_id: str) -> Optional[List[float]]:
        if memory_id in missing_ids:
            return None
        return [0.01] * 1536

    return _fn


# --------------------------------------------------------------------------
# Unit-level: run_id validation refusals
# --------------------------------------------------------------------------


def test_09_run_id_validation_rejects_empty() -> None:
    with pytest.raises(BackfillError):
        _validate_run_id("")


def test_09b_run_id_validation_rejects_whitespace() -> None:
    with pytest.raises(BackfillError):
        _validate_run_id("   ")


@pytest.mark.parametrize("wildcard", ["*", "%", "all", "ALL"])
def test_10_run_id_validation_rejects_wildcard(wildcard: str) -> None:
    with pytest.raises(BackfillError):
        _validate_run_id(wildcard)


def test_11_run_id_validation_rejects_wt_link_prefix() -> None:
    with pytest.raises(BackfillError):
        _validate_run_id("wt-link-abc123")


@pytest.mark.parametrize(
    "prefix", ["sprint2-abc", "sprint5-abc", "sprint6-abc", "sprint7-abc"]
)
def test_11b_run_id_validation_rejects_reserved_prefix(prefix: str) -> None:
    with pytest.raises(BackfillError):
        _validate_run_id(prefix)


def test_11c_run_id_validation_allows_test_prefix_when_flag_set() -> None:
    result = _validate_run_id("sprint7-backfill-test-abc", allow_test=True)
    assert result == "sprint7-backfill-test-abc"


# --------------------------------------------------------------------------
# Integration tests against live Neo4j + mocked Pinecone
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_01_dry_run_produces_count_without_writes(
    neo4j_async_driver: AsyncDriver,
    production_counts: Dict[str, int],  # force fixture ordering
) -> None:
    """5 memories, each with 3 above-threshold candidates, dry-run yields
    count = 15 and writes zero edges."""
    run_id = _mk_test_run_id()
    src_ids = [f"s7-01-src-{i}-{run_id[-8:]}" for i in range(5)]
    proj = "sprint7-test"

    def _candidates(**_kwargs: Any) -> List[Dict[str, Any]]:
        return _make_candidates(
            [
                (f"s7-01-cand-a-{run_id[-8:]}", 0.95),
                (f"s7-01-cand-b-{run_id[-8:]}", 0.89),
                (f"s7-01-cand-c-{run_id[-8:]}", 0.84),
            ]
        )

    pinecone_mock = _build_pinecone_mock(query_side_effect=_candidates)

    try:
        # Seed only the source memories; candidates don't need to exist in
        # the graph for the dry run because we never invoke MERGE.
        await _create_test_nodes(
            neo4j_async_driver, [(eid, proj) for eid in src_ids], run_id
        )

        report = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            dry_run=True,
            rate_limit_qps=0,  # disable rate limit for speed
            project_filter=proj,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )

        assert report.dry_run is True
        assert report.memories_processed == 5
        assert report.edges_created == 15
        assert report.by_project[proj]["processed"] == 5
        assert report.by_project[proj]["edges"] == 15

        # Contract: ZERO edges actually landed in Neo4j despite dry run.
        real_edges = await _count_similar_edges_by_run_id(
            neo4j_async_driver, f"backfill-{run_id}"
        )
        assert real_edges == 0
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_02_live_run_creates_edges(
    neo4j_async_driver: AsyncDriver,
) -> None:
    """Live run with 5 src + 3 candidates each yields 15 tagged edges.

    Candidates live in a DIFFERENT project so the backfill's project_filter
    iterates only the source nodes — otherwise the backfill would walk
    every ``:base`` node in the test graph and double-count. The MERGE
    only requires both endpoints to be ``:base``; it does not constrain
    them by project.
    """
    run_id = _mk_test_run_id()
    proj_src = "sprint7-test-src"
    proj_cand = "sprint7-test-cand"
    src_ids = [f"s7-02-src-{i}-{run_id[-8:]}" for i in range(5)]
    cand_ids = [
        f"s7-02-cand-a-{run_id[-8:]}",
        f"s7-02-cand-b-{run_id[-8:]}",
        f"s7-02-cand-c-{run_id[-8:]}",
    ]

    def _candidates(**_kwargs: Any) -> List[Dict[str, Any]]:
        return _make_candidates(
            [(cand_ids[0], 0.95), (cand_ids[1], 0.89), (cand_ids[2], 0.84)]
        )

    pinecone_mock = _build_pinecone_mock(query_side_effect=_candidates)

    try:
        # Seed sources in proj_src, candidates in proj_cand. The
        # project_filter restricts iteration to sources only; MERGE still
        # binds candidates via the (:base {entity_id}) lookup regardless
        # of their project property.
        await _create_test_nodes(
            neo4j_async_driver,
            [(eid, proj_src) for eid in src_ids]
            + [(cid, proj_cand) for cid in cand_ids],
            run_id,
        )

        report = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            dry_run=False,
            rate_limit_qps=0,
            project_filter=proj_src,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )

        assert report.memories_processed == 5
        assert report.edges_created == 15

        tagged = await _count_similar_edges_by_run_id(
            neo4j_async_driver, f"backfill-{run_id}"
        )
        assert tagged == 15
    finally:
        # Clean up edges + nodes. delete_edges via run_id first, then DETACH DELETE.
        assoc_rollback(
            run_id=f"backfill-{run_id}",
            dry_run=False,
            driver=None,
            uri=TEST_NEO4J_URI,
            database=TEST_DB,
        )
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_03_rate_limit_is_respected(
    neo4j_async_driver: AsyncDriver,
) -> None:
    """rate_limit_qps=2.0 over 5 memories should take at least ~2s wall-clock.

    The backfill makes 2 Pinecone calls per memory (fetch + query_vector),
    so 5 memories = 10 calls. At 2 QPS the minimum wall-clock for 10 calls
    is (10-1) * 0.5 = 4.5s. We use a conservative 2s floor to keep CI
    from flaking while still proving the rate limiter is engaged (a naive
    unbounded run finishes in < 0.5s on the same hardware).
    """
    run_id = _mk_test_run_id()
    proj = "sprint7-test"
    src_ids = [f"s7-03-src-{i}-{run_id[-8:]}" for i in range(5)]

    pinecone_mock = _build_pinecone_mock(query_return=[])

    try:
        await _create_test_nodes(
            neo4j_async_driver, [(eid, proj) for eid in src_ids], run_id
        )

        t0 = time.monotonic()
        report = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            dry_run=False,
            rate_limit_qps=2.0,
            project_filter=proj,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )
        elapsed = time.monotonic() - t0

        assert report.memories_processed == 5
        # Lower bound: with 10 rate-limited calls at 2 QPS, elapsed must be
        # at least 2 seconds (minus some leeway for the first call being free).
        assert elapsed >= 2.0, (
            f"Rate limit was not respected: {elapsed:.2f}s < 2.0s for "
            f"10 calls at 2 QPS"
        )
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_04_max_total_limit(
    neo4j_async_driver: AsyncDriver,
) -> None:
    """10 memories, max_total=3 ⇒ exactly 3 processed."""
    run_id = _mk_test_run_id()
    proj = "sprint7-test"
    src_ids = [f"s7-04-src-{i:02d}-{run_id[-8:]}" for i in range(10)]

    pinecone_mock = _build_pinecone_mock(query_return=[])

    try:
        await _create_test_nodes(
            neo4j_async_driver, [(eid, proj) for eid in src_ids], run_id
        )

        report = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            max_total=3,
            dry_run=False,
            rate_limit_qps=0,
            project_filter=proj,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )

        assert report.memories_processed == 3
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_05_resume_from_cursor(
    neo4j_async_driver: AsyncDriver,
) -> None:
    """10 memories, resume_from = #4 ⇒ memories 5-9 processed, 0-4 skipped.

    Uses zero-padded entity_ids so the lexicographic cursor walk is
    deterministic.
    """
    run_id = _mk_test_run_id()
    proj = "sprint7-test"
    src_ids = sorted([f"s7-05-src-{i:02d}-{run_id[-8:]}" for i in range(10)])

    pinecone_mock = _build_pinecone_mock(query_return=[])

    try:
        await _create_test_nodes(
            neo4j_async_driver, [(eid, proj) for eid in src_ids], run_id
        )

        # resume_from = src_ids[4]. The contract is "skip up to AND
        # including this id", so src_ids[5:] should be processed.
        report = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            dry_run=False,
            rate_limit_qps=0,
            resume_from=src_ids[4],
            project_filter=proj,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )

        assert report.memories_scanned == 10
        assert report.memories_processed == 5, (
            f"Expected 5 memories processed after resume cursor, "
            f"got {report.memories_processed}"
        )
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_06_checkpoint_written_every_100(
    neo4j_async_driver: AsyncDriver,
    tmp_path: Path,
) -> None:
    """250 memories processed ⇒ final checkpoint count is 250 (writes at 100, 200, and final)."""
    run_id = _mk_test_run_id()
    proj = "sprint7-test"
    src_ids = sorted([f"s7-06-src-{i:04d}-{run_id[-8:]}" for i in range(250)])

    pinecone_mock = _build_pinecone_mock(query_return=[])

    checkpoint_path = str(tmp_path / f"backfill_{run_id}.checkpoint")

    try:
        await _create_test_nodes(
            neo4j_async_driver, [(eid, proj) for eid in src_ids], run_id
        )

        report = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=100,
            dry_run=False,
            rate_limit_qps=0,
            project_filter=proj,
            checkpoint_path=checkpoint_path,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )

        assert report.memories_processed == 250

        # Final checkpoint file must exist and contain count=250.
        assert os.path.exists(checkpoint_path)
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            cp = json.load(fh)
        assert cp["count"] == 250
        assert cp["run_id"] == run_id
        # The last processed memory_id must be the final src_id.
        assert cp["last_processed_memory_id"] == src_ids[-1]
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_07_project_filter(
    neo4j_async_driver: AsyncDriver,
) -> None:
    """10 memories in two projects, filter to one ⇒ only 5 processed."""
    run_id = _mk_test_run_id()
    proj_a = "sprint7-A"
    proj_b = "sprint7-B"
    src_a = [f"s7-07-A-{i}-{run_id[-8:]}" for i in range(5)]
    src_b = [f"s7-07-B-{i}-{run_id[-8:]}" for i in range(5)]

    pinecone_mock = _build_pinecone_mock(query_return=[])

    try:
        await _create_test_nodes(
            neo4j_async_driver,
            [(eid, proj_a) for eid in src_a]
            + [(eid, proj_b) for eid in src_b],
            run_id,
        )

        report = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            dry_run=False,
            rate_limit_qps=0,
            project_filter=proj_a,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )

        assert report.memories_processed == 5
        assert proj_a in report.by_project
        assert proj_b not in report.by_project
        assert report.by_project[proj_a]["processed"] == 5
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_08_missing_embedding_in_pinecone(
    neo4j_async_driver: AsyncDriver,
) -> None:
    """5 memories, fetch returns None for 2 of them ⇒ those 2 are tracked
    as skipped_no_embedding and not linked."""
    run_id = _mk_test_run_id()
    proj = "sprint7-test"
    src_ids = [f"s7-08-src-{i}-{run_id[-8:]}" for i in range(5)]
    # The first 2 are "missing" in Pinecone.
    missing = set(src_ids[:2])

    async def _fetch_missing(
        _pinecone: Any, memory_id: str
    ) -> Optional[List[float]]:
        if memory_id in missing:
            return None
        return [0.01] * 1536

    pinecone_mock = _build_pinecone_mock(query_return=[])

    try:
        await _create_test_nodes(
            neo4j_async_driver, [(eid, proj) for eid in src_ids], run_id
        )

        report = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            dry_run=False,
            rate_limit_qps=0,
            project_filter=proj,
            fetch_embedding_override=_fetch_missing,
            _allow_test_run_id=True,
        )

        assert report.memories_skipped_no_embedding == 2
        assert report.memories_processed == 3
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_12_idempotency(
    neo4j_async_driver: AsyncDriver,
) -> None:
    """Running the same backfill twice creates the same edges on first run
    and does not duplicate on second run (MERGE upsert semantics).

    Candidates live in a DIFFERENT project so the backfill iterates only
    source memories — same pattern as test_02.
    """
    run_id = _mk_test_run_id()
    proj_src = "sprint7-test-src"
    proj_cand = "sprint7-test-cand"
    src_ids = [f"s7-12-src-{i}-{run_id[-8:]}" for i in range(3)]
    cand_ids = [f"s7-12-cand-{i}-{run_id[-8:]}" for i in range(3)]

    def _candidates(**_kwargs: Any) -> List[Dict[str, Any]]:
        return _make_candidates(
            [(cid, 0.95 - i * 0.02) for i, cid in enumerate(cand_ids)]
        )

    pinecone_mock = _build_pinecone_mock(query_side_effect=_candidates)

    try:
        await _create_test_nodes(
            neo4j_async_driver,
            [(eid, proj_src) for eid in src_ids]
            + [(cid, proj_cand) for cid in cand_ids],
            run_id,
        )

        # First run.
        r1 = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            dry_run=False,
            rate_limit_qps=0,
            project_filter=proj_src,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )
        first_count = await _count_similar_edges_by_run_id(
            neo4j_async_driver, f"backfill-{run_id}"
        )
        assert r1.memories_processed == 3
        assert first_count > 0

        # Second run over exactly the same state.
        r2 = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            dry_run=False,
            rate_limit_qps=0,
            project_filter=proj_src,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )
        second_count = await _count_similar_edges_by_run_id(
            neo4j_async_driver, f"backfill-{run_id}"
        )
        # Exact same edge count — MERGE is a no-op on re-run.
        assert second_count == first_count, (
            f"Idempotency failed: first run {first_count} edges, "
            f"second run {second_count} edges"
        )
    finally:
        assoc_rollback(
            run_id=f"backfill-{run_id}",
            dry_run=False,
            driver=None,
            uri=TEST_NEO4J_URI,
            database=TEST_DB,
        )
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_13_rollback_integration(
    neo4j_async_driver: AsyncDriver,
) -> None:
    """After a live backfill creates edges, assoc_rollback removes all of
    them and leaves the underlying nodes intact."""
    run_id = _mk_test_run_id()
    proj_src = "sprint7-test-src"
    proj_cand = "sprint7-test-cand"
    src_ids = [f"s7-13-src-{i}-{run_id[-8:]}" for i in range(5)]
    cand_ids = [f"s7-13-cand-{i}-{run_id[-8:]}" for i in range(3)]

    def _candidates(**_kwargs: Any) -> List[Dict[str, Any]]:
        return _make_candidates(
            [(cand_ids[0], 0.95), (cand_ids[1], 0.89), (cand_ids[2], 0.84)]
        )

    pinecone_mock = _build_pinecone_mock(query_side_effect=_candidates)

    try:
        await _create_test_nodes(
            neo4j_async_driver,
            [(eid, proj_src) for eid in src_ids]
            + [(cid, proj_cand) for cid in cand_ids],
            run_id,
        )

        backfill_report = await backfill_similarity_edges(
            run_id=run_id,
            driver=neo4j_async_driver,
            pinecone_client=pinecone_mock,
            database=TEST_DB,
            batch_size=50,
            dry_run=False,
            rate_limit_qps=0,
            project_filter=proj_src,
            fetch_embedding_override=_noop_fetch_embedding,
            _allow_test_run_id=True,
        )
        assert backfill_report.edges_created > 0

        tagged_run_id = f"backfill-{run_id}"
        pre_rollback_edges = await _count_similar_edges_by_run_id(
            neo4j_async_driver, tagged_run_id
        )
        assert pre_rollback_edges == backfill_report.edges_created

        # Call Sprint 2's rollback CLI as a library. It uses the sync
        # driver API; we pass uri so it opens its own driver.
        rollback_report = assoc_rollback(
            run_id=tagged_run_id,
            dry_run=False,
            driver=None,
            uri=TEST_NEO4J_URI,
            database=TEST_DB,
        )
        assert rollback_report.total == pre_rollback_edges

        # All edges gone.
        post_rollback_edges = await _count_similar_edges_by_run_id(
            neo4j_async_driver, tagged_run_id
        )
        assert post_rollback_edges == 0

        # Nodes still present (rollback is edge-level, not node-level).
        remaining_nodes = await _count_test_label_nodes(neo4j_async_driver)
        assert remaining_nodes == len(src_ids) + len(cand_ids)
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


# --------------------------------------------------------------------------
# Final production-invariant check
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_zzz_production_counts_unchanged(
    neo4j_async_driver: AsyncDriver,
    production_counts: Dict[str, int],
) -> None:
    """Final invariant: no test leakage into production labels/edges.

    ``:base`` may drift upward from organic writes by the live Fusion Memory
    service. ``:Session``, ``FOLLOWS``, ``INCLUDES`` must be byte-for-byte
    unchanged — Sprint 7 never writes to any of those. No leftover
    ``:BackfillTestNode``, no surviving ``sprint7-backfill-test-*``
    SIMILAR_TO edges.
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
        leftover_edges = (
            await (
                await session.run(
                    "MATCH ()-[r:SIMILAR_TO]->() "
                    "WHERE r.run_id STARTS WITH 'backfill-sprint7-backfill-test-' "
                    "RETURN count(r) AS c"
                )
            ).single()
        )["c"]

    assert test_label_after == 0, (
        f"Teardown leaked: {test_label_after} :{TEST_LABEL} nodes survived "
        "the full test run. BLOCKER — re-run teardown manually."
    )
    assert leftover_edges == 0, (
        f"Test leaked {leftover_edges} SIMILAR_TO edges tagged with "
        "backfill-sprint7-backfill-test-* — the idempotency or rollback "
        "tests failed to clean up."
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
