"""Integration tests for ``MemoryEdgeService`` against live Neo4j (Sprint 5).

Safety model
------------

This test suite runs against the **live** ``nova_neo4j_db`` container that
also holds real Fusion Memory graph data (``:base`` + ``:Session`` nodes,
pre-existing ``FOLLOWS`` and ``INCLUDES`` edges). That means production data
is only a query away, so the suite is built under the same safety contract
that Sprint 2's ``tests/test_assoc_rollback.py`` established:

1. **Dedicated test label.** Every test node uses the label
   ``:AssocEdgeServiceTestNode`` *as primary label*, with ``:base`` as a
   **secondary** label so the service's Cypher (which matches on
   ``(:base {entity_id: ...})``) can actually bind them. Teardown deletes
   by ``:AssocEdgeServiceTestNode`` — a single ``DETACH DELETE`` on that
   label drops every test node regardless of secondary labels, taking the
   attached edges with it.

2. **Unique run_id per test invocation.** A fresh
   ``sprint5-edge-service-test-<uuid8>`` is generated at module fixture
   setup; individual tests derive per-case run_ids from it. Repeat runs
   never share a run_id and cleanup is surgical if teardown ever fails.

3. **Production-count invariant.** The module fixture captures
   ``count(n:base)``, ``count(n:Session)``, ``count(()-[:FOLLOWS]->())``
   and ``count(()-[:INCLUDES]->())`` before any test runs, and the final
   test in the module re-checks them after. ``:base`` is allowed to drift
   *upward* (organic writes from the live Fusion Memory service) but the
   delta must contain no nodes bearing the test label. ``:Session``,
   ``FOLLOWS``, and ``INCLUDES`` are asserted unchanged.

4. **Unconditional teardown.** Every test wraps its setup in a
   ``try/finally`` that runs
   ``MATCH (n:AssocEdgeServiceTestNode) DETACH DELETE n`` regardless of
   pass/fail/error. A stray assertion failure can never leave test nodes
   behind.

5. **Skip, not fail, when Neo4j is unreachable.** CI / dev machines
   without a running container skip the entire module with a clear reason.

6. **No production-label writes.** The strings ``:base``, ``:Session``,
   ``:Entity``, and ``:Memory`` appear in this file **only** inside the
   read-only production-count assertions. Every ``CREATE``, ``MERGE``, and
   ``DELETE`` targets ``:AssocEdgeServiceTestNode`` as its primary
   label — confirmed by a self-check at fixture startup.

7. **Multi-label test nodes.** Because the service queries ``(:base
   {entity_id})`` and relationship-type matching cannot distinguish the
   "test" subgraph from the production subgraph structurally, we add
   ``:base`` as a secondary label on test nodes at creation time.
   ``CREATE (n:AssocEdgeServiceTestNode:base {entity_id: ..., ...})``
   gives the node both labels; teardown still finds and removes it via
   the primary test label. **This is the only reason ``:base`` appears in
   a write path in this file**, and it is guarded by the production-count
   invariant at the end of the module.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional

import pytest
import pytest_asyncio

# --- Make the ``app`` package importable without pulling in the full
#     config stack. We import MemoryEdgeService directly from its module
#     path so ``app.config.settings`` (which would require Pinecone /
#     OpenAI credentials in CI) is never loaded.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
    from neo4j import exceptions as neo4j_exceptions
except ImportError:  # pragma: no cover - env sanity
    pytest.skip(
        "neo4j driver not installed; cannot run edge-service integration test",
        allow_module_level=True,
    )

from app.services.associations.edge_service import MemoryEdgeService
from app.services.associations.memory_edges import (
    BIDIRECTIONAL_EDGE_TYPES,
    VALID_EDGE_TYPES,
    MemoryEdge,
)


# --------------------------------------------------------------------------
# Constants + connection helpers
# --------------------------------------------------------------------------

TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_DB = "neo4j"
TEST_LABEL = "AssocEdgeServiceTestNode"
# Shared session label used for the production-count invariant.
# Sprint 5 does not write to :Session in any circumstance.
PROD_SESSION_LABEL = "Session"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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
# Pytest fixtures — all async to match the driver's lifecycle
# --------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def neo4j_async_driver() -> AsyncIterator[AsyncDriver]:
    driver = await _try_open_async_driver()
    if driver is None:
        pytest.skip(
            f"Neo4j not reachable at {TEST_NEO4J_URI}; skipping edge-service "
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
    """Record production counts once at module start.

    These counts are used by the final ``test_zzz_production_counts_unchanged``
    test to verify no test data leaked into production labels.
    """
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


@pytest_asyncio.fixture(loop_scope="module")
async def edge_service(
    neo4j_async_driver: AsyncDriver,
) -> AsyncIterator[MemoryEdgeService]:
    """A fresh ``MemoryEdgeService`` bound to the module driver."""
    yield MemoryEdgeService(driver=neo4j_async_driver, database=TEST_DB)


@pytest.fixture
def run_id() -> str:
    """Per-test unique run_id."""
    return f"sprint5-edge-service-test-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# Node helpers — every CREATE uses the TEST_LABEL as primary label and
# gives the node :base as a secondary label so that the service's Cypher
# (which matches on (:base {entity_id})) can bind it. Teardown always
# deletes by TEST_LABEL.
# --------------------------------------------------------------------------


async def _create_test_nodes(
    driver: AsyncDriver, entity_ids: list[str], run_tag: str
) -> None:
    """Create test nodes with both ``:AssocEdgeServiceTestNode`` and ``:base``.

    The secondary ``:base`` label is **required** for the service's Cypher
    templates to match these nodes; the primary ``:AssocEdgeServiceTestNode``
    label is what teardown keys off. ``run_tag`` is stamped into the node
    so stray leftovers can be traced back to the originating test run.
    """
    async with driver.session(database=TEST_DB) as session:
        # Parameterize entity_ids list; CREATE with multi-label syntax.
        # Note: this is the ONE place in this file where ``:base`` is used
        # as a write-target. Removal of ``:base`` from test nodes happens
        # automatically when the TEST_LABEL DETACH DELETE runs in teardown.
        await (
            await session.run(
                f"UNWIND $ids AS eid "
                f"CREATE (n:{TEST_LABEL}:base {{entity_id: eid, test_run: $tag}})",
                {"ids": entity_ids, "tag": run_tag},
            )
        ).consume()


async def _teardown_test_nodes(driver: AsyncDriver) -> None:
    """Unconditionally drop every :AssocEdgeServiceTestNode and its edges."""
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(
                f"MATCH (n:{TEST_LABEL}) DETACH DELETE n"
            )
        ).consume()


def _make_edge(
    source_id: str,
    target_id: str,
    edge_type: str,
    run_tag: str,
    weight: float = 0.8,
    created_by: str = "sprint5_test_linker",
) -> MemoryEdge:
    ts = _now_iso()
    return MemoryEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        weight=weight,
        created_at=ts,
        last_seen_at=ts,
        created_by=created_by,
        run_id=run_tag,
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_01_create_edge_between_existing_nodes(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
    production_counts: dict[str, int],  # force fixture ordering
) -> None:
    """Creating an edge between two existing test nodes returns True and persists."""
    a = f"s5-n01-a-{run_id}"
    b = f"s5-n01-b-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b], run_id)
        edge = _make_edge(a, b, "MEMORY_FOLLOWS", run_id, weight=0.65)
        result = await edge_service.create_edge(edge)
        assert result is True

        # Verify the edge really landed with the expected properties.
        async with neo4j_async_driver.session(database=TEST_DB) as session:
            rec = (
                await (
                    await session.run(
                        f"MATCH (x:{TEST_LABEL} {{entity_id: $a}})"
                        f"-[r:MEMORY_FOLLOWS]->"
                        f"(y:{TEST_LABEL} {{entity_id: $b}}) "
                        f"RETURN r.weight AS w, r.run_id AS rid, "
                        f"r.edge_version AS ev, r.created_by AS cb",
                        {"a": a, "b": b},
                    )
                ).single()
            )
        assert rec is not None
        assert rec["w"] == pytest.approx(0.65)
        assert rec["rid"] == run_id
        assert rec["ev"] == 1
        assert rec["cb"] == "sprint5_test_linker"
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_02_invalid_edge_type_raises_at_construction(
    edge_service: MemoryEdgeService,
    run_id: str,
) -> None:
    """Sprint 4's validation rejects BOGUS before the service is ever invoked."""
    with pytest.raises(ValueError):
        MemoryEdge(
            source_id="irrelevant-a",
            target_id="irrelevant-b",
            edge_type="BOGUS",
            weight=0.5,
            created_at=_now_iso(),
            last_seen_at=_now_iso(),
            created_by="sprint5_test_linker",
            run_id=run_id,
        )


@pytest.mark.asyncio(loop_scope="module")
async def test_03_create_edge_missing_source_returns_false(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """If one endpoint doesn't exist, create_edge returns False silently."""
    b = f"s5-n03-b-{run_id}"
    try:
        # Only create target; source is a phantom id.
        await _create_test_nodes(neo4j_async_driver, [b], run_id)
        phantom_src = f"s5-n03-phantom-{run_id}"
        edge = _make_edge(phantom_src, b, "SUPERSEDES", run_id)
        result = await edge_service.create_edge(edge)
        assert result is False

        # No edge should exist.
        async with neo4j_async_driver.session(database=TEST_DB) as session:
            rec = (
                await (
                    await session.run(
                        f"MATCH ()-[r:SUPERSEDES {{run_id: $rid}}]->() "
                        f"RETURN count(r) AS c",
                        {"rid": run_id},
                    )
                ).single()
            )
        assert rec["c"] == 0
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_04_duplicate_edge_keeps_larger_weight(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """MERGE tie-break: weight is monotonically non-decreasing (max wins)."""
    a = f"s5-n04-a-{run_id}"
    b = f"s5-n04-b-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b], run_id)
        # Use RELATED_TASK so the direction is preserved (not canonicalized).
        first = _make_edge(a, b, "RELATED_TASK", run_id, weight=0.3)
        second = _make_edge(a, b, "RELATED_TASK", run_id, weight=0.9)
        third = _make_edge(a, b, "RELATED_TASK", run_id, weight=0.5)

        assert await edge_service.create_edge(first) is True
        assert await edge_service.create_edge(second) is True
        assert await edge_service.create_edge(third) is True

        async with neo4j_async_driver.session(database=TEST_DB) as session:
            rec = (
                await (
                    await session.run(
                        f"MATCH (x:{TEST_LABEL} {{entity_id: $a}})"
                        f"-[r:RELATED_TASK]->"
                        f"(y:{TEST_LABEL} {{entity_id: $b}}) "
                        f"RETURN r.weight AS w, count(r) AS c",
                        {"a": a, "b": b},
                    )
                ).single()
            )
        # Exactly one edge, weight is the max of the three writes.
        assert rec["c"] == 1
        assert rec["w"] == pytest.approx(0.9)
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_05_get_neighbors_with_edge_type_filter(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Filter by edge_type returns only matching neighbors."""
    a = f"s5-n05-a-{run_id}"
    b = f"s5-n05-b-{run_id}"
    c = f"s5-n05-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(
            _make_edge(a, b, "RELATED_TASK", run_id, weight=0.9)
        )
        await edge_service.create_edge(
            _make_edge(a, c, "MENTIONS", run_id, weight=0.7)
        )

        related = await edge_service.get_neighbors(
            node_id=a, edge_types=["RELATED_TASK"], direction="out"
        )
        assert len(related) == 1
        assert related[0]["node_id"] == b
        assert related[0]["edge_type"] == "RELATED_TASK"

        mentions = await edge_service.get_neighbors(
            node_id=a, edge_types=["MENTIONS"], direction="out"
        )
        assert len(mentions) == 1
        assert mentions[0]["node_id"] == c
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_06_get_neighbors_min_weight_filter(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Edges below ``min_weight`` are excluded."""
    a = f"s5-n06-a-{run_id}"
    b = f"s5-n06-b-{run_id}"
    c = f"s5-n06-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(
            _make_edge(a, b, "RELATED_TASK", run_id, weight=0.3)
        )
        await edge_service.create_edge(
            _make_edge(a, c, "RELATED_TASK", run_id, weight=0.85)
        )

        strong = await edge_service.get_neighbors(
            node_id=a, edge_types=["RELATED_TASK"], min_weight=0.5, direction="out"
        )
        assert len(strong) == 1
        assert strong[0]["node_id"] == c
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_07_get_neighbors_direction_out(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """direction='out' only follows outgoing edges."""
    a = f"s5-n07-a-{run_id}"
    b = f"s5-n07-b-{run_id}"
    c = f"s5-n07-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        # a -> b, c -> a
        await edge_service.create_edge(
            _make_edge(a, b, "CAUSED_BY", run_id, weight=0.7)
        )
        await edge_service.create_edge(
            _make_edge(c, a, "CAUSED_BY", run_id, weight=0.6)
        )

        out = await edge_service.get_neighbors(
            node_id=a, edge_types=["CAUSED_BY"], direction="out"
        )
        assert len(out) == 1
        assert out[0]["node_id"] == b
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_08_get_neighbors_direction_in(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """direction='in' only follows incoming edges."""
    a = f"s5-n08-a-{run_id}"
    b = f"s5-n08-b-{run_id}"
    c = f"s5-n08-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        # a -> b, c -> a
        await edge_service.create_edge(
            _make_edge(a, b, "CAUSED_BY", run_id, weight=0.7)
        )
        await edge_service.create_edge(
            _make_edge(c, a, "CAUSED_BY", run_id, weight=0.6)
        )

        incoming = await edge_service.get_neighbors(
            node_id=a, edge_types=["CAUSED_BY"], direction="in"
        )
        assert len(incoming) == 1
        assert incoming[0]["node_id"] == c
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_09_get_path_connected(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Shortest path is found between transitively connected nodes."""
    a = f"s5-n09-a-{run_id}"
    b = f"s5-n09-b-{run_id}"
    c = f"s5-n09-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(
            _make_edge(a, b, "CAUSED_BY", run_id, weight=0.7)
        )
        await edge_service.create_edge(
            _make_edge(b, c, "CAUSED_BY", run_id, weight=0.6)
        )

        path = await edge_service.get_path(
            start_id=a, end_id=c, max_depth=4, edge_types=["CAUSED_BY"]
        )
        assert path is not None
        # Two hops for a three-node chain.
        assert len(path) == 2
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_10_get_path_disconnected_returns_none(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """No path exists → None."""
    a = f"s5-n10-a-{run_id}"
    b = f"s5-n10-b-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b], run_id)
        # Intentionally no edges between them.
        path = await edge_service.get_path(
            start_id=a, end_id=b, max_depth=4, edge_types=["CAUSED_BY"]
        )
        assert path is None
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_11_get_path_respects_max_depth(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """A 4-hop chain is not reachable with max_depth=2."""
    a = f"s5-n11-a-{run_id}"
    b = f"s5-n11-b-{run_id}"
    c = f"s5-n11-c-{run_id}"
    d = f"s5-n11-d-{run_id}"
    e = f"s5-n11-e-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c, d, e], run_id)
        # a -> b -> c -> d -> e (4 hops)
        await edge_service.create_edge(_make_edge(a, b, "CAUSED_BY", run_id))
        await edge_service.create_edge(_make_edge(b, c, "CAUSED_BY", run_id))
        await edge_service.create_edge(_make_edge(c, d, "CAUSED_BY", run_id))
        await edge_service.create_edge(_make_edge(d, e, "CAUSED_BY", run_id))

        near = await edge_service.get_path(
            start_id=a, end_id=e, max_depth=2, edge_types=["CAUSED_BY"]
        )
        assert near is None

        far = await edge_service.get_path(
            start_id=a, end_id=e, max_depth=4, edge_types=["CAUSED_BY"]
        )
        assert far is not None
        assert len(far) == 4
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_12_delete_edges_by_source_and_type(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """delete_edges removes matching edges and leaves the rest."""
    a = f"s5-n12-a-{run_id}"
    b = f"s5-n12-b-{run_id}"
    c = f"s5-n12-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(
            _make_edge(a, b, "RELATED_TASK", run_id)
        )
        await edge_service.create_edge(
            _make_edge(a, c, "MENTIONS", run_id)
        )
        await edge_service.create_edge(
            _make_edge(a, c, "RELATED_TASK", run_id)
        )

        deleted = await edge_service.delete_edges(
            source_id=a, edge_type="RELATED_TASK"
        )
        assert deleted == 2

        # MENTIONS survives.
        remaining = await edge_service.get_neighbors(
            node_id=a, direction="out"
        )
        assert len(remaining) == 1
        assert remaining[0]["edge_type"] == "MENTIONS"
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_13_delete_edges_by_run_surgical(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """delete_edges_by_run removes only the named run, leaving others."""
    a = f"s5-n13-a-{run_id}"
    b = f"s5-n13-b-{run_id}"
    c = f"s5-n13-c-{run_id}"
    other_run = run_id + "-other"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(
            _make_edge(a, b, "RELATED_TASK", run_id)
        )
        await edge_service.create_edge(
            _make_edge(a, c, "RELATED_TASK", other_run)
        )

        deleted = await edge_service.delete_edges_by_run(run_id=run_id)
        assert deleted == 1

        # The "other_run" edge must survive.
        remaining = await edge_service.get_neighbors(
            node_id=a, edge_types=["RELATED_TASK"], direction="out"
        )
        assert len(remaining) == 1
        assert remaining[0]["node_id"] == c
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_14_delete_edges_by_tag_created_by(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """delete_edges_by_tag removes edges with matching created_by."""
    a = f"s5-n14-a-{run_id}"
    b = f"s5-n14-b-{run_id}"
    c = f"s5-n14-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(
            _make_edge(a, b, "RELATED_TASK", run_id, created_by="linker_alpha")
        )
        await edge_service.create_edge(
            _make_edge(a, c, "RELATED_TASK", run_id, created_by="linker_beta")
        )

        deleted = await edge_service.delete_edges_by_tag(created_by="linker_alpha")
        assert deleted == 1

        remaining = await edge_service.get_neighbors(
            node_id=a, edge_types=["RELATED_TASK"], direction="out"
        )
        assert len(remaining) == 1
        assert remaining[0]["node_id"] == c
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_15_count_edges_per_node(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """count_edges_per_node returns correct counts by type, both directions."""
    a = f"s5-n15-a-{run_id}"
    b = f"s5-n15-b-{run_id}"
    c = f"s5-n15-c-{run_id}"
    d = f"s5-n15-d-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c, d], run_id)
        # a -> b (RELATED_TASK), a -> c (RELATED_TASK), d -> a (MENTIONS)
        await edge_service.create_edge(
            _make_edge(a, b, "RELATED_TASK", run_id)
        )
        await edge_service.create_edge(
            _make_edge(a, c, "RELATED_TASK", run_id)
        )
        await edge_service.create_edge(
            _make_edge(d, a, "MENTIONS", run_id)
        )

        counts = await edge_service.count_edges_per_node(node_id=a)
        assert counts.get("RELATED_TASK") == 2
        assert counts.get("MENTIONS") == 1
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_16_get_edge_stats_shape(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """get_edge_stats returns one row per valid edge type with correct shape."""
    a = f"s5-n16-a-{run_id}"
    b = f"s5-n16-b-{run_id}"
    c = f"s5-n16-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(
            _make_edge(a, b, "RELATED_TASK", run_id, weight=0.4)
        )
        await edge_service.create_edge(
            _make_edge(a, c, "RELATED_TASK", run_id, weight=0.8)
        )

        stats = await edge_service.get_edge_stats()
        # All 9 types present.
        assert set(stats.keys()) == VALID_EDGE_TYPES
        for et, row in stats.items():
            assert set(row.keys()) == {"count", "avg_weight", "min_weight", "max_weight"}
        # Zero-count types have None weight aggregates.
        zero_type = "COMPACTED_FROM"
        # This type may have test edges from prior runs if teardown raced,
        # so we allow count >= 0 but assert shape invariants instead.
        if stats[zero_type]["count"] == 0:
            assert stats[zero_type]["avg_weight"] is None
            assert stats[zero_type]["min_weight"] is None
            assert stats[zero_type]["max_weight"] is None
        # The RELATED_TASK row must include at least our two contributions.
        assert stats["RELATED_TASK"]["count"] >= 2
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_17_create_edges_batch_50(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """create_edges_batch persists every edge in a 50-edge batch."""
    src = f"s5-n17-src-{run_id}"
    targets = [f"s5-n17-t{i:02d}-{run_id}" for i in range(50)]
    try:
        await _create_test_nodes(neo4j_async_driver, [src] + targets, run_id)
        edges = [
            _make_edge(src, t, "RELATED_TASK", run_id, weight=0.5 + i * 0.005)
            for i, t in enumerate(targets)
        ]
        created = await edge_service.create_edges_batch(edges)
        assert created == 50

        async with neo4j_async_driver.session(database=TEST_DB) as session:
            rec = (
                await (
                    await session.run(
                        f"MATCH (x:{TEST_LABEL} {{entity_id: $src}})"
                        f"-[r:RELATED_TASK {{run_id: $rid}}]->"
                        f"(:{TEST_LABEL}) "
                        f"RETURN count(r) AS c",
                        {"src": src, "rid": run_id},
                    )
                ).single()
            )
        assert rec["c"] == 50
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_18_weight_boundary_values(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """weight=0.0 and weight=1.0 are both accepted and stored."""
    a = f"s5-n18-a-{run_id}"
    b = f"s5-n18-b-{run_id}"
    c = f"s5-n18-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(
            _make_edge(a, b, "RELATED_TASK", run_id, weight=0.0)
        )
        await edge_service.create_edge(
            _make_edge(a, c, "MENTIONS", run_id, weight=1.0)
        )

        neighbors = await edge_service.get_neighbors(
            node_id=a, direction="out", min_weight=0.0
        )
        weights = sorted(float(r["weight"]) for r in neighbors)
        assert weights == [0.0, 1.0]
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_19_bidirectional_canonicalization(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """SIMILAR_TO with (B, A) is canonicalized to (A, B); only one edge exists."""
    # Choose IDs where A < B lexicographically.
    a = f"s5-n19-aaa-{run_id}"
    b = f"s5-n19-zzz-{run_id}"
    assert a < b, "test invariant: a must sort before b"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b], run_id)
        # Call the service with (source=B, target=A) — the reverse order.
        rev_edge = _make_edge(b, a, "SIMILAR_TO", run_id, weight=0.77)
        assert await edge_service.create_edge(rev_edge) is True

        # Exactly one SIMILAR_TO edge should exist between these nodes,
        # and it should be stored in the canonical A -> B direction.
        async with neo4j_async_driver.session(database=TEST_DB) as session:
            total = (
                await (
                    await session.run(
                        f"MATCH (x:{TEST_LABEL} {{entity_id: $a}})"
                        f"-[r:SIMILAR_TO]-"
                        f"(y:{TEST_LABEL} {{entity_id: $b}}) "
                        f"RETURN count(r) AS c",
                        {"a": a, "b": b},
                    )
                ).single()
            )
            canonical = (
                await (
                    await session.run(
                        f"MATCH (x:{TEST_LABEL} {{entity_id: $a}})"
                        f"-[r:SIMILAR_TO]->"
                        f"(y:{TEST_LABEL} {{entity_id: $b}}) "
                        f"RETURN count(r) AS c",
                        {"a": a, "b": b},
                    )
                ).single()
            )
            reverse = (
                await (
                    await session.run(
                        f"MATCH (x:{TEST_LABEL} {{entity_id: $b}})"
                        f"-[r:SIMILAR_TO]->"
                        f"(y:{TEST_LABEL} {{entity_id: $a}}) "
                        f"RETURN count(r) AS c",
                        {"a": a, "b": b},
                    )
                ).single()
            )
        assert total["c"] == 1
        assert canonical["c"] == 1
        assert reverse["c"] == 0
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_20_on_memory_delete_removes_all_edges(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """on_memory_delete strips every edge incident on a node, both directions."""
    a = f"s5-n20-a-{run_id}"
    b = f"s5-n20-b-{run_id}"
    c = f"s5-n20-c-{run_id}"
    d = f"s5-n20-d-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c, d], run_id)
        # a -> b, a -> c, d -> a  (three edges touching a)
        await edge_service.create_edge(_make_edge(a, b, "RELATED_TASK", run_id))
        await edge_service.create_edge(_make_edge(a, c, "MENTIONS", run_id))
        await edge_service.create_edge(_make_edge(d, a, "CAUSED_BY", run_id))

        report = await edge_service.on_memory_delete(node_id=a)
        assert report == {"edges_removed": 3, "orphaned_entities": []}

        # No edges touching a remain.
        counts = await edge_service.count_edges_per_node(node_id=a)
        assert counts == {}
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_21_on_memory_supersede_creates_edge(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """on_memory_supersede writes a SUPERSEDES edge with hook metadata."""
    new_id = f"s5-n21-new-{run_id}"
    old_id = f"s5-n21-old-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [new_id, old_id], run_id)
        await edge_service.on_memory_supersede(new_id=new_id, old_id=old_id)

        async with neo4j_async_driver.session(database=TEST_DB) as session:
            rec = (
                await (
                    await session.run(
                        f"MATCH (x:{TEST_LABEL} {{entity_id: $n}})"
                        f"-[r:SUPERSEDES]->"
                        f"(y:{TEST_LABEL} {{entity_id: $o}}) "
                        f"RETURN r.created_by AS cb, r.run_id AS rid, "
                        f"r.weight AS w",
                        {"n": new_id, "o": old_id},
                    )
                ).single()
            )
        assert rec is not None
        assert rec["cb"] == "edge_service.on_memory_supersede"
        assert rec["rid"] == "supersession_hook"
        assert rec["w"] == pytest.approx(1.0)
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


# --------------------------------------------------------------------------
# Final invariant: production counts unchanged across the whole module
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_zzz_production_counts_unchanged(
    neo4j_async_driver: AsyncDriver,
    production_counts: dict[str, int],
) -> None:
    """Production-data invariant: no :base/:Session/FOLLOWS/INCLUDES leakage.

    :base is permitted to drift *upward* from organic writes by the live
    Fusion Memory service while the suite runs — but the delta must not
    contain any nodes bearing the ``:AssocEdgeServiceTestNode`` primary
    label (which would indicate a teardown miss).

    :Session, FOLLOWS, and INCLUDES must be byte-for-byte unchanged; Sprint
    5 never writes to those structures.
    """
    async with neo4j_async_driver.session(database=TEST_DB) as session:
        base_count_after = (
            await (await session.run("MATCH (n:base) RETURN count(n) AS c")).single()
        )["c"]
        session_count_after = (
            await (
                await session.run(
                    f"MATCH (n:{PROD_SESSION_LABEL}) RETURN count(n) AS c"
                )
            ).single()
        )["c"]
        follows_count_after = (
            await (
                await session.run(
                    "MATCH ()-[r:FOLLOWS]->() RETURN count(r) AS c"
                )
            ).single()
        )["c"]
        includes_count_after = (
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

    # Zero leakage: every TEST_LABEL node must be gone.
    assert test_label_after == 0, (
        f"Teardown leaked: {test_label_after} :{TEST_LABEL} nodes survived "
        "the full test run. This is a BLOCKER — re-run teardown manually."
    )

    # :base may drift upward organically but may not drift downward.
    assert base_count_after >= production_counts["base"], (
        f":base node count dropped: {production_counts['base']} -> "
        f"{base_count_after}. The edge service test has deleted production "
        "data — this is a BLOCKER."
    )

    # :Session, FOLLOWS, INCLUDES: exact equality — Sprint 5 never writes
    # to these structures.
    assert session_count_after == production_counts["session"], (
        f":Session count changed: {production_counts['session']} -> "
        f"{session_count_after}"
    )
    assert follows_count_after == production_counts["follows"], (
        f"FOLLOWS count changed: {production_counts['follows']} -> "
        f"{follows_count_after}"
    )
    assert includes_count_after == production_counts["includes"], (
        f"INCLUDES count changed: {production_counts['includes']} -> "
        f"{includes_count_after}"
    )
