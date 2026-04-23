"""Phase 8a live-test for ``get_provenance`` against real nova-core edges.

Sprint 17 deliverable: exercises the ``get_provenance`` MCP handler against
three real provenance edges seeded between four real ``:base`` nodes in the
live Neo4j graph. This complements the hermetic / synthetic-node coverage in
``test_provenance_api.py`` and proves the read path works end-to-end on
production-shaped data.

Safety model:
    - All three seeded edges are tagged with ``run_id = wt-phase8-livetest-2026-04-23``.
    - Module teardown invokes ``scripts.assoc_rollback`` to delete every edge
      with that ``run_id``. This is the same rollback contract used in
      production — by run_id only, never by edge type.
    - The four ``:base`` nodes themselves are untouched (they are pre-existing
      real memories); only the 3 synthetic edges are added and removed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator, Optional

import pytest
import pytest_asyncio

# Make the app package importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
    from neo4j import exceptions as neo4j_exceptions
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    pytest.skip(
        "neo4j driver not installed; cannot run provenance live-test",
        allow_module_level=True,
    )

from unittest.mock import MagicMock

from app.services.associations.edge_service import MemoryEdgeService
from mcp_server import get_provenance as get_provenance_tool
from scripts.assoc_rollback import assoc_rollback
from scripts.seed_phase8_livetest_provenance import (
    DEFAULT_RUN_ID,
    ENTITY_A,
    ENTITY_B,
    ENTITY_C,
    ENTITY_D,
    _build_edges,
)

TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_DB = "neo4j"
RUN_ID = DEFAULT_RUN_ID


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
# Fixtures — module-scoped seed + rollback on teardown
# --------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def neo4j_async_driver() -> AsyncIterator[AsyncDriver]:
    driver = await _try_open_async_driver()
    if driver is None:
        pytest.skip(
            f"Neo4j not reachable at {TEST_NEO4J_URI}; skipping live-test."
        )
    try:
        yield driver
    finally:
        await driver.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def seeded_edges(
    neo4j_async_driver: AsyncDriver,
) -> AsyncIterator[MemoryEdgeService]:
    """Seed 3 provenance edges; roll back by run_id in teardown."""
    service = MemoryEdgeService(driver=neo4j_async_driver, database=TEST_DB)

    # Defensive pre-clean: if a previous failed run left edges behind, drop
    # them before seeding so we do not MERGE onto stale properties.
    sync_driver = GraphDatabase.driver(TEST_NEO4J_URI, auth=None)
    try:
        assoc_rollback(
            run_id=RUN_ID,
            dry_run=False,
            driver=sync_driver,
            database=TEST_DB,
        )
    finally:
        sync_driver.close()

    edges = _build_edges(RUN_ID)
    for edge in edges:
        ok = await service.create_edge(edge)
        assert ok, f"failed to create seed edge {edge}"

    try:
        yield service
    finally:
        # Rollback every edge with run_id=<RUN_ID>. Uses the sync driver
        # because assoc_rollback is sync by design.
        sync_driver = GraphDatabase.driver(TEST_NEO4J_URI, auth=None)
        try:
            report = assoc_rollback(
                run_id=RUN_ID,
                dry_run=False,
                driver=sync_driver,
                database=TEST_DB,
            )
            assert report.total >= 3, (
                f"expected >=3 edges rolled back, got {report.total}"
            )
        finally:
            sync_driver.close()


# --------------------------------------------------------------------------
# MCP Context stub (mirrors test_provenance_api.py:410-432)
# --------------------------------------------------------------------------


class _FakeLifespan:
    def __init__(self, memory_service):
        self.memory_service = memory_service


class _FakeRequestCtx:
    def __init__(self, memory_service):
        self.lifespan_context = _FakeLifespan(memory_service)


class _FakeContext:
    def __init__(self, memory_service):
        self.request_context = _FakeRequestCtx(memory_service)


def _make_mcp_ctx(live_edge_service: MemoryEdgeService) -> _FakeContext:
    ms = MagicMock()
    ms._initialized = True
    ms._mcp_edge_service = live_edge_service
    return _FakeContext(ms)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_live_provenance_mixed_chain_full_depth(
    seeded_edges: MemoryEdgeService,
) -> None:
    """From A with max_depth=5: expect {B, C, D} in the chain.

    Topology::
        A --[SUPERSEDES]--> B
        A --[PROMOTED_FROM]--> C
        C --[COMPACTED_FROM]--> D
    """
    ctx = _make_mcp_ctx(seeded_edges)
    result = await get_provenance_tool(ctx, memory_id=ENTITY_A, max_depth=5)

    assert "error" not in result
    assert result["memory_id"] == ENTITY_A
    assert result["exists"] is True
    assert result["exists_checked"] is True
    assert result["depth_limited"] is False

    chain = result["provenance_chain"]
    chain_ids = {hop["memory_id"] for hop in chain}
    assert chain_ids == {ENTITY_B, ENTITY_C, ENTITY_D}, (
        f"unexpected chain ids: {chain_ids}"
    )
    assert len(chain) == 3

    # Verify per-hop depth and edge_type.
    by_id = {hop["memory_id"]: hop for hop in chain}
    assert by_id[ENTITY_B]["edge_type"] == "SUPERSEDES"
    assert by_id[ENTITY_B]["depth"] == 1
    assert by_id[ENTITY_C]["edge_type"] == "PROMOTED_FROM"
    assert by_id[ENTITY_C]["depth"] == 1
    assert by_id[ENTITY_D]["edge_type"] == "COMPACTED_FROM"
    assert by_id[ENTITY_D]["depth"] == 2

    # Leaves: B has no outgoing provenance edge, D has no outgoing edge.
    assert set(result["original_sources"]) == {ENTITY_B, ENTITY_D}

    # Overall depth from the service = max hop depth.
    assert result["depth"] == 2

    # Mixed edge types — all three provenance types are represented.
    edge_types = {hop["edge_type"] for hop in chain}
    assert edge_types == {"SUPERSEDES", "PROMOTED_FROM", "COMPACTED_FROM"}


@pytest.mark.asyncio(loop_scope="module")
async def test_live_provenance_depth_limited(
    seeded_edges: MemoryEdgeService,
) -> None:
    """max_depth=1 cuts the chain to B and C; D (depth=2) is excluded."""
    ctx = _make_mcp_ctx(seeded_edges)
    result = await get_provenance_tool(ctx, memory_id=ENTITY_A, max_depth=1)

    assert "error" not in result
    assert result["exists"] is True

    chain_ids = {hop["memory_id"] for hop in result["provenance_chain"]}
    assert chain_ids == {ENTITY_B, ENTITY_C}
    assert ENTITY_D not in chain_ids
    assert result["depth_limited"] is True
    assert result["max_depth"] == 1
