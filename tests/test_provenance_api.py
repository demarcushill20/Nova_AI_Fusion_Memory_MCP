"""Integration tests for ``MemoryEdgeService.get_provenance`` (Phase 5d).

Tests the read-only provenance traversal API that walks SUPERSEDES,
PROMOTED_FROM, and COMPACTED_FROM edges backward to find original
episodic sources.

Safety model: follows the same patterns as ``test_memory_edge_service.py``.
All test nodes use ``ProvenanceTestNode`` as primary label with ``:base``
as secondary label.  Teardown deletes by ``ProvenanceTestNode``.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import pytest
import pytest_asyncio

# --- Make the ``app`` package importable without the full config stack.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
    from neo4j import exceptions as neo4j_exceptions
except ImportError:  # pragma: no cover
    pytest.skip(
        "neo4j driver not installed; cannot run provenance integration test",
        allow_module_level=True,
    )

from unittest.mock import MagicMock

from app.services.associations.edge_service import MemoryEdgeService
from app.services.associations.memory_edges import MemoryEdge

# Import the MCP tool under test. We invoke it directly (as a plain async
# function) rather than via a FastMCP client — same pattern used in
# ``tests/test_mcp_association_tools.py``.
from mcp_server import get_provenance as get_provenance_tool

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_DB = "neo4j"
TEST_LABEL = "ProvenanceTestNode"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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
            f"Neo4j not reachable at {TEST_NEO4J_URI}; skipping provenance tests."
        )
    try:
        yield driver
    finally:
        await driver.close()


@pytest_asyncio.fixture(loop_scope="module")
async def edge_service(
    neo4j_async_driver: AsyncDriver,
) -> AsyncIterator[MemoryEdgeService]:
    yield MemoryEdgeService(driver=neo4j_async_driver, database=TEST_DB)


@pytest.fixture
def run_id() -> str:
    return f"prov-test-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# Node helpers
# --------------------------------------------------------------------------


async def _create_test_nodes(
    driver: AsyncDriver, entity_ids: list[str], run_tag: str
) -> None:
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(
                f"UNWIND $ids AS eid "
                f"CREATE (n:{TEST_LABEL}:base {{entity_id: eid, test_run: $tag}})",
                {"ids": entity_ids, "tag": run_tag},
            )
        ).consume()


async def _teardown_test_nodes(driver: AsyncDriver) -> None:
    async with driver.session(database=TEST_DB) as session:
        await (
            await session.run(f"MATCH (n:{TEST_LABEL}) DETACH DELETE n")
        ).consume()


def _make_edge(
    source_id: str,
    target_id: str,
    edge_type: str,
    run_tag: str,
    weight: float = 1.0,
) -> MemoryEdge:
    ts = _now_iso()
    return MemoryEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        weight=weight,
        created_at=ts,
        last_seen_at=ts,
        created_by="provenance_test",
        run_id=run_tag,
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_single_supersession(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """A supersedes B  =>  provenance of A = [B], original_sources = [B]."""
    a = f"prov-sup-a-{run_id}"
    b = f"prov-sup-b-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b], run_id)
        await edge_service.create_edge(_make_edge(a, b, "SUPERSEDES", run_id))

        result = await edge_service.get_provenance(a)

        assert result["memory_id"] == a
        assert len(result["provenance_chain"]) == 1
        assert result["provenance_chain"][0]["memory_id"] == b
        assert result["provenance_chain"][0]["edge_type"] == "SUPERSEDES"
        assert result["provenance_chain"][0]["depth"] == 1
        assert result["original_sources"] == [b]
        assert result["depth"] == 1
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_chain(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """A supersedes B, B promoted from C => chain = [B, C], sources = [C]."""
    a = f"prov-chain-a-{run_id}"
    b = f"prov-chain-b-{run_id}"
    c = f"prov-chain-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(_make_edge(a, b, "SUPERSEDES", run_id))
        await edge_service.create_edge(
            _make_edge(b, c, "PROMOTED_FROM", run_id)
        )

        result = await edge_service.get_provenance(a)

        assert result["memory_id"] == a
        chain_ids = [hop["memory_id"] for hop in result["provenance_chain"]]
        assert b in chain_ids
        assert c in chain_ids
        assert result["original_sources"] == [c]
        assert result["depth"] == 2
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_compaction_fan_out(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Summary compacted from [S1, S2, S3] => 3 original sources."""
    summary = f"prov-compact-sum-{run_id}"
    s1 = f"prov-compact-s1-{run_id}"
    s2 = f"prov-compact-s2-{run_id}"
    s3 = f"prov-compact-s3-{run_id}"
    try:
        await _create_test_nodes(
            neo4j_async_driver, [summary, s1, s2, s3], run_id
        )
        for src in [s1, s2, s3]:
            await edge_service.create_edge(
                _make_edge(summary, src, "COMPACTED_FROM", run_id)
            )

        result = await edge_service.get_provenance(summary)

        assert result["memory_id"] == summary
        chain_ids = {hop["memory_id"] for hop in result["provenance_chain"]}
        assert chain_ids == {s1, s2, s3}
        assert sorted(result["original_sources"]) == sorted([s1, s2, s3])
        assert result["depth"] == 1
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_no_edges(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Memory with no provenance edges => empty chain."""
    lone = f"prov-lone-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [lone], run_id)

        result = await edge_service.get_provenance(lone)

        assert result["memory_id"] == lone
        assert result["provenance_chain"] == []
        assert result["original_sources"] == []
        assert result["depth"] == 0
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_max_depth(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Chain A->B->C->D->E with max_depth=2 => only B,C discovered."""
    nodes = [f"prov-depth-{i}-{run_id}" for i in range(5)]
    try:
        await _create_test_nodes(neo4j_async_driver, nodes, run_id)
        for i in range(4):
            await edge_service.create_edge(
                _make_edge(nodes[i], nodes[i + 1], "SUPERSEDES", run_id)
            )

        result = await edge_service.get_provenance(nodes[0], max_depth=2)

        chain_ids = {hop["memory_id"] for hop in result["provenance_chain"]}
        # Only nodes[1] and nodes[2] are reachable within 2 hops
        assert nodes[1] in chain_ids
        assert nodes[2] in chain_ids
        assert nodes[3] not in chain_ids
        assert nodes[4] not in chain_ids
        assert result["depth"] == 2
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_cycle_safety(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Circular provenance (A->B->C->A) completes without infinite loop."""
    a = f"prov-cycle-a-{run_id}"
    b = f"prov-cycle-b-{run_id}"
    c = f"prov-cycle-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(_make_edge(a, b, "SUPERSEDES", run_id))
        await edge_service.create_edge(_make_edge(b, c, "SUPERSEDES", run_id))
        await edge_service.create_edge(
            _make_edge(c, a, "SUPERSEDES", run_id)
        )

        # Should complete without hanging or raising
        result = await edge_service.get_provenance(a)

        assert result["memory_id"] == a
        # The chain should contain b and c (Neo4j variable-length paths
        # do not revisit nodes within a single path).
        chain_ids = {hop["memory_id"] for hop in result["provenance_chain"]}
        assert b in chain_ids
        assert c in chain_ids
        # Neo4j prevents relationship revisits, not node revisits, so
        # the full cycle A->B->C->A is traversed.  Every node has an
        # outgoing provenance edge, so no node is a leaf.
        assert result["original_sources"] == []
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_nonexistent_memory(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
) -> None:
    """Non-existent memory_id returns empty result, no error."""
    fake_id = f"prov-ghost-{uuid.uuid4().hex[:8]}"

    result = await edge_service.get_provenance(fake_id)

    assert result["memory_id"] == fake_id
    assert result["provenance_chain"] == []
    assert result["original_sources"] == []
    assert result["depth"] == 0


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_diamond_fan_in(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Diamond: A->B->D, A->C->D => D is the single original source."""
    a = f"prov-diamond-a-{run_id}"
    b = f"prov-diamond-b-{run_id}"
    c = f"prov-diamond-c-{run_id}"
    d = f"prov-diamond-d-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c, d], run_id)
        await edge_service.create_edge(_make_edge(a, b, "SUPERSEDES", run_id))
        await edge_service.create_edge(_make_edge(a, c, "SUPERSEDES", run_id))
        await edge_service.create_edge(_make_edge(b, d, "SUPERSEDES", run_id))
        await edge_service.create_edge(_make_edge(c, d, "SUPERSEDES", run_id))

        result = await edge_service.get_provenance(a)

        chain_ids = {hop["memory_id"] for hop in result["provenance_chain"]}
        assert chain_ids == {b, c, d}
        assert result["original_sources"] == [d]
        assert result["depth"] == 2
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_depth_limited_indicator(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Chain A->B->C->D with max_depth=2 => depth_limited is True."""
    nodes = [f"prov-dlim-{i}-{run_id}" for i in range(4)]
    try:
        await _create_test_nodes(neo4j_async_driver, nodes, run_id)
        for i in range(3):
            await edge_service.create_edge(
                _make_edge(nodes[i], nodes[i + 1], "SUPERSEDES", run_id)
            )

        result = await edge_service.get_provenance(nodes[0], max_depth=2)

        assert result["depth_limited"] is True
        assert result["max_depth"] == 2

        full_result = await edge_service.get_provenance(nodes[0])
        assert full_result["depth_limited"] is False
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


# --------------------------------------------------------------------------
# MCP tool-layer tests
#
# These exercise the MCP ``get_provenance`` tool handler (in ``mcp_server``)
# against the live Neo4j graph, to verify the response-shape polish layered
# on top of the service: ``exists``, ``max_depth`` propagation, response-
# size cap, and silent depth clamping at the tool boundary.
#
# The MCP tool expects a FastMCP ``Context`` providing a ``MemoryService``
# via ``ctx.request_context.lifespan_context.memory_service``.  We build
# a minimal stub (mirroring ``tests/test_mcp_association_tools.py``) and
# pre-seed the cached ``_mcp_edge_service`` with the live ``edge_service``
# fixture so the tool exercises the real Cypher path against Neo4j.
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
    """Build a fake MCP Context backed by the live edge_service fixture."""
    ms = MagicMock()
    ms._initialized = True
    # Pre-seed the cached helper so _get_edge_service() returns the live
    # instance without hitting ms.graph_client.driver.
    ms._mcp_edge_service = live_edge_service
    return _FakeContext(ms)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_mcp_unknown_id_returns_clean_result(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
) -> None:
    """Unknown memory_id => dict with exists=False, empty chain, no error."""
    fake_id = f"prov-mcp-ghost-{uuid.uuid4().hex[:8]}"
    ctx = _make_mcp_ctx(edge_service)

    result = await get_provenance_tool(ctx, memory_id=fake_id)

    assert isinstance(result, dict)
    assert "error" not in result
    assert result["exists"] is False
    # Probe ran cleanly and found no node — distinct from probe failure.
    assert result["exists_checked"] is True
    assert result["provenance_chain"] == []
    assert result["original_sources"] == []
    assert result["truncated"] is False


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_mcp_empty_id_returns_error(
    edge_service: MemoryEdgeService,
) -> None:
    """Empty memory_id => error envelope, not an exception."""
    ctx = _make_mcp_ctx(edge_service)

    result = await get_provenance_tool(ctx, memory_id="")

    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_mcp_max_depth_clamped(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """max_depth=99 => silently clamped to 10; 3-hop chain fully returned."""
    nodes = [f"prov-mcp-clamp-{i}-{run_id}" for i in range(4)]
    try:
        await _create_test_nodes(neo4j_async_driver, nodes, run_id)
        for i in range(3):
            await edge_service.create_edge(
                _make_edge(nodes[i], nodes[i + 1], "SUPERSEDES", run_id)
            )

        ctx = _make_mcp_ctx(edge_service)
        result = await get_provenance_tool(
            ctx, memory_id=nodes[0], max_depth=99
        )

        assert "error" not in result
        # max_depth is propagated from the service, which clamps to 10
        assert result["max_depth"] == 10
        assert result["exists"] is True
        # Full 3-hop chain must be present
        chain_ids = {hop["memory_id"] for hop in result["provenance_chain"]}
        assert chain_ids == {nodes[1], nodes[2], nodes[3]}
        assert result["depth"] == 3
        assert result["depth_limited"] is False
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_mcp_response_size_cap(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """Summary with 35 COMPACTED_FROM sources => chain capped at 30."""
    summary = f"prov-mcp-cap-sum-{run_id}"
    sources = [f"prov-mcp-cap-s{i}-{run_id}" for i in range(35)]
    try:
        await _create_test_nodes(neo4j_async_driver, [summary] + sources, run_id)
        for src in sources:
            await edge_service.create_edge(
                _make_edge(summary, src, "COMPACTED_FROM", run_id)
            )

        ctx = _make_mcp_ctx(edge_service)
        result = await get_provenance_tool(ctx, memory_id=summary)

        assert "error" not in result
        assert result["chain_count"] == 30
        assert result["full_chain_count"] == 35
        assert result["truncated"] is True
        assert result["exists"] is True
        # When truncated, every id in original_sources must appear in the
        # returned provenance_chain — otherwise callers get dangling refs.
        chain_ids = {hop["memory_id"] for hop in result["provenance_chain"]}
        assert all(s in chain_ids for s in result["original_sources"])
    finally:
        await _teardown_test_nodes(neo4j_async_driver)


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_mcp_node_exists_probe_failure(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
) -> None:
    """node_exists raising => exists=False, exists_checked=False (diagnostic)."""
    fake_id = f"prov-mcp-probefail-{uuid.uuid4().hex[:8]}"

    # Monkeypatch this edge_service instance's node_exists to raise.
    original = edge_service.node_exists

    async def _boom(_memory_id: str) -> bool:
        raise RuntimeError("synthetic probe failure")

    edge_service.node_exists = _boom  # type: ignore[assignment]
    try:
        ctx = _make_mcp_ctx(edge_service)
        result = await get_provenance_tool(ctx, memory_id=fake_id)

        assert isinstance(result, dict)
        assert "error" not in result
        assert result["exists"] is False
        assert result["exists_checked"] is False
        assert result["provenance_chain"] == []
    finally:
        edge_service.node_exists = original  # type: ignore[assignment]


@pytest.mark.asyncio(loop_scope="module")
async def test_get_provenance_mcp_negative_depth_clamped(
    edge_service: MemoryEdgeService,
    neo4j_async_driver: AsyncDriver,
    run_id: str,
) -> None:
    """max_depth=-1 => silently clamped to 1; no exception, valid shape."""
    a = f"prov-mcp-neg-a-{run_id}"
    b = f"prov-mcp-neg-b-{run_id}"
    c = f"prov-mcp-neg-c-{run_id}"
    try:
        await _create_test_nodes(neo4j_async_driver, [a, b, c], run_id)
        await edge_service.create_edge(_make_edge(a, b, "SUPERSEDES", run_id))
        await edge_service.create_edge(_make_edge(b, c, "SUPERSEDES", run_id))

        ctx = _make_mcp_ctx(edge_service)
        result = await get_provenance_tool(
            ctx, memory_id=a, max_depth=-1
        )

        # No exception, valid dict, no error envelope
        assert isinstance(result, dict)
        assert "error" not in result
        # Depth silently clamped to 1 => only b reachable, not c
        assert result["max_depth"] == 1
        chain_ids = {hop["memory_id"] for hop in result["provenance_chain"]}
        assert chain_ids == {b}
        assert c not in chain_ids
        assert result["exists"] is True
    finally:
        await _teardown_test_nodes(neo4j_async_driver)
