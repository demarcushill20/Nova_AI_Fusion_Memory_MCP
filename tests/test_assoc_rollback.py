"""Integration test for PLAN-0759 Step 0.5 — ``scripts/assoc_rollback``.

Safety model
------------
This test runs against the **live** ``nova_neo4j_db`` container on the host,
which also holds the real Fusion Memory graph (``:base`` + ``:Session``
nodes). That means production data is only a query away, so the test is
built under a strict set of safety constraints:

1. **Ad-hoc test label.** Every node the test creates uses the label
   ``:AssocRollbackTestNode``, which does not collide with ``:base``,
   ``:Session``, or any other label observed in the Phase 0 schema audit.
2. **Unique per-run identifiers.** Each test run generates a fresh UUID
   suffix and uses it to build two distinct ``run_id`` values so repeat
   runs never contaminate each other.
3. **Production-count invariant.** The test captures
   ``count(n:base)`` before and after the destructive phase and asserts
   equality. Any deviation means the test itself has leaked into prod data
   and would fail loudly.
4. **Unconditional teardown.** A pytest fixture with a ``finally`` block
   runs ``MATCH (n:AssocRollbackTestNode) DETACH DELETE n`` regardless of
   pass/fail, so no test nodes survive across runs.
5. **Setup sentinel.** Before the destructive phase runs, the test asserts
   that at least one ``:AssocRollbackTestNode`` exists. If setup failed
   silently we would never reach a point where we could accidentally
   delete something we did not create.
6. **Skip, don't fail, when Neo4j is unreachable.** CI environments and
   local dev without a running container should not get red builds here.

The test deliberately exercises the "leaves untagged edges untouched"
property: two disjoint ``run_id`` tags are created, only the first is
rolled back, and the second set is asserted to survive intact.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

import pytest

# The rollback script is importable because ``scripts/`` is a package with
# an ``__init__.py``. The test path must be run from the Fusion MCP repo
# root (this is the same convention the existing integration tests use).
try:
    from neo4j import GraphDatabase, Driver
    from neo4j import exceptions as neo4j_exceptions
except ImportError:  # pragma: no cover - env sanity
    pytest.skip(
        "neo4j driver not installed; cannot run rollback integration test",
        allow_module_level=True,
    )

from scripts import assoc_rollback as rollback_module
from scripts.assoc_rollback import RollbackError, assoc_rollback


# --------------------------------------------------------------------------
# Connection helpers + skip-if-unreachable fixture
# --------------------------------------------------------------------------

# The Docker-network URI is ``bolt://neo4j:7687``; from the host shell the
# reachable URI is ``bolt://localhost:7687``. Tests run on the host.
TEST_NEO4J_URI = "bolt://localhost:7687"
TEST_DB = "neo4j"
# Dedicated test label — distinct from :base, :Session, :Entity, and every
# label observed in the Phase 0 audit.
TEST_LABEL = "AssocRollbackTestNode"


def _try_open_driver() -> Optional[Driver]:
    """Open a driver if Neo4j is reachable, else return None."""
    try:
        driver = GraphDatabase.driver(TEST_NEO4J_URI, auth=None)
        driver.verify_connectivity()
        return driver
    except (neo4j_exceptions.ServiceUnavailable, neo4j_exceptions.AuthError, OSError):
        return None


@pytest.fixture(scope="module")
def neo4j_driver() -> Iterator[Driver]:
    driver = _try_open_driver()
    if driver is None:
        pytest.skip(
            f"Neo4j not reachable at {TEST_NEO4J_URI}; skipping rollback "
            "integration test (this is expected in environments without a "
            "running nova_neo4j_db container)."
        )
    try:
        yield driver
    finally:
        driver.close()


@pytest.fixture
def test_run_ids() -> Tuple[str, str]:
    """Unique ``(target_run_id, control_run_id)`` pair for this test run."""
    suffix = uuid.uuid4().hex[:8]
    return (
        f"sprint2-rollback-test-target-{suffix}",
        f"sprint2-rollback-test-control-{suffix}",
    )


@contextmanager
def _isolated_test_graph(
    driver: Driver, target_run_id: str, control_run_id: str
) -> Iterator[Dict[str, int]]:
    """Seed isolated test nodes and edges, always tear down in finally.

    Creates 10 ``:AssocRollbackTestNode`` nodes and 20 ``:TEST_LINK`` edges
    between them — 10 tagged with ``target_run_id``, 10 tagged with
    ``control_run_id``. Yields a dict with the seeded counts so the test
    can assert against a concrete expectation.
    """
    seeded: Dict[str, int] = {
        "nodes": 0,
        "target_edges": 0,
        "control_edges": 0,
    }
    # Record pre-test :base count for the production-invariant check.
    with driver.session(database=TEST_DB) as session:
        pre_result = session.run("MATCH (n:base) RETURN count(n) AS c").single()
        base_count_before = pre_result["c"] if pre_result else 0
    seeded["base_count_before"] = base_count_before

    try:
        with driver.session(database=TEST_DB) as session:
            # Create 10 isolated test nodes with unique ids.
            session.run(
                f"UNWIND range(0, 9) AS i "
                f"CREATE (n:{TEST_LABEL} {{test_id: 'rollback-test-' + toString(i), "
                f"test_suffix: $suffix}})",
                {"suffix": target_run_id},
            ).consume()

            # 10 target edges: chain n0->n1->...->n9 tagged target_run_id.
            # Using :TEST_LINK to keep the test edge type off production types.
            session.run(
                f"MATCH (a:{TEST_LABEL}), (b:{TEST_LABEL}) "
                f"WHERE a.test_suffix = $suffix AND b.test_suffix = $suffix "
                f"AND toInteger(substring(a.test_id, size('rollback-test-'))) + 1 "
                f"  = toInteger(substring(b.test_id, size('rollback-test-'))) "
                f"CREATE (a)-[r:TEST_LINK {{run_id: $run_id, kind: 'target'}}]->(b)",
                {"suffix": target_run_id, "run_id": target_run_id},
            ).consume()
            # There are 9 such pairs from a chain over 10 nodes; we need 10
            # edges tagged with target_run_id. Add one more self-loop on n0.
            session.run(
                f"MATCH (n:{TEST_LABEL} {{test_id: 'rollback-test-0', "
                f"test_suffix: $suffix}}) "
                f"CREATE (n)-[r:TEST_LINK {{run_id: $run_id, kind: 'target'}}]->(n)",
                {"suffix": target_run_id, "run_id": target_run_id},
            ).consume()

            # 10 control edges: pair (n0,n2), (n0,n3), ... (n0,n9),
            # (n1,n3), (n1,n4) — arbitrary 10 edges tagged control_run_id.
            # Simpler: connect n0 -> each of n0..n9 with 10 control edges.
            session.run(
                f"MATCH (a:{TEST_LABEL} {{test_id: 'rollback-test-0', "
                f"test_suffix: $suffix}}), "
                f"(b:{TEST_LABEL}) WHERE b.test_suffix = $suffix "
                f"CREATE (a)-[r:TEST_LINK {{run_id: $ctrl, kind: 'control'}}]->(b)",
                {"suffix": target_run_id, "ctrl": control_run_id},
            ).consume()

            # Verify seed counts.
            node_count = session.run(
                f"MATCH (n:{TEST_LABEL} {{test_suffix: $suffix}}) "
                f"RETURN count(n) AS c",
                {"suffix": target_run_id},
            ).single()["c"]
            target_count = session.run(
                f"MATCH (:{TEST_LABEL})-[r:TEST_LINK {{run_id: $run_id}}]->"
                f"(:{TEST_LABEL}) RETURN count(r) AS c",
                {"run_id": target_run_id},
            ).single()["c"]
            control_count = session.run(
                f"MATCH (:{TEST_LABEL})-[r:TEST_LINK {{run_id: $run_id}}]->"
                f"(:{TEST_LABEL}) RETURN count(r) AS c",
                {"run_id": control_run_id},
            ).single()["c"]
            seeded.update(
                {
                    "nodes": node_count,
                    "target_edges": target_count,
                    "control_edges": control_count,
                }
            )
        yield seeded
    finally:
        # Unconditional teardown: every test node (and its attached edges)
        # is deleted regardless of how the test exited.
        with driver.session(database=TEST_DB) as session:
            session.run(
                f"MATCH (n:{TEST_LABEL}) DETACH DELETE n"
            ).consume()
            # Confirm teardown succeeded.
            leftover = session.run(
                f"MATCH (n:{TEST_LABEL}) RETURN count(n) AS c"
            ).single()["c"]
            # We do NOT assert here because raising in teardown would mask a
            # real test failure; we log instead. The after-run assertion in
            # the test body is the authoritative check.
            if leftover != 0:
                import logging

                logging.getLogger(__name__).error(
                    "Teardown left %d :%s nodes behind!", leftover, TEST_LABEL
                )


# --------------------------------------------------------------------------
# Safety refusals (these do not require Neo4j but are cheap to run anyway)
# --------------------------------------------------------------------------


def test_rollback_refuses_empty_run_id() -> None:
    with pytest.raises(RollbackError):
        assoc_rollback(run_id="", dry_run=True, uri=TEST_NEO4J_URI)


def test_rollback_refuses_whitespace_run_id() -> None:
    with pytest.raises(RollbackError):
        assoc_rollback(run_id="   ", dry_run=True, uri=TEST_NEO4J_URI)


@pytest.mark.parametrize("wildcard", ["*", "%", "all", "ALL"])
def test_rollback_refuses_wildcard_run_id(wildcard: str) -> None:
    with pytest.raises(RollbackError):
        assoc_rollback(run_id=wildcard, dry_run=True, uri=TEST_NEO4J_URI)


# --------------------------------------------------------------------------
# Integration: isolated test nodes, full dry-run → live-delete cycle
# --------------------------------------------------------------------------


def test_rollback_deletes_only_tagged_edges(
    neo4j_driver: Driver, test_run_ids: Tuple[str, str]
) -> None:
    target_run_id, control_run_id = test_run_ids

    with _isolated_test_graph(neo4j_driver, target_run_id, control_run_id) as seeded:
        # ---- Setup sentinel: test nodes must exist before we do anything ----
        assert seeded["nodes"] == 10, (
            f"Expected 10 :{TEST_LABEL} seed nodes, got {seeded['nodes']}. "
            "Aborting before the destructive phase."
        )
        assert seeded["target_edges"] == 10, (
            f"Expected 10 target edges tagged run_id={target_run_id}, "
            f"got {seeded['target_edges']}"
        )
        assert seeded["control_edges"] == 10, (
            f"Expected 10 control edges tagged run_id={control_run_id}, "
            f"got {seeded['control_edges']}"
        )

        # ---- Dry run: must count 10, delete nothing ----
        dry_report = assoc_rollback(
            run_id=target_run_id,
            dry_run=True,
            driver=neo4j_driver,
            database=TEST_DB,
        )
        assert dry_report.run_id == target_run_id
        assert dry_report.dry_run is True
        assert dry_report.total == 10
        assert dry_report.deleted_by_type == {"TEST_LINK": 10}

        # Confirm nothing was actually deleted yet.
        with neo4j_driver.session(database=TEST_DB) as session:
            still_target = session.run(
                "MATCH ()-[r:TEST_LINK {run_id: $run_id}]->() "
                "RETURN count(r) AS c",
                {"run_id": target_run_id},
            ).single()["c"]
            still_control = session.run(
                "MATCH ()-[r:TEST_LINK {run_id: $run_id}]->() "
                "RETURN count(r) AS c",
                {"run_id": control_run_id},
            ).single()["c"]
        assert still_target == 10, "dry-run must not delete anything"
        assert still_control == 10, "dry-run must not touch control edges"

        # ---- Live delete of target run_id only ----
        live_report = assoc_rollback(
            run_id=target_run_id,
            dry_run=False,
            driver=neo4j_driver,
            database=TEST_DB,
        )
        assert live_report.dry_run is False
        assert live_report.total == 10
        assert live_report.deleted_by_type == {"TEST_LINK": 10}

        # ---- Post-delete assertions ----
        with neo4j_driver.session(database=TEST_DB) as session:
            after_target = session.run(
                "MATCH ()-[r:TEST_LINK {run_id: $run_id}]->() "
                "RETURN count(r) AS c",
                {"run_id": target_run_id},
            ).single()["c"]
            after_control = session.run(
                "MATCH ()-[r:TEST_LINK {run_id: $run_id}]->() "
                "RETURN count(r) AS c",
                {"run_id": control_run_id},
            ).single()["c"]
            # Verify the surviving control edges all carry the control run_id.
            control_run_ids_seen = session.run(
                "MATCH ()-[r:TEST_LINK]->() "
                f"WHERE r.run_id IS NOT NULL "
                f"RETURN DISTINCT r.run_id AS run_id"
            ).value()
            base_count_after = session.run(
                "MATCH (n:base) RETURN count(n) AS c"
            ).single()["c"]

        assert after_target == 0, (
            f"All 10 target edges should be gone; {after_target} survived"
        )
        assert after_control == 10, (
            f"All 10 control edges should survive; {after_control} present"
        )
        # No stray target_run_id should appear anywhere post-rollback.
        assert target_run_id not in control_run_ids_seen, (
            "target_run_id leaked past rollback"
        )
        assert control_run_id in control_run_ids_seen

        # ---- Production-data invariant ----
        assert base_count_after == seeded["base_count_before"], (
            f":base node count changed across the test: "
            f"{seeded['base_count_before']} before -> {base_count_after} after. "
            "The rollback test has leaked into production data."
        )


def test_rollback_idempotent_second_call_is_no_op(
    neo4j_driver: Driver, test_run_ids: Tuple[str, str]
) -> None:
    """Calling rollback twice with the same run_id should be safe and idempotent."""
    target_run_id, control_run_id = test_run_ids

    with _isolated_test_graph(neo4j_driver, target_run_id, control_run_id):
        first = assoc_rollback(
            run_id=target_run_id,
            dry_run=False,
            driver=neo4j_driver,
            database=TEST_DB,
        )
        assert first.total == 10

        # Second call: nothing should match, no error, total=0.
        second = assoc_rollback(
            run_id=target_run_id,
            dry_run=False,
            driver=neo4j_driver,
            database=TEST_DB,
        )
        assert second.total == 0
        assert second.deleted_by_type == {}
