"""Phase 8 live-test seeder for provenance edges (Sprint 17).

Idempotently seeds three provenance edges between four pre-selected,
edge-free ``:base`` nodes so the live-test suite can exercise
``get_provenance`` against real data. Always uses the edge service layer
— no raw Cypher. Rollback is by ``run_id`` via ``scripts.assoc_rollback``.

Topology seeded::

    A --[SUPERSEDES]--> B         (pure supersession, leaf = B)
    A --[PROMOTED_FROM]--> C
    C --[COMPACTED_FROM]--> D     (mixed chain: A -> C -> D, leaf = D)

The four entity_ids are fixed constants (selected 2026-04-23 from live
nova-core data) — the 4 IDs are the seeded invariant, do not parameterize.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Make the app package importable when run from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import AsyncGraphDatabase  # noqa: E402

from app.services.associations.edge_service import MemoryEdgeService  # noqa: E402
from app.services.associations.memory_edges import MemoryEdge  # noqa: E402


LOGGER = logging.getLogger("seed_phase8_livetest_provenance")

DEFAULT_RUN_ID = "wt-phase8-livetest-2026-04-23"

# The four nova-core :base nodes selected 2026-04-23. All four verified
# to have zero SUPERSEDES/PROMOTED_FROM/COMPACTED_FROM edges at selection.
ENTITY_A = "005b347afc0f10618dadd5a283f37a58"
ENTITY_B = "015916fdfc5a8d32b280b96d4a46015e"
ENTITY_C = "0161deeeae900bce4cd31d15760f06e1"
ENTITY_D = "0294ca7688ba0396bf0bf3d1b7355683"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _build_edges(run_id: str) -> List[MemoryEdge]:
    ts = _now_iso()
    common = {
        "weight": 1.0,
        "created_at": ts,
        "last_seen_at": ts,
        "created_by": "sprint-17",
        "run_id": run_id,
        "edge_version": 1,
    }
    return [
        MemoryEdge(
            source_id=ENTITY_A,
            target_id=ENTITY_B,
            edge_type="SUPERSEDES",
            **common,
        ),
        MemoryEdge(
            source_id=ENTITY_A,
            target_id=ENTITY_C,
            edge_type="PROMOTED_FROM",
            **common,
        ),
        MemoryEdge(
            source_id=ENTITY_C,
            target_id=ENTITY_D,
            edge_type="COMPACTED_FROM",
            **common,
        ),
    ]


async def _count_prov_edges(driver, database: str) -> dict:
    async with driver.session(database=database) as session:
        result = await session.run(
            "MATCH ()-[r]->() "
            "WHERE type(r) IN ['SUPERSEDES','PROMOTED_FROM','COMPACTED_FROM'] "
            "RETURN type(r) AS t, count(r) AS c"
        )
        return {rec["t"]: rec["c"] async for rec in result}


async def _seed(run_id: str, dry_run: bool, uri: str, database: str) -> int:
    edges = _build_edges(run_id)

    if dry_run:
        print(f"[dry-run] Would create {len(edges)} edges tagged run_id={run_id}:")
        for e in edges:
            print(f"  {e.source_id} -[{e.edge_type}]-> {e.target_id}")
        return 0

    driver = AsyncGraphDatabase.driver(uri, auth=None)
    try:
        await driver.verify_connectivity()
        before = await _count_prov_edges(driver, database)
        print(f"Provenance edge counts before: {before}")

        service = MemoryEdgeService(driver=driver, database=database)
        created = 0
        for edge in edges:
            ok = await service.create_edge(edge)
            if ok:
                created += 1
                LOGGER.info(
                    "created %s -[%s]-> %s",
                    edge.source_id, edge.edge_type, edge.target_id,
                )

        after = await _count_prov_edges(driver, database)
        print(f"Provenance edge counts after:  {after}")
        print(f"Edges created this run: {created} (run_id={run_id})")
        return created
    finally:
        await driver.close()


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="seed_phase8_livetest_provenance",
        description="Seed 3 provenance edges between 4 fixed nova-core :base nodes (Sprint 17).",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_seed(args.run_id, args.dry_run, args.uri, args.database))
    return 0


if __name__ == "__main__":
    sys.exit(main())
