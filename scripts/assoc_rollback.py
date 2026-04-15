"""Associative-link rollback utility for PLAN-0759 Phase 0 / Step 0.5.

Purpose
-------
When PLAN-0759 Phase 1+ introduces new associative-linking edges, a bad
linker run (wrong threshold, buggy entity extractor, a Python crash
mid-batch) can leave a batch of invalid edges in the graph. The v2 plan
explicitly warns against the "delete by raw relationship type" antipattern,
because that would collide with edge types like ``FOLLOWS`` which are owned
by other subsystems (ADR-0759 §6).

The only safe rollback contract is **delete by ``run_id`` metadata** — every
edge a linker writes must carry a ``run_id`` property, and a rollback is
exactly "delete every edge tagged with run_id=X, regardless of type". This
script implements that contract.

Scope
-----
This is a **standalone utility script**. It is imported by
``tests/test_assoc_rollback.py`` for integration testing but is NOT wired
into ``memory_service.py``, ``memory_router.py``, ``graph_client.py``, or
any production code path. PLAN-0759 Phase 1+ will invoke it from the
operator's shell (or from higher-level orchestration) when a rollback is
needed; the Sprint 2 deliverable is just that the tool exists, is safe, and
is tested.

Usage
-----
Dry-run (no deletions, just counts per edge type)::

    python -m scripts.assoc_rollback --run-id run-abc123 --dry-run \
        --uri bolt://localhost:7687

Live rollback::

    python -m scripts.assoc_rollback --run-id run-abc123 \
        --uri bolt://localhost:7687

Safety refusals
---------------
- Missing / empty ``--run-id``: refuse, exit 2.
- Wildcard ``--run-id`` (``*`` or ``%``): refuse, exit 2. We never want a
  rollback that matches everything.
- Any exception from the Cypher layer propagates as exit code 3.

The script is **not** read-only — in non-dry-run mode it issues a DELETE.
That is intentional and is the whole point. The test fixture isolates this
by operating only on an ad-hoc ``:AssocRollbackTestNode`` label that does
not collide with any production label.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from neo4j import GraphDatabase, Driver
    from neo4j import exceptions as neo4j_exceptions
except ImportError as exc:  # pragma: no cover
    print(
        "ERROR: neo4j python driver is not installed. Install with "
        "`pip install neo4j` (pinned in requirements.txt).",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


LOGGER = logging.getLogger("assoc_rollback")


WILDCARD_RUN_IDS: Tuple[str, ...] = ("*", "%", "all", "ALL")


@dataclass
class RollbackReport:
    """Structured rollback summary returned to the caller / printed to stdout."""

    run_id: str
    dry_run: bool
    deleted_by_type: Dict[str, int] = field(default_factory=dict)
    total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dry_run": self.dry_run,
            "deleted_by_type": dict(self.deleted_by_type),
            "total": self.total,
        }


class RollbackError(RuntimeError):
    """Raised when rollback input is invalid (safety refusal)."""


# ---------------------------------------------------------------------------
# Connection helper (mirrors audit_neo4j_schema; intentionally duplicated to
# keep scripts self-contained and avoid cross-script imports).
# ---------------------------------------------------------------------------


def _open_driver(
    uri: str, user: Optional[str], password: Optional[str]
) -> Driver:
    auth = (user, password) if password else None
    driver = GraphDatabase.driver(uri, auth=auth)
    driver.verify_connectivity()
    return driver


# ---------------------------------------------------------------------------
# Core rollback logic — both the CLI and the integration test call this.
# ---------------------------------------------------------------------------


def _validate_run_id(run_id: Optional[str]) -> str:
    if run_id is None:
        raise RollbackError("--run-id is required and must not be empty")
    stripped = run_id.strip()
    if not stripped:
        raise RollbackError("--run-id is required and must not be empty")
    if stripped in WILDCARD_RUN_IDS:
        raise RollbackError(
            f"--run-id='{stripped}' is a wildcard and is refused by policy. "
            "Rollback must target a concrete linker run."
        )
    return stripped


def assoc_rollback(
    *,
    run_id: str,
    dry_run: bool,
    driver: Optional[Driver] = None,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: str = "neo4j",
) -> RollbackReport:
    """Delete edges whose ``run_id`` property matches ``run_id``.

    Either pass a pre-opened ``driver`` (the test path), or pass
    ``uri``/``user``/``password`` and this function will open one itself
    (the CLI path). If a driver is passed in, the caller is responsible
    for closing it.

    Parameters
    ----------
    run_id
        Concrete linker run id. Empty and wildcard values are refused.
    dry_run
        When True, counts and returns matching edges per type but does NOT
        delete anything.
    driver
        Optional pre-opened ``neo4j.Driver``. Preferred by tests so setup
        and rollback share a single connection.
    uri, user, password, database
        Used only when ``driver`` is None.

    Returns
    -------
    RollbackReport
        Summary of the (would-be) deletion, broken down by relationship type.

    Raises
    ------
    RollbackError
        On empty or wildcard ``run_id``.
    neo4j.exceptions.Neo4jError
        On any query failure. Propagates to the CLI exit-code handler.
    """
    run_id = _validate_run_id(run_id)

    owned_driver = False
    if driver is None:
        if uri is None:
            raise RollbackError(
                "assoc_rollback requires either an open driver or a uri"
            )
        driver = _open_driver(uri, user, password)
        owned_driver = True

    report = RollbackReport(run_id=run_id, dry_run=dry_run)

    try:
        with driver.session(database=database) as session:
            # Step 1: always compute per-type counts of matching edges.
            # This gives us a dry-run preview AND the live-delete breakdown.
            count_cypher = (
                "MATCH ()-[r]->() "
                "WHERE r.run_id = $run_id "
                "RETURN type(r) AS rel_type, count(r) AS c"
            )
            LOGGER.info(
                "Counting edges tagged run_id=%s (dry_run=%s)", run_id, dry_run
            )
            count_result = session.run(count_cypher, {"run_id": run_id})
            for record in count_result:
                report.deleted_by_type[record["rel_type"]] = record["c"]
            report.total = sum(report.deleted_by_type.values())

            if dry_run:
                LOGGER.info(
                    "DRY-RUN: would delete %d edges across %d types for run_id=%s",
                    report.total,
                    len(report.deleted_by_type),
                    run_id,
                )
                return report

            if report.total == 0:
                LOGGER.info(
                    "No edges matched run_id=%s; nothing to delete.", run_id
                )
                return report

            # Step 2: live delete. Use a SINGLE write transaction so all-or-
            # nothing applies at the edge-batch level. The WHERE clause
            # filters on run_id, NOT on relationship type, which is exactly
            # the rollback contract the v2 plan specifies.
            delete_cypher = (
                "MATCH ()-[r]->() "
                "WHERE r.run_id = $run_id "
                "DELETE r"
            )
            LOGGER.info(
                "Deleting %d edges tagged run_id=%s", report.total, run_id
            )
            session.run(delete_cypher, {"run_id": run_id}).consume()
            LOGGER.info("Delete complete for run_id=%s", run_id)
            return report
    finally:
        if owned_driver:
            driver.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assoc_rollback",
        description=(
            "PLAN-0759 Phase 0 / Step 0.5 — delete associative-linking edges "
            "by run_id metadata. Never deletes by raw edge type."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Concrete run_id metadata to match. Wildcards are refused.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matching edge counts per type without deleting anything.",
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help=(
            "Neo4j bolt URI. Defaults to $NEO4J_URI or bolt://localhost:7687 "
            "(host-shell URI; the Fusion MCP default bolt://neo4j:7687 only "
            "resolves inside the Docker network)."
        ),
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username. Defaults to $NEO4J_USER or 'neo4j'.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD"),
        help=(
            "Neo4j password. Defaults to $NEO4J_PASSWORD. Leave unset if the "
            "target container has NEO4J_AUTH=none."
        ),
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name. Defaults to 'neo4j'.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Silence the "unknown relationship type" notification warnings that
    # Neo4j 5 emits when a MATCH touches a type the DB has never seen.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    try:
        report = assoc_rollback(
            run_id=args.run_id,
            dry_run=args.dry_run,
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        )
    except RollbackError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except neo4j_exceptions.AuthError as exc:
        print(
            f"ERROR: Neo4j authentication failed for user '{args.user}' at "
            f"{args.uri}: {exc}",
            file=sys.stderr,
        )
        return 2
    except (neo4j_exceptions.ServiceUnavailable, OSError) as exc:
        print(
            f"ERROR: Could not connect to Neo4j at {args.uri}: {exc}",
            file=sys.stderr,
        )
        return 2
    except neo4j_exceptions.Neo4jError as exc:
        print(f"ERROR: Rollback query failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
