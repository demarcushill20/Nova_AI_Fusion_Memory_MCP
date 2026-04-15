"""Session-id coverage monitor for PLAN-0759 Phase 4 (Sprint 11).

Overview
--------

This is the **Phase 4 completion gate** for PLAN-0759. The v2 plan's Step
4.3 requires that a high-enough fraction of live ``:base`` memory nodes
carry a ``session_id`` property before temporal linking can be marked
complete — otherwise the ``TemporalLinker`` has no way to scope
predecessor lookups and the Phase 4 feature ships into a graph that
cannot actually benefit from it.

The monitor issues two read-only Cypher queries against the live
``nova_neo4j_db`` container:

1. **All-time coverage**:

   .. code-block:: cypher

      MATCH (n:base)
      RETURN count(n) AS total,
             count(n.session_id) AS with_session_id,
             count(n) - count(n.session_id) AS without_session_id

2. **Recent coverage** (default ``--cutoff-days 30``). The v2 plan is
   explicit that the gate is about *recent* memories, because the
   Chronological Upgrade (Fusion Memory Phases 1-3) may not have
   retroactively backfilled older nodes:

   .. code-block:: cypher

      MATCH (n:base)
      WHERE n.event_time > $cutoff_iso
      RETURN count(n) AS recent_total,
             count(n.session_id) AS recent_with_session

   Note that the recent filter uses ``event_time`` (the Fusion Memory
   per-node ISO timestamp — see the Phase 0 schema audit at
   ``nova-core/MEMORY/plans/PLAN-0759/phase0_schema_audit.md`` which
   enumerates the sampled ``:base`` properties). A ``created_at``
   property does **not** exist on ``:base`` nodes in this graph.

Output
------

- JSON report to stdout (machine-readable, used by CI).
- Human-readable Markdown mirror written to
  ``nova-core/MEMORY/plans/PLAN-0759/phase4_coverage_report.md`` so the
  operator can review the gate decision asynchronously.

Exit codes
----------

- **0**: both the all-time and the recent coverage percentages meet the
  threshold. The Phase 4 gate is considered PASS.
- **2**: at least one coverage percentage is strictly below the
  threshold. The Phase 4 gate is BLOCKED; Sprint 11 must not ship the
  ``TemporalLinker`` hook until the upstream Chronological Upgrade is
  verified in a follow-up pass.
- **1**: unexpected runtime failure (driver error, etc).

The v2 plan's default threshold is **50%**; the monitor accepts
``--threshold-pct`` to tighten or (for diagnostic runs only) loosen it.
Loosening is intentional friction: the default argument is what ships
in CI.

Usage
-----

.. code-block:: bash

   python3 -m scripts.assoc_session_coverage_check
   python3 -m scripts.assoc_session_coverage_check --cutoff-days 30 --threshold-pct 50
   python3 -m scripts.assoc_session_coverage_check --uri bolt://localhost:7687

No writes are issued against Neo4j. This script is **read-only** —
enforced by the absence of any ``MERGE``/``CREATE``/``SET``/``DELETE``
in the query text.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from neo4j import GraphDatabase
from neo4j import exceptions as neo4j_exceptions


DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_DATABASE = "neo4j"
DEFAULT_CUTOFF_DAYS = 30
DEFAULT_THRESHOLD_PCT = 50.0

# Where the human-readable report is mirrored. This lives in the
# nova-core operator vault, not in the Fusion Memory repo, because
# the operator reviews Phase-gate reports from the plan folder.
REPORT_PATH = Path(
    "/home/nova/nova-core/MEMORY/plans/PLAN-0759/phase4_coverage_report.md"
)

ALL_TIME_CYPHER = (
    "MATCH (n:base)\n"
    "RETURN count(n) AS total,\n"
    "       count(n.session_id) AS with_session_id,\n"
    "       count(n) - count(n.session_id) AS without_session_id"
)

RECENT_CYPHER = (
    "MATCH (n:base)\n"
    "WHERE n.event_time > $cutoff_iso\n"
    "RETURN count(n) AS recent_total,\n"
    "       count(n.session_id) AS recent_with_session"
)


def _pct(num: int, denom: int) -> float:
    """Safe percentage — returns 0.0 on divide-by-zero."""
    if denom == 0:
        return 0.0
    return round((num / denom) * 100.0, 2)


def _compute_cutoff_iso(cutoff_days: int) -> str:
    """UTC ISO-8601 cutoff string for the ``--cutoff-days`` window."""
    return (
        datetime.now(tz=timezone.utc) - timedelta(days=cutoff_days)
    ).isoformat()


def run_coverage_check(
    uri: str,
    database: str,
    cutoff_days: int,
    threshold_pct: float,
) -> dict[str, Any]:
    """Open a driver, run the two read-only queries, return a report dict.

    The report structure is stable and documented; the Markdown mirror
    and the JSON stdout both derive from this dict.
    """
    cutoff_iso = _compute_cutoff_iso(cutoff_days)

    driver = GraphDatabase.driver(uri, auth=None)
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            all_time_rec = session.run(ALL_TIME_CYPHER).single()
            recent_rec = session.run(
                RECENT_CYPHER, {"cutoff_iso": cutoff_iso}
            ).single()
    finally:
        driver.close()

    total = int(all_time_rec["total"]) if all_time_rec else 0
    with_sid = int(all_time_rec["with_session_id"]) if all_time_rec else 0
    without_sid = int(all_time_rec["without_session_id"]) if all_time_rec else 0
    coverage_pct = _pct(with_sid, total)

    recent_total = int(recent_rec["recent_total"]) if recent_rec else 0
    recent_with_sid = (
        int(recent_rec["recent_with_session"]) if recent_rec else 0
    )
    recent_coverage_pct = _pct(recent_with_sid, recent_total)

    gate_all_time = coverage_pct >= threshold_pct
    gate_recent = recent_coverage_pct >= threshold_pct
    gate_pass = gate_all_time and gate_recent

    return {
        "measured_at": datetime.now(tz=timezone.utc).isoformat(),
        "uri": uri,
        "database": database,
        "cutoff_days": cutoff_days,
        "cutoff_iso": cutoff_iso,
        "threshold_pct": threshold_pct,
        "all_time": {
            "total": total,
            "with_session_id": with_sid,
            "without_session_id": without_sid,
            "coverage_pct": coverage_pct,
            "gate_pass": gate_all_time,
        },
        "recent": {
            "cutoff_days": cutoff_days,
            "cutoff_iso": cutoff_iso,
            "recent_total": recent_total,
            "recent_with_session_id": recent_with_sid,
            "recent_coverage_pct": recent_coverage_pct,
            "gate_pass": gate_recent,
        },
        "gate": "PASS" if gate_pass else "BLOCKED",
        "exit_code": 0 if gate_pass else 2,
    }


def _format_markdown_report(report: dict[str, Any]) -> str:
    """Render the JSON report as a human-readable Markdown page."""
    at = report["all_time"]
    rec = report["recent"]
    gate = report["gate"]
    badge = "PASS" if gate == "PASS" else "BLOCKED"

    lines: list[str] = [
        "# PLAN-0759 Phase 4 — session_id Coverage Gate Report",
        "",
        f"- **Measured at**: {report['measured_at']}",
        f"- **Neo4j URI**: `{report['uri']}`",
        f"- **Database**: `{report['database']}`",
        f"- **Cutoff window**: last {report['cutoff_days']} days",
        f"- **Cutoff ISO**: {report['cutoff_iso']}",
        f"- **Threshold**: {report['threshold_pct']}%",
        f"- **Gate**: **{badge}**",
        "",
        "## All-time `:base` coverage",
        "",
        f"- Total `:base` nodes: **{at['total']}**",
        f"- With `session_id`: **{at['with_session_id']}**",
        f"- Without `session_id`: **{at['without_session_id']}**",
        f"- Coverage: **{at['coverage_pct']}%**",
        f"- Gate (all-time): **{'PASS' if at['gate_pass'] else 'BLOCKED'}**",
        "",
        "## Recent `:base` coverage",
        "",
        f"- Cutoff: last {rec['cutoff_days']} days ({rec['cutoff_iso']})",
        f"- Recent total: **{rec['recent_total']}**",
        f"- Recent with `session_id`: **{rec['recent_with_session_id']}**",
        f"- Recent coverage: **{rec['recent_coverage_pct']}%**",
        f"- Gate (recent): **{'PASS' if rec['gate_pass'] else 'BLOCKED'}**",
        "",
        "## Decision",
        "",
    ]
    if gate == "PASS":
        lines.extend(
            [
                "The `session_id` coverage on live `:base` nodes meets the",
                f"Phase 4 gate threshold of {report['threshold_pct']}% for both",
                "the all-time population and the recent window. Sprint 11 may",
                "proceed to wire the `TemporalLinker` hook into",
                "`MemoryService.perform_upsert()` under the",
                "`ASSOC_TEMPORAL_WRITE_ENABLED` flag, land the integration",
                "tests, and record the latency baseline.",
            ]
        )
    else:
        lines.extend(
            [
                "The `session_id` coverage on live `:base` nodes is **below**",
                f"the Phase 4 gate threshold of {report['threshold_pct']}%.",
                "Phase 4 completion is **BLOCKED**. Sprint 11 must not wire",
                "the temporal linker until the upstream Fusion Memory",
                "Chronological Upgrade is verified to be backfilling",
                "`session_id` onto freshly written memories. Return to the",
                "orchestrator with this report so the operator can decide",
                "whether to (a) run an explicit backfill pass, (b) tighten",
                "the write path so all new memories carry `session_id`, or",
                "(c) re-scope Phase 4.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _write_report(report: dict[str, Any]) -> None:
    """Mirror the JSON report as Markdown to the plan folder."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_format_markdown_report(report), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Returns a shell exit code: 0 on gate PASS, 2 on gate BLOCKED, 1 on
    driver/IO failure (distinct so CI can tell "coverage too low" from
    "could not measure coverage").
    """
    parser = argparse.ArgumentParser(
        description="PLAN-0759 Phase 4 session_id coverage gate",
    )
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--cutoff-days", type=int, default=DEFAULT_CUTOFF_DAYS)
    parser.add_argument(
        "--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT
    )
    parser.add_argument(
        "--no-write-report",
        action="store_true",
        help="Do not mirror the Markdown report to the plan folder.",
    )
    args = parser.parse_args(argv)

    try:
        report = run_coverage_check(
            uri=args.uri,
            database=args.database,
            cutoff_days=args.cutoff_days,
            threshold_pct=args.threshold_pct,
        )
    except (
        neo4j_exceptions.ServiceUnavailable,
        neo4j_exceptions.AuthError,
        OSError,
    ) as exc:
        err_report = {
            "gate": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "exit_code": 1,
        }
        print(json.dumps(err_report, indent=2))
        return 1

    if not args.no_write_report:
        _write_report(report)

    print(json.dumps(report, indent=2))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
