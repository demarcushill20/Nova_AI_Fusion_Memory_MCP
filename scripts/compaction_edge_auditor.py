"""COMPACTED_FROM edge auditor for PLAN-0759 Sprint 21 Phase 4.

Belt-and-suspenders check that the prompt-driven compaction wiring shipped
in Sprint 20 is actually producing COMPACTED_FROM edges in Neo4j. Agents
can silently ignore prompt instructions; this auditor runs post-cron and
raises an alert if zero edges were created in the last 24 hours.

Two alert states are distinguished so the operator can tell them apart:

* ``alert_dead_letter`` — zero edges in the lookback window AND zero
  edges total across all time. The wiring has never produced output;
  agents are not emitting ``_compacted_from`` metadata at all.
* ``alert_regression`` — zero edges in the lookback window but
  ``total > 0``. Edges existed historically and then stopped; something
  regressed.

All external calls (Neo4j read, vault write, Slack POST) are fail-open.
The auditor exits 0 by default so cron stays clean; ``--fail-on-alert``
flips it to exit 1 when an alert variant fires.

CLI:

.. code-block:: bash

    NEO4J_URI=bolt://localhost:7687 \\
        python3 -m scripts.compaction_edge_auditor \\
            --lookback-hours 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# Reuse helpers from the Sprint 19 SLO alert shim rather than duplicate
# them. All three helpers are fail-open and already covered by tests.
from scripts.slo_alert_check import (
    _DEFAULT_VAULT,
    _safe_slo_token,
    write_vault_alert,
)

logger = logging.getLogger("compaction_edge_auditor")

_REPO_ROOT = Path(__file__).resolve().parent.parent

_STATUS_OK = "ok"
_STATUS_DEAD_LETTER = "alert_dead_letter"
_STATUS_REGRESSION = "alert_regression"


# --------------------------------------------------------------------- #
# Neo4j query                                                           #
# --------------------------------------------------------------------- #


async def _fetch_compaction_counts_async(
    since_iso: str,
) -> dict[str, Any] | None:
    """Query Neo4j for COMPACTED_FROM counts. Fail-open to None.

    Returns ``{"recent": int, "total": int, "sample_run_ids": list[str]}``
    on success. ``None`` on any connection / query / import failure — the
    caller treats that as "status ok, logged" (we do NOT promote
    connection errors to alerts; the operator sees them in the log).

    ``since_iso`` is an ISO-8601 string (UTC). Edges store ``created_at``
    as ISO strings (see ``edge_service.on_memory_compact`` at
    edge_service.py:951 where ``created_at=now_iso``); ISO string lex-order
    matches chronological order so ``>=`` comparison in Cypher is correct.
    """
    try:
        from neo4j import AsyncGraphDatabase
    except Exception as exc:
        logger.warning("neo4j client import failed: %s", exc)
        return None

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    auth_kwargs: dict[str, Any] = {}
    if user and password:
        auth_kwargs["auth"] = (user, password)

    recent_cypher = (
        "MATCH ()-[r:COMPACTED_FROM]->() "
        "WHERE r.created_at >= $since_iso "
        "RETURN count(r) AS c, collect(DISTINCT r.run_id)[..5] AS sample_run_ids"
    )
    total_cypher = "MATCH ()-[r:COMPACTED_FROM]->() RETURN count(r) AS total"

    driver = None
    try:
        driver = AsyncGraphDatabase.driver(uri, **auth_kwargs)
        async with driver.session() as session:
            recent_result = await session.run(
                recent_cypher, since_iso=since_iso
            )
            recent_record = await recent_result.single()
            recent = int(recent_record["c"]) if recent_record else 0
            sample_run_ids: list[str] = []
            if recent_record:
                raw_ids = recent_record.get("sample_run_ids") or []
                sample_run_ids = [str(r) for r in raw_ids if r is not None]

            total_result = await session.run(total_cypher)
            total_record = await total_result.single()
            total = int(total_record["total"]) if total_record else 0
        return {
            "recent": recent,
            "total": total,
            "sample_run_ids": sample_run_ids,
        }
    except Exception as exc:
        logger.warning("neo4j compaction-count query failed: %s", exc)
        return None
    finally:
        if driver is not None:
            try:
                await driver.close()
            except Exception:
                pass


def fetch_compaction_counts(since_iso: str) -> dict[str, Any] | None:
    """Sync wrapper around the async Neo4j query. Fail-open to None."""
    try:
        return asyncio.run(_fetch_compaction_counts_async(since_iso))
    except Exception as exc:
        logger.warning("neo4j compaction-count top-level failed: %s", exc)
        return None


# --------------------------------------------------------------------- #
# Alert decision                                                        #
# --------------------------------------------------------------------- #


def classify_status(counts: dict[str, Any] | None) -> str:
    """Map Neo4j counts to one of the three status strings.

    ``None`` input (query skipped or failed) maps to ``ok`` — connection
    errors are logged upstream but never alert.
    """
    if counts is None:
        return _STATUS_OK
    recent = int(counts.get("recent", 0))
    total = int(counts.get("total", 0))
    if recent > 0:
        return _STATUS_OK
    if total > 0:
        return _STATUS_REGRESSION
    return _STATUS_DEAD_LETTER


# --------------------------------------------------------------------- #
# Alert rendering                                                       #
# --------------------------------------------------------------------- #


def build_audit_markdown(
    status: str,
    recent: int,
    total: int,
    lookback_hours: int,
    sample_run_ids: list[str],
) -> str:
    """Render a vault note for an alert status. ``ok`` never calls this."""
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if status == _STATUS_DEAD_LETTER:
        headline = "dead-letter: zero COMPACTED_FROM edges ever"
        remediation = (
            "The prompt-driven compaction wiring shipped in Sprint 20 has "
            "produced zero edges since Neo4j was first populated. Likely "
            "causes: (a) the daily-summary cron has not run since the "
            "wiring shipped, (b) agents are not emitting the "
            "`_compacted_from` metadata key on `upsert_memory`, (c) the "
            "Fusion MCP hook that creates the edge is broken. Check "
            "`scripts/daily_summary.py` prompt, verify the next cron run "
            "emits `_compacted_from` in its `upsert_memory` call, and "
            "inspect `app/services/memory_service.py` around line 1382."
        )
    elif status == _STATUS_REGRESSION:
        headline = (
            f"regression: zero COMPACTED_FROM edges in the last "
            f"{lookback_hours}h (total history: {total})"
        )
        remediation = (
            "COMPACTED_FROM edges existed historically but have stopped. "
            "Most likely the daily-summary cron has silently regressed "
            "(prompt change dropped the `_compacted_from` key, the agent "
            "started ignoring the instruction, or the cron is no longer "
            "firing). Inspect the most recent "
            "`LOGS/daily_summary_cron.log`, confirm a summary ran, and "
            "verify its `upsert_memory` call carried `_compacted_from`."
        )
    else:
        # Defensive; classify_status() should never return ok here.
        headline = "unknown audit status"
        remediation = "Inspect the auditor log."

    sample_block = (
        "\n".join(f"- `{rid}`" for rid in sample_run_ids[:5])
        if sample_run_ids
        else "- (none observed in lookback window)"
    )
    return (
        "---\n"
        "type: compaction-audit\n"
        "plan_id: PLAN-0759-assoc-linking\n"
        f"status: {status}\n"
        f"recent_count: {recent}\n"
        f"total_count: {total}\n"
        f"lookback_hours: {lookback_hours}\n"
        f"created: {created}\n"
        "tags:\n"
        '- "#type/alert"\n'
        '- "#source/compaction-edge-auditor"\n'
        f'- "#audit/{status}"\n'
        "---\n\n"
        f"# Compaction edge audit: {headline}\n\n"
        f"- **Status**: `{status}`\n"
        f"- **Recent edges ({lookback_hours}h)**: `{recent}`\n"
        f"- **Total edges (all-time)**: `{total}`\n"
        f"- **Generated**: {created} (UTC)\n\n"
        "## Sample run_ids observed in window\n\n"
        f"{sample_block}\n\n"
        "## Remediation\n\n"
        f"{remediation}\n"
    )


def post_slack(
    webhook_url: str | None,
    status: str,
    recent: int,
    total: int,
    lookback_hours: int,
    dry_run: bool,
) -> bool:
    """POST an audit summary to Slack. Fail-open on every error path.

    This is structurally identical to ``slo_alert_check.post_slack`` but
    typed to an audit payload rather than an SLOResult.
    """
    payload = {
        "text": (
            f"COMPACTION AUDIT {status}: "
            f"recent_{lookback_hours}h={recent} total={total}"
        )
    }
    if dry_run:
        print(f"[DRY-RUN] would POST to Slack: {json.dumps(payload)}")
        return True
    if not webhook_url:
        logger.warning("SLACK_ALERT_WEBHOOK unset; skipping Slack notify")
        return False
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
        return True
    except Exception as exc:
        logger.warning("slack post failed: %s", exc)
        return False


# --------------------------------------------------------------------- #
# CLI + orchestration                                                   #
# --------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "COMPACTED_FROM edge auditor for PLAN-0759 Sprint 21 Phase 4. "
            "Detects silent prompt-wiring regressions."
        )
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Window (hours) over which to count recent edges. Default 24.",
    )
    parser.add_argument("--vault-path", type=Path, default=_DEFAULT_VAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit 1 when status is an alert variant (default: exit 0).",
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip Neo4j entirely; returns status ok. Useful for CI.",
    )
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> int:
    """Execute one audit pass. Returns the desired process exit code."""
    from datetime import timedelta

    lookback_hours = max(1, int(args.lookback_hours))
    since_dt = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(
        "running compaction-edge audit: lookback_hours=%d since_iso=%s "
        "skip_neo4j=%s dry_run=%s",
        lookback_hours,
        since_iso,
        args.skip_neo4j,
        args.dry_run,
    )

    counts: dict[str, Any] | None = None
    if not args.skip_neo4j:
        counts = fetch_compaction_counts(since_iso)
    status = classify_status(counts)

    recent = int(counts.get("recent", 0)) if counts else 0
    total = int(counts.get("total", 0)) if counts else 0
    sample_run_ids = list(counts.get("sample_run_ids", [])) if counts else []

    # Always emit the grep-friendly summary line, regardless of status.
    summary_line = (
        f"compaction_audit recent_{lookback_hours}h={recent} "
        f"total={total} status={status}"
    )
    print(summary_line)

    if status in (_STATUS_DEAD_LETTER, _STATUS_REGRESSION):
        logger.warning(summary_line)
        content = build_audit_markdown(
            status=status,
            recent=recent,
            total=total,
            lookback_hours=lookback_hours,
            sample_run_ids=sample_run_ids,
        )
        today_iso = date.today().isoformat()
        # Sanitize status token before templating — defense-in-depth
        # against a future refactor sneaking traversal-laden values into
        # the filename pattern.
        safe_status = _safe_slo_token(status)
        filename = f"compaction-audit-{safe_status}-{today_iso}.md"
        relative = Path("70-debugging") / filename
        # write_vault_alert currently gates on frontmatter type=='slo-alert'
        # in the reused helper. For our audit note the frontmatter type is
        # 'compaction-audit', so the clobber check will refuse to overwrite
        # even our own prior note — which is exactly the daily idempotency
        # behavior we want. First call of the day writes; same-day re-runs
        # are no-ops (the helper logs + returns False).
        wrote = write_vault_alert(
            args.vault_path, relative, content, args.dry_run
        )
        if not wrote:
            logger.info(
                "vault write returned False (expected for same-day re-run)"
            )
        slack_url = os.environ.get("SLACK_ALERT_WEBHOOK")
        post_slack(
            slack_url,
            status=status,
            recent=recent,
            total=total,
            lookback_hours=lookback_hours,
            dry_run=args.dry_run,
        )
    else:
        logger.info(summary_line)

    if args.fail_on_alert and status != _STATUS_OK:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run_audit(args)
    except Exception as exc:
        logger.error("unhandled error in run_audit: %s", exc, exc_info=True)
        return 1 if args.fail_on_alert else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
