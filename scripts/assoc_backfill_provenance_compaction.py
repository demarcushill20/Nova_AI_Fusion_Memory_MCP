"""PLAN-0759 Phase 5c — retroactive backfill for COMPACTED_FROM edges.

Context
-------

Phase 5c adds a write-time compaction hook to
:func:`app.services.memory_service.MemoryService.perform_upsert`. When an
upstream caller (today: ``nova-core/agents/memory_consolidator.py`` /
``nova-core/agents/memory_router.py`` — **not yet wired at the time this
script lands**) compacts N source memories into a single summary memory,
it passes ``metadata["_compacted_from"] = {"source_ids": [...],
"algorithm": "...", "reason": "..."}`` so the hook can MERGE one
``(summary:base)-[:COMPACTED_FROM]->(source:base)`` edge per source id.

This script is the **retroactive rail**: it scans the live Neo4j graph
for ``:base`` nodes that were compacted *before* the write-time hook was
wired (detectable via a historical ``compacted_from`` / ``_compacted_from``
/ ``source_ids`` property on the summary node) and retro-emits the
``COMPACTED_FROM`` edges via the same
:class:`MemoryEdgeService.on_memory_compact` codepath as the write-time
hook.

**Current state of the graph (as of Phase 5c landing):** compaction data
does NOT yet flow into Fusion Memory — ``nova-core/agents/`` compaction
callers have not been wired to pass the ``_compacted_from`` metadata, so
no historical compaction markers exist on ``:base`` nodes. A dry-run of
this script against the current graph will report "0 candidates found"
and exit cleanly. The script is intentionally implemented to the full
shape now so that the day Phase 5c's cross-repo wiring lands (the
separate B-follow-up), the backfill rail is ready without another
engineering pass.

Sources of truth (do not duplicate logic — call these directly)
---------------------------------------------------------------

- :meth:`MemoryEdgeService.on_memory_compact` — the canonical compaction
  edge writer. Enforces whitelist via ``build_merge_edge_cypher``,
  stamps ``edge_version``, ``created_by="edge_service.on_memory_compact"``,
  and emits one edge per source id with optional attribution metadata.
- :data:`memory_edges.EDGE_VERSION` — schema version stamp (inherited
  through ``on_memory_compact`` → ``MemoryEdge``).

Usage
-----
Dry-run (safe, writes nothing)::

    python -m scripts.assoc_backfill_provenance_compaction \\
        --run-id phase5c-compact-2026-04-21 --dry-run --verbose

Full live run (writes edges)::

    python -m scripts.assoc_backfill_provenance_compaction \\
        --run-id phase5c-compact-2026-04-21 --rate-limit-qps 5.0

Rollback::

    python -m scripts.assoc_rollback \\
        --run-id backfill-phase5c-compact-2026-04-21
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
    from neo4j import exceptions as neo4j_exceptions
except ImportError as exc:  # pragma: no cover
    print(
        "ERROR: neo4j python driver is not installed. Install with "
        "`pip install neo4j` (pinned in requirements.txt).",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


LOGGER = logging.getLogger("assoc_backfill_provenance_compaction")


# ---------------------------------------------------------------------------
# Refusal constants
# ---------------------------------------------------------------------------

WILDCARD_RUN_IDS: Tuple[str, ...] = ("*", "%", "all", "ALL")

#: Sprint test scaffolding prefixes that must NEVER appear in a real backfill
#: run_id.
RESERVED_TEST_PREFIXES: Tuple[str, ...] = (
    "sprint2-",
    "sprint5-",
    "sprint6-",
    "sprint7-",
    "sprint8-",
    "sprint9-",
)

#: Write-time linker prefixes owned by the wired linkers. Always refused so
#: rollback can cleanly distinguish write-time edges from backfill edges.
WT_TEMPORAL_PREFIX: str = "wt-temporal-"
WT_ENTITY_PREFIX: str = "wt-entity-"
WT_LINK_PREFIX: str = "wt-link-"
WT_SUPERSEDE_PREFIX: str = "wt-supersede-"
WT_PROMOTE_PREFIX: str = "wt-promote-"
WT_COMPACT_PREFIX: str = "wt-compact-"


class BackfillError(RuntimeError):
    """Raised when backfill input is invalid (safety refusal)."""


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class BackfillReport:
    """Structured summary of the compaction backfill run."""

    run_id: str
    dry_run: bool
    started_at: str
    completed_at: Optional[str] = None
    candidates_scanned: int = 0
    candidates_skipped_missing_fields: int = 0
    edges_created: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dry_run": self.dry_run,
            "candidates_scanned": self.candidates_scanned,
            "candidates_skipped_missing_fields": self.candidates_skipped_missing_fields,
            "edges_created": self.edges_created,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# run_id validation
# ---------------------------------------------------------------------------


def _validate_run_id(run_id: Optional[str], *, allow_test: bool = False) -> str:
    """Return a normalized run_id or raise :class:`BackfillError`."""
    if run_id is None:
        raise BackfillError("--run-id is required and must not be empty")
    stripped = run_id.strip()
    if not stripped:
        raise BackfillError("--run-id is required and must not be empty")
    if stripped in WILDCARD_RUN_IDS:
        raise BackfillError(
            f"--run-id='{stripped}' is a wildcard and is refused by policy. "
            "Backfill must target a concrete run."
        )
    for prefix in (
        WT_TEMPORAL_PREFIX,
        WT_ENTITY_PREFIX,
        WT_LINK_PREFIX,
        WT_SUPERSEDE_PREFIX,
        WT_PROMOTE_PREFIX,
        WT_COMPACT_PREFIX,
    ):
        if stripped.startswith(prefix):
            raise BackfillError(
                f"--run-id='{stripped}' starts with '{prefix}' which is "
                "reserved for a write-time linker path. Pick a distinct "
                "prefix (e.g. 'phase5c-compact-YYYYMMDD')."
            )
    if not allow_test:
        for prefix in RESERVED_TEST_PREFIXES:
            if stripped.startswith(prefix):
                raise BackfillError(
                    f"--run-id='{stripped}' starts with '{prefix}' which is "
                    "reserved for sprint test scaffolding. Pick a distinct "
                    "prefix (e.g. 'phase5c-compact-YYYYMMDD')."
                )
    return stripped


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _AsyncRateLimiter:
    """Token-spacing rate limiter — qps==0 disables.

    Reproduced from :mod:`scripts.assoc_backfill_temporal` so this script
    has no inter-script imports.
    """

    def __init__(self, qps: float) -> None:
        self._qps = float(qps) if qps and qps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_allowed: float = 0.0

    async def wait(self) -> None:
        if self._qps <= 0:
            return
        min_interval = 1.0 / self._qps
        async with self._lock:
            now = time.monotonic()
            wait_s = self._next_allowed - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + min_interval


# ---------------------------------------------------------------------------
# Candidate scan — searches for historical compaction markers
# ---------------------------------------------------------------------------

#: Cypher that finds ``:base`` nodes with a historical compaction marker.
#:
#: Three property names are checked because the upstream wiring is not
#: finalized yet:
#:   - ``n.compacted_from`` (scalar property naming)
#:   - ``n._compacted_from`` (matches the metadata key that the write-time
#:     hook consumes — kept here in case a legacy writer persisted it as
#:     a flat node property rather than a metadata-only signal).
#:   - ``n.source_ids`` (most likely shape for a historical flat-property
#:     list of source ids).
#:
#: On the current graph this query returns 0 rows.
_COMPACTION_CANDIDATES_CYPHER: str = (
    "MATCH (n:base) "
    "WHERE n.compacted_from IS NOT NULL "
    "   OR n._compacted_from IS NOT NULL "
    "   OR n.source_ids IS NOT NULL "
    "RETURN n.entity_id AS summary_id, "
    "       coalesce(n.compacted_from, n._compacted_from, n.source_ids) "
    "           AS source_ids "
    "ORDER BY n.entity_id ASC"
)


async def _iter_compaction_candidates(
    driver: "AsyncDriver",
    *,
    database: str,
) -> "AsyncIterator[Dict[str, Any]]":
    """Async generator yielding compaction candidate rows.

    Each row is a dict with keys ``summary_id`` and ``source_ids``. The
    caller is responsible for validating that ``source_ids`` is a list of
    non-empty strings.
    """
    async with driver.session(database=database) as session:
        result = await session.run(_COMPACTION_CANDIDATES_CYPHER)
        async for rec in result:
            yield {
                "summary_id": rec["summary_id"],
                "source_ids": rec["source_ids"],
            }
        await result.consume()


# ---------------------------------------------------------------------------
# Core library entry point
# ---------------------------------------------------------------------------


async def backfill_compaction_edges(
    *,
    run_id: str,
    driver: "AsyncDriver",
    database: str = "neo4j",
    max_total: Optional[int] = None,
    dry_run: bool = False,
    rate_limit_qps: float = 5.0,
    verbose: bool = False,
    _allow_test_run_id: bool = False,
    shutdown_event: Optional[asyncio.Event] = None,
) -> BackfillReport:
    """Scan historical compaction markers and retro-emit ``COMPACTED_FROM`` edges.

    Parameters
    ----------
    run_id:
        Concrete run identifier. Wildcards / reserved prefixes refused.
    driver:
        Pre-opened async Neo4j driver. Caller owns lifecycle.
    database:
        Neo4j database name. Defaults to ``"neo4j"``.
    max_total:
        Stop after processing this many candidates. ``None`` = unlimited.
    dry_run:
        When True, count the edges that would be emitted but do NOT write
        anything.
    rate_limit_qps:
        Upper bound on per-candidate edge writes per second.
        Default 5.0. 0 disables.
    verbose:
        Enable DEBUG-level logging.
    _allow_test_run_id:
        Bypass test-prefix refusal. Only used by integration tests.
    shutdown_event:
        Optional ``asyncio.Event``. If set mid-run, the loop exits
        cleanly after the current candidate. Wired to SIGINT in the CLI.
    """
    # Late import — keeps CLI --help cheap and avoids forcing the app
    # package import path when the script is inspected in isolation.
    from app.services.associations.edge_service import MemoryEdgeService

    if verbose:
        LOGGER.setLevel(logging.DEBUG)

    normalized_run_id = _validate_run_id(run_id, allow_test=_allow_test_run_id)
    tagged_run_id = f"backfill-{normalized_run_id}"

    report = BackfillReport(
        run_id=normalized_run_id,
        dry_run=dry_run,
        started_at=datetime.now(tz=timezone.utc).isoformat(),
    )

    rate_limiter = _AsyncRateLimiter(rate_limit_qps)
    edge_service = MemoryEdgeService(driver)

    LOGGER.info(
        "backfill.start run_id=%s tagged=%s dry_run=%s max_total=%s "
        "rate_limit_qps=%s",
        normalized_run_id,
        tagged_run_id,
        dry_run,
        max_total,
        rate_limit_qps,
    )

    processed_counter = 0

    try:
        async for row in _iter_compaction_candidates(
            driver=driver, database=database
        ):
            if shutdown_event is not None and shutdown_event.is_set():
                LOGGER.warning(
                    "backfill.shutdown_requested processed=%d", processed_counter
                )
                break

            report.candidates_scanned += 1

            if max_total is not None and processed_counter >= max_total:
                LOGGER.info(
                    "backfill.max_total_reached processed=%d max_total=%d",
                    processed_counter,
                    max_total,
                )
                break

            summary_id = row.get("summary_id")
            source_ids = row.get("source_ids")

            if not summary_id:
                report.candidates_skipped_missing_fields += 1
                LOGGER.debug(
                    "backfill.skip_missing_summary_id row=%r", row
                )
                continue

            # ``on_memory_compact`` requires a list of non-empty strings.
            # Historical property writers may have persisted a scalar, a
            # list with empty entries, or a list with non-string entries;
            # skip any of those shapes rather than fabricate data.
            if (
                not isinstance(source_ids, list)
                or not source_ids
                or not all(
                    isinstance(sid, str) and sid for sid in source_ids
                )
            ):
                report.candidates_skipped_missing_fields += 1
                LOGGER.debug(
                    "backfill.skip_malformed_source_ids summary_id=%r source_ids=%r",
                    summary_id,
                    source_ids,
                )
                continue

            await rate_limiter.wait()

            if dry_run:
                # Count the edges that would be emitted (one per source).
                report.edges_created += len(source_ids)
                processed_counter += 1
                LOGGER.debug(
                    "backfill.dry_run candidate summary_id=%s source_ids=%r",
                    summary_id,
                    source_ids,
                )
                continue

            try:
                emitted = await edge_service.on_memory_compact(
                    summary_id=summary_id,
                    source_ids=source_ids,
                    run_id=tagged_run_id,
                )
                report.edges_created += int(emitted) if emitted else 0
            except Exception as exc:  # noqa: BLE001 — fail-open per candidate
                report.errors.append(
                    f"summary_id={summary_id} source_ids={source_ids}: "
                    f"{type(exc).__name__}: {exc}"
                )
                LOGGER.warning(
                    "backfill.upsert_failed summary_id=%s error=%s",
                    summary_id,
                    exc,
                )

            processed_counter += 1
    except asyncio.CancelledError:
        LOGGER.warning("backfill.cancelled processed=%d", processed_counter)
        raise
    finally:
        report.completed_at = datetime.now(tz=timezone.utc).isoformat()

    LOGGER.info(
        "backfill.done run_id=%s candidates_scanned=%d edges_created=%d "
        "skipped=%d errors=%d",
        normalized_run_id,
        report.candidates_scanned,
        report.edges_created,
        report.candidates_skipped_missing_fields,
        len(report.errors),
    )
    return report


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assoc_backfill_provenance_compaction",
        description=(
            "PLAN-0759 Phase 5c — retroactively create COMPACTED_FROM "
            "edges for any :base memory carrying a historical compaction "
            "marker. On the current graph this is expected to find 0 "
            "candidates; the script is the backfill rail, ready for when "
            "Phase 5c's cross-repo wiring lands."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Concrete run identifier. Wildcards and reserved prefixes "
            "(wt-*, sprint{2,5,6,7,8,9}-) are refused."
        ),
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j bolt URI. Defaults to $NEO4J_URI or bolt://localhost:7687.",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username. Defaults to $NEO4J_USER or 'neo4j'.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD"),
        help="Neo4j password. Defaults to $NEO4J_PASSWORD.",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name. Defaults to 'neo4j'.",
    )
    parser.add_argument(
        "--max-total",
        type=str,
        default="unlimited",
        help="Max candidates to process. Integer or 'unlimited'. Default unlimited.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count candidates without writing any edges.",
    )
    parser.add_argument(
        "--rate-limit-qps",
        type=float,
        default=5.0,
        help="Per-candidate edge-write rate ceiling. Default 5.0. 0 disables.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _parse_max_total(raw: str) -> Optional[int]:
    if raw is None:
        return None
    low = raw.strip().lower()
    if low in ("", "unlimited", "none", "all"):
        return None
    try:
        val = int(raw)
    except ValueError as exc:
        raise BackfillError(
            f"--max-total must be an integer or 'unlimited', got {raw!r}"
        ) from exc
    if val < 0:
        raise BackfillError(
            f"--max-total must be >= 0 or 'unlimited', got {val}"
        )
    return val


async def _async_main(args: argparse.Namespace) -> int:
    # Validate the run-id early so reserved-prefix refusal short-circuits
    # before any Neo4j connection attempt.
    try:
        _validate_run_id(args.run_id)
    except BackfillError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    auth = (args.user, args.password) if args.password else None
    driver = AsyncGraphDatabase.driver(args.uri, auth=auth)
    try:
        await driver.verify_connectivity()
    except (neo4j_exceptions.AuthError,) as exc:
        print(
            f"ERROR: Neo4j authentication failed for user '{args.user}' at "
            f"{args.uri}: {exc}",
            file=sys.stderr,
        )
        await driver.close()
        return 1
    except (neo4j_exceptions.ServiceUnavailable, OSError) as exc:
        print(
            f"ERROR: Could not connect to Neo4j at {args.uri}: {exc}",
            file=sys.stderr,
        )
        await driver.close()
        return 1

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        LOGGER.warning("SIGINT received; requesting graceful shutdown")
        shutdown_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except NotImplementedError:
        pass

    try:
        max_total = _parse_max_total(args.max_total)
    except BackfillError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        await driver.close()
        return 1

    try:
        report = await backfill_compaction_edges(
            run_id=args.run_id,
            driver=driver,
            database=args.database,
            max_total=max_total,
            dry_run=args.dry_run,
            rate_limit_qps=args.rate_limit_qps,
            verbose=args.verbose,
            shutdown_event=shutdown_event,
        )
    except BackfillError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        await driver.close()
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Backfill failed unexpectedly: {exc}", file=sys.stderr)
        traceback.print_exc()
        await driver.close()
        return 1
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, ValueError):
            pass

    await driver.close()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    if report.candidates_scanned == 0:
        print(
            "0 candidates found for retroactive compaction backfill — "
            "nothing to do. Compaction data does not yet flow into Fusion "
            "MCP; this script is the backfill rail, ready for when "
            "Phase 5c's cross-repo wiring lands.",
            file=sys.stderr,
        )
    elif report.dry_run:
        print(
            f"DRY RUN — would create {report.edges_created} COMPACTED_FROM "
            f"edges across {report.candidates_scanned} candidates "
            f"({report.candidates_skipped_missing_fields} skipped for "
            f"missing fields).",
            file=sys.stderr,
        )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("Interrupted by user; exiting.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
