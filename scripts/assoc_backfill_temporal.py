"""PLAN-0759 Phase 4 / Step 4.4 — backfill MEMORY_FOLLOWS edges for existing memories.

This script mirrors :mod:`scripts.assoc_backfill_similarity` and
:mod:`scripts.assoc_backfill_entities` for the temporal linker. It walks every
session in the live Neo4j graph, orders the ``:base`` memories in each session
by ``event_seq``, and ``MERGE``s a directed
``(later:base)-[:MEMORY_FOLLOWS]->(earlier:base)`` edge for each adjacent pair.

It is **deliberately separate** from the write-time :class:`TemporalLinker`
so a backfill can use operator-supplied ``run_id`` tagging (rollback safety),
scan the full graph deterministically, and emit a structured
:class:`BackfillReport` instead of fire-and-forget background tasks.

Edge direction
--------------

``MEMORY_FOLLOWS`` is directed (``:base`` → ``:base``) and always points from
the later memory to the earlier one. See
``app/services/associations/temporal_linker.py`` — the write-time linker
constructs the edge as ``(current)-[:MEMORY_FOLLOWS]->(predecessor)`` which
reads as "current memory follows predecessor" in natural English ("A follows
B" means A came after B). The backfill preserves that direction exactly.

Sources of truth (do not duplicate logic — call these directly)
---------------------------------------------------------------

- ``edge_cypher.build_merge_edge_cypher("MEMORY_FOLLOWS")`` — the canonical
  MERGE template with whitelist-validated edge type, MATCH-pinned ``:base``
  endpoints, and weight/last_seen_at refresh semantics.
- ``memory_edges.EDGE_VERSION`` — schema version stamp.
- ``memory_edges.VALID_EDGE_TYPES`` — whitelist (inherited via
  ``build_merge_edge_cypher``).

Coverage gate
-------------

Phase 4 requires ≥50% of ``:base`` memories to carry a ``session_id`` before
the temporal linker can produce meaningful chains. The backfill refuses to
run if the all-time or recent coverage is below the threshold, mirroring
``scripts/assoc_session_coverage_check``. Override with
``--skip-coverage-gate`` for diagnostic runs only.

Usage
-----
Dry-run against a small subset::

    python -m scripts.assoc_backfill_temporal \\
        --run-id phase6-temporal-2026-04-15 --dry-run --max-total 50 --verbose

Full live run::

    python -m scripts.assoc_backfill_temporal \\
        --run-id phase6-temporal-2026-04-15 --rate-limit-qps 20.0

Rollback::

    python -m scripts.assoc_rollback --run-id backfill-phase6-temporal-2026-04-15
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


LOGGER = logging.getLogger("assoc_backfill_temporal")


# ---------------------------------------------------------------------------
# Refusal constants
# ---------------------------------------------------------------------------

WILDCARD_RUN_IDS: Tuple[str, ...] = ("*", "%", "all", "ALL")

#: Sprint test scaffolding prefixes that must NEVER appear in a real backfill
#: run_id. Reused from assoc_backfill_similarity / assoc_backfill_entities.
RESERVED_TEST_PREFIXES: Tuple[str, ...] = (
    "sprint2-",
    "sprint5-",
    "sprint6-",
    "sprint7-",
    "sprint8-",
    "sprint9-",
)

#: Write-time linker prefixes owned by the three wired linkers. Always
#: refused so rollback can cleanly distinguish write-time edges from backfill
#: edges, regardless of which linker family the backfill targets.
WT_TEMPORAL_PREFIX: str = "wt-temporal-"
WT_ENTITY_PREFIX: str = "wt-entity-"
WT_LINK_PREFIX: str = "wt-link-"


#: Phase 4 session_id coverage gate threshold. Matches the default in
#: ``scripts.assoc_session_coverage_check``.
COVERAGE_GATE_THRESHOLD_PCT: float = 50.0

#: Recent-window length in days for the coverage gate. Matches the default
#: in ``scripts.assoc_session_coverage_check``.
COVERAGE_GATE_RECENT_DAYS: int = 30


class BackfillError(RuntimeError):
    """Raised when backfill input is invalid (safety refusal)."""


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class BackfillReport:
    """Structured summary of the temporal backfill run."""

    run_id: str
    dry_run: bool
    started_at: str
    completed_at: Optional[str] = None
    sessions_scanned: int = 0
    sessions_with_edges: int = 0
    sessions_skipped_singleton: int = 0
    memories_in_sessions: int = 0
    memories_skipped_no_event_seq: int = 0
    adjacent_pairs_considered: int = 0
    edges_created: int = 0
    by_session_size: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    checkpoint_final: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dry_run": self.dry_run,
            "sessions_scanned": self.sessions_scanned,
            "sessions_with_edges": self.sessions_with_edges,
            "sessions_skipped_singleton": self.sessions_skipped_singleton,
            "memories_in_sessions": self.memories_in_sessions,
            "memories_skipped_no_event_seq": self.memories_skipped_no_event_seq,
            "adjacent_pairs_considered": self.adjacent_pairs_considered,
            "edges_created": self.edges_created,
            "by_session_size": dict(self.by_session_size),
            "errors": list(self.errors),
            "checkpoint_final": self.checkpoint_final,
        }

    def bump_session_size(self, size: int) -> None:
        # Bucket: 1, 2, 3, 4, 5-9, 10-19, 20-49, 50-99, 100+
        if size <= 0:
            return
        if size <= 4:
            key = str(size)
        elif size <= 9:
            key = "5-9"
        elif size <= 19:
            key = "10-19"
        elif size <= 49:
            key = "20-49"
        elif size <= 99:
            key = "50-99"
        else:
            key = "100+"
        self.by_session_size[key] = self.by_session_size.get(key, 0) + 1


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
    if stripped.startswith(WT_TEMPORAL_PREFIX):
        raise BackfillError(
            f"--run-id='{stripped}' starts with '{WT_TEMPORAL_PREFIX}' which "
            "is reserved for the TemporalLinker write-time path. Pick a "
            "distinct prefix (e.g. 'phase6-temporal-YYYYMMDD')."
        )
    if stripped.startswith(WT_ENTITY_PREFIX):
        raise BackfillError(
            f"--run-id='{stripped}' starts with '{WT_ENTITY_PREFIX}' which is "
            "reserved for the EntityLinker write-time path. Pick a distinct "
            "prefix."
        )
    if stripped.startswith(WT_LINK_PREFIX):
        raise BackfillError(
            f"--run-id='{stripped}' starts with '{WT_LINK_PREFIX}' which is "
            "reserved for the SimilarityLinker write-time path. Pick a "
            "distinct prefix."
        )
    if not allow_test:
        for prefix in RESERVED_TEST_PREFIXES:
            if stripped.startswith(prefix):
                raise BackfillError(
                    f"--run-id='{stripped}' starts with '{prefix}' which is "
                    "reserved for sprint test scaffolding. Pick a distinct "
                    "prefix (e.g. 'phase6-temporal-YYYYMMDD')."
                )
    return stripped


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


def _default_checkpoint_path(run_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    return f"/tmp/assoc_backfill_temporal_{safe}.checkpoint"


def _write_checkpoint_atomic(
    path: str, run_id: str, last_session_id: Optional[str], count: int
) -> None:
    payload = {
        "run_id": run_id,
        "last_processed_session_id": last_session_id,
        "count": count,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _AsyncRateLimiter:
    """Token-spacing rate limiter — qps==0 disables.

    Identical to assoc_backfill_similarity._AsyncRateLimiter; reproduced
    here so this script has no inter-script imports.
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
# Coverage gate
# ---------------------------------------------------------------------------


_COVERAGE_ALL_TIME_CYPHER: str = (
    "MATCH (n:base)\n"
    "RETURN count(n) AS total,\n"
    "       count(n.session_id) AS with_session_id"
)

_COVERAGE_RECENT_CYPHER: str = (
    "MATCH (n:base)\n"
    "WHERE n.event_time > $cutoff_iso\n"
    "RETURN count(n) AS recent_total,\n"
    "       count(n.session_id) AS recent_with_session"
)


async def _check_coverage_gate(
    driver: "AsyncDriver",
    *,
    database: str,
    threshold_pct: float,
    recent_days: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Return ``(passed, report)`` for the Phase 4 coverage gate.

    Mirrors :mod:`scripts.assoc_session_coverage_check` but returns the result
    to the caller instead of exiting the process.
    """
    from datetime import timedelta

    cutoff_iso = (
        datetime.now(tz=timezone.utc) - timedelta(days=recent_days)
    ).isoformat()

    async with driver.session(database=database) as session:
        all_time = await (await session.run(_COVERAGE_ALL_TIME_CYPHER)).single()
        recent = await (
            await session.run(_COVERAGE_RECENT_CYPHER, {"cutoff_iso": cutoff_iso})
        ).single()

    total = int(all_time["total"]) if all_time else 0
    with_sid = int(all_time["with_session_id"]) if all_time else 0
    recent_total = int(recent["recent_total"]) if recent else 0
    recent_with_sid = int(recent["recent_with_session"]) if recent else 0

    def pct(n: int, d: int) -> float:
        return round((n / d) * 100.0, 2) if d > 0 else 0.0

    all_time_pct = pct(with_sid, total)
    recent_pct = pct(recent_with_sid, recent_total)
    passed = all_time_pct >= threshold_pct and recent_pct >= threshold_pct

    return passed, {
        "all_time_total": total,
        "all_time_with_session_id": with_sid,
        "all_time_coverage_pct": all_time_pct,
        "recent_total": recent_total,
        "recent_with_session_id": recent_with_sid,
        "recent_coverage_pct": recent_pct,
        "threshold_pct": threshold_pct,
        "recent_days": recent_days,
        "cutoff_iso": cutoff_iso,
    }


# ---------------------------------------------------------------------------
# Session iteration — cursor-based over distinct session_id values
# ---------------------------------------------------------------------------


async def _iter_sessions(
    driver: "AsyncDriver",
    *,
    database: str,
    project_filter: Optional[str],
    batch_size: int,
) -> "AsyncIterator[str]":
    """Async generator yielding distinct ``session_id`` values in sorted order.

    Paginates deterministically via ``session_id`` ordering so a crashed
    run can resume from the last persisted cursor. Excludes memories with
    a NULL ``session_id`` (the TemporalLinker cannot scope them).
    """
    cursor: str = ""
    while True:
        if project_filter is not None:
            query = (
                "MATCH (n:base) "
                "WHERE n.session_id IS NOT NULL "
                "AND n.session_id > $cursor "
                "AND n.project = $project_filter "
                "WITH DISTINCT n.session_id AS sid "
                "ORDER BY sid "
                "LIMIT $batch_size "
                "RETURN sid"
            )
            params: Dict[str, Any] = {
                "cursor": cursor,
                "project_filter": project_filter,
                "batch_size": batch_size,
            }
        else:
            query = (
                "MATCH (n:base) "
                "WHERE n.session_id IS NOT NULL "
                "AND n.session_id > $cursor "
                "WITH DISTINCT n.session_id AS sid "
                "ORDER BY sid "
                "LIMIT $batch_size "
                "RETURN sid"
            )
            params = {"cursor": cursor, "batch_size": batch_size}

        async with driver.session(database=database) as session:
            result = await session.run(query, params)
            batch = [rec["sid"] async for rec in result]
            await result.consume()

        if not batch:
            return
        for sid in batch:
            yield sid
        cursor = batch[-1]


_SESSION_MEMBERS_CYPHER: str = (
    "MATCH (n:base) "
    "WHERE n.session_id = $session_id "
    "AND n.event_seq IS NOT NULL "
    "AND n.entity_id IS NOT NULL "
    "RETURN n.entity_id AS memory_id, n.event_seq AS event_seq "
    "ORDER BY n.event_seq ASC, n.entity_id ASC"
)


async def _fetch_session_members(
    driver: "AsyncDriver",
    *,
    database: str,
    session_id: str,
) -> List[Tuple[str, int]]:
    """Return the ``(memory_id, event_seq)`` pairs for one session, ordered.

    Memories with a NULL ``event_seq`` are excluded at the query level — we
    cannot order or chain them. The caller tracks the skip count via the
    total-vs-returned delta, computed in the main loop.
    """
    async with driver.session(database=database) as session:
        result = await session.run(
            _SESSION_MEMBERS_CYPHER, {"session_id": session_id}
        )
        rows = [(rec["memory_id"], int(rec["event_seq"])) async for rec in result]
        await result.consume()
    return rows


_SESSION_TOTAL_CYPHER: str = (
    "MATCH (n:base) "
    "WHERE n.session_id = $session_id "
    "RETURN count(n) AS total"
)


async def _count_session_total(
    driver: "AsyncDriver",
    *,
    database: str,
    session_id: str,
) -> int:
    """Return the total number of ``:base`` nodes tagged with this session_id.

    Used to report the skipped-for-null-event_seq delta. Cheap: one COUNT
    query per session.
    """
    async with driver.session(database=database) as session:
        result = await session.run(
            _SESSION_TOTAL_CYPHER, {"session_id": session_id}
        )
        rec = await result.single()
        await result.consume()
    return int(rec["total"]) if rec else 0


# ---------------------------------------------------------------------------
# Core library entry point
# ---------------------------------------------------------------------------


async def backfill_temporal_edges(
    *,
    run_id: str,
    driver: "AsyncDriver",
    database: str = "neo4j",
    batch_size: int = 100,
    max_total: Optional[int] = None,
    dry_run: bool = False,
    rate_limit_qps: float = 20.0,
    resume_from: Optional[str] = None,
    project_filter: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False,
    skip_coverage_gate: bool = False,
    _allow_test_run_id: bool = False,
    shutdown_event: Optional[asyncio.Event] = None,
) -> BackfillReport:
    """Iterate sessions and create ``MEMORY_FOLLOWS`` edges between adjacent memories.

    Parameters
    ----------
    run_id:
        Concrete run identifier. Wildcards / reserved prefixes refused.
    driver:
        Pre-opened async Neo4j driver. Caller owns lifecycle.
    database:
        Neo4j database name. Defaults to ``"neo4j"``.
    batch_size:
        Page size for the distinct-session scan. Default 100.
    max_total:
        Stop after processing this many sessions. ``None`` = unlimited.
    dry_run:
        When True, count the pairs that would be emitted but do NOT write
        any edges.
    rate_limit_qps:
        Upper bound on per-session upsert *batches* per second. Each batch
        issues ``len(members) - 1`` MEMORY_FOLLOWS upserts.
        Default 20.0. 0 disables.
    resume_from:
        Skip all sessions up to and including this ``session_id``.
    project_filter:
        Only process memories where ``n.project = $filter``.
    checkpoint_path:
        Path to write the cursor file every 100 processed sessions.
    verbose:
        Enable DEBUG-level logging.
    skip_coverage_gate:
        Diagnostic override — run even if the session_id coverage is below
        the Phase 4 threshold.
    _allow_test_run_id:
        Bypass test-prefix refusal. Only used by integration tests.
    shutdown_event:
        Optional ``asyncio.Event``. If set mid-run, the loop finishes the
        current session and exits cleanly. Wired to SIGINT in the CLI.
    """
    # Late imports keep the module top-level I/O free.
    from app.services.associations.edge_cypher import build_merge_edge_cypher
    from app.services.associations.memory_edges import EDGE_VERSION

    if verbose:
        LOGGER.setLevel(logging.DEBUG)

    normalized_run_id = _validate_run_id(run_id, allow_test=_allow_test_run_id)
    tagged_run_id = f"backfill-{normalized_run_id}"

    effective_checkpoint_path = checkpoint_path or _default_checkpoint_path(
        normalized_run_id
    )

    report = BackfillReport(
        run_id=normalized_run_id,
        dry_run=dry_run,
        started_at=datetime.now(tz=timezone.utc).isoformat(),
        checkpoint_final=effective_checkpoint_path,
    )

    # Phase 4 coverage gate — refuse if session_id coverage is too low to
    # justify running. Allow override for diagnostic runs.
    if not skip_coverage_gate:
        passed, coverage = await _check_coverage_gate(
            driver,
            database=database,
            threshold_pct=COVERAGE_GATE_THRESHOLD_PCT,
            recent_days=COVERAGE_GATE_RECENT_DAYS,
        )
        LOGGER.info(
            "backfill.coverage_gate all_time=%.2f%% recent=%.2f%% "
            "threshold=%.2f%% passed=%s",
            coverage["all_time_coverage_pct"],
            coverage["recent_coverage_pct"],
            coverage["threshold_pct"],
            passed,
        )
        if not passed:
            raise BackfillError(
                f"Phase 4 session_id coverage gate failed: "
                f"all_time={coverage['all_time_coverage_pct']}% "
                f"recent={coverage['recent_coverage_pct']}% "
                f"threshold={coverage['threshold_pct']}%. "
                "Run `python -m scripts.assoc_session_coverage_check` for a "
                "full diagnostic report, or pass --skip-coverage-gate to "
                "override for a diagnostic run."
            )

    # Build the MERGE template exactly once. The whitelist check inside
    # ``build_merge_edge_cypher`` is the only place the edge type is
    # validated before f-string interpolation.
    merge_cypher = build_merge_edge_cypher("MEMORY_FOLLOWS")

    rate_limiter = _AsyncRateLimiter(rate_limit_qps)

    LOGGER.info(
        "backfill.start run_id=%s tagged=%s dry_run=%s batch_size=%d "
        "max_total=%s rate_limit_qps=%s project_filter=%s resume_from=%s",
        normalized_run_id,
        tagged_run_id,
        dry_run,
        batch_size,
        max_total,
        rate_limit_qps,
        project_filter,
        resume_from,
    )

    processed_counter = 0
    cursor_passed_resume = resume_from is None
    last_session_id: Optional[str] = None

    try:
        async for session_id in _iter_sessions(
            driver=driver,
            database=database,
            project_filter=project_filter,
            batch_size=batch_size,
        ):
            if shutdown_event is not None and shutdown_event.is_set():
                LOGGER.warning(
                    "backfill.shutdown_requested session_id=%s processed=%d",
                    session_id,
                    processed_counter,
                )
                break

            report.sessions_scanned += 1

            if not cursor_passed_resume:
                if session_id == resume_from:
                    cursor_passed_resume = True
                continue

            if max_total is not None and processed_counter >= max_total:
                LOGGER.info(
                    "backfill.max_total_reached processed=%d max_total=%d",
                    processed_counter,
                    max_total,
                )
                break

            # Fetch every member of this session with a usable event_seq.
            try:
                members = await _fetch_session_members(
                    driver, database=database, session_id=session_id
                )
            except Exception as exc:  # noqa: BLE001 — fail-open per session
                report.errors.append(
                    f"session_id={session_id}: fetch_members: "
                    f"{type(exc).__name__}: {exc}"
                )
                LOGGER.warning(
                    "backfill.fetch_members_failed session_id=%s error=%s",
                    session_id,
                    exc,
                )
                last_session_id = session_id
                continue

            # Count the null-event_seq delta. Cheap extra query per session.
            try:
                total_in_session = await _count_session_total(
                    driver, database=database, session_id=session_id
                )
            except Exception as exc:  # noqa: BLE001
                # Non-fatal: we can still emit edges for the members we have.
                LOGGER.debug(
                    "backfill.count_failed session_id=%s error=%s",
                    session_id,
                    exc,
                )
                total_in_session = len(members)

            report.memories_in_sessions += len(members)
            report.memories_skipped_no_event_seq += max(
                0, total_in_session - len(members)
            )
            report.bump_session_size(len(members))

            if len(members) < 2:
                report.sessions_skipped_singleton += 1
                LOGGER.debug(
                    "backfill.skip_singleton session_id=%s members=%d",
                    session_id,
                    len(members),
                )
                last_session_id = session_id
                processed_counter += 1
                continue

            # Dedup adjacent duplicate (memory_id, event_seq) tuples just in
            # case — the ORDER BY would otherwise be non-strict. Self-loops
            # are rejected by the MemoryEdge dataclass contract and would
            # fall out here anyway, but belt-and-braces.
            pairs: List[Tuple[str, str]] = []
            for i in range(1, len(members)):
                earlier_id, _ = members[i - 1]
                later_id, _ = members[i]
                if later_id == earlier_id:
                    continue
                pairs.append((later_id, earlier_id))

            report.adjacent_pairs_considered += len(pairs)

            if not pairs:
                LOGGER.debug(
                    "backfill.skip_no_pairs session_id=%s", session_id
                )
                last_session_id = session_id
                processed_counter += 1
                continue

            # Rate limit at the per-session batch granularity so operators
            # think in "sessions per second" just like the entities backfill
            # uses "memories per second".
            await rate_limiter.wait()

            edges_this_session = 0
            now_iso = datetime.now(tz=timezone.utc).isoformat()

            if dry_run:
                # In dry-run, count every pair as if it would land. MERGE
                # idempotency means re-runs would land 0 new edges, so this
                # upper-bounds the first-run impact.
                edges_this_session = len(pairs)
                report.edges_created += edges_this_session
                if edges_this_session > 0:
                    report.sessions_with_edges += 1
                processed_counter += 1
                last_session_id = session_id
            else:
                try:
                    async with driver.session(database=database) as session:
                        for later_id, earlier_id in pairs:
                            params = {
                                "src_id": later_id,
                                "dst_id": earlier_id,
                                "weight": 1.0,
                                "created_at": now_iso,
                                "last_seen_at": now_iso,
                                "created_by": "assoc_backfill_temporal",
                                "run_id": tagged_run_id,
                                "edge_version": EDGE_VERSION,
                                # Neo4j 5 rejects Map-valued relationship
                                # properties. Attribution is fully captured
                                # by created_by + run_id; event_seq is
                                # derivable from the endpoint nodes.
                                "metadata": None,
                            }
                            result = await session.run(merge_cypher, params)
                            records = [rec async for rec in result]
                            await result.consume()
                            if records:
                                edges_this_session += 1
                except Exception as exc:  # noqa: BLE001 — fail-open per session
                    report.errors.append(
                        f"session_id={session_id}: upsert: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    LOGGER.warning(
                        "backfill.upsert_failed session_id=%s error=%s",
                        session_id,
                        exc,
                    )
                    last_session_id = session_id
                    processed_counter += 1
                    continue

                report.edges_created += edges_this_session
                if edges_this_session > 0:
                    report.sessions_with_edges += 1
                processed_counter += 1
                last_session_id = session_id

            LOGGER.debug(
                "backfill.session_done session_id=%s members=%d pairs=%d "
                "edges=%d",
                session_id,
                len(members),
                len(pairs),
                edges_this_session,
            )

            if processed_counter % 100 == 0:
                _write_checkpoint_atomic(
                    effective_checkpoint_path,
                    normalized_run_id,
                    last_session_id,
                    processed_counter,
                )
                LOGGER.info(
                    "backfill.checkpoint path=%s count=%d last=%s",
                    effective_checkpoint_path,
                    processed_counter,
                    last_session_id,
                )
    except asyncio.CancelledError:
        LOGGER.warning("backfill.cancelled processed=%d", processed_counter)
        raise
    finally:
        try:
            _write_checkpoint_atomic(
                effective_checkpoint_path,
                normalized_run_id,
                last_session_id,
                processed_counter,
            )
        except Exception as exc:  # noqa: BLE001 — don't mask original
            LOGGER.warning(
                "backfill.final_checkpoint_failed path=%s error=%s",
                effective_checkpoint_path,
                exc,
            )
        report.completed_at = datetime.now(tz=timezone.utc).isoformat()

    LOGGER.info(
        "backfill.done run_id=%s sessions_scanned=%d sessions_with_edges=%d "
        "sessions_singleton=%d memories=%d pairs=%d edges=%d errors=%d",
        normalized_run_id,
        report.sessions_scanned,
        report.sessions_with_edges,
        report.sessions_skipped_singleton,
        report.memories_in_sessions,
        report.adjacent_pairs_considered,
        report.edges_created,
        len(report.errors),
    )
    return report


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assoc_backfill_temporal",
        description=(
            "PLAN-0759 Phase 4 / Step 4.4 — retroactively create "
            "MEMORY_FOLLOWS edges between adjacent :base memories within "
            "each session, mirroring Sprint 10's TemporalLinker semantics."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Concrete run identifier. Wildcards and reserved prefixes "
            "(wt-temporal-, wt-entity-, wt-link-, sprint{2,5,6,7,8,9}-) "
            "are refused."
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
        "--batch-size",
        type=int,
        default=100,
        help="Session pagination size. Default 100.",
    )
    parser.add_argument(
        "--max-total",
        type=str,
        default="unlimited",
        help="Max sessions to process. Integer or 'unlimited'. Default unlimited.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count pairs without writing any edges.",
    )
    parser.add_argument(
        "--rate-limit-qps",
        type=float,
        default=20.0,
        help="Per-session upsert batch rate ceiling. Default 20.0. 0 disables.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="session_id cursor. Skip sessions up to and including this id.",
    )
    parser.add_argument(
        "--project-filter",
        default=None,
        help="Only process memories where n.project = <value>.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to write cursor checkpoint every 100 sessions.",
    )
    parser.add_argument(
        "--skip-coverage-gate",
        action="store_true",
        help=(
            "Diagnostic override — run even if session_id coverage is below "
            "the Phase 4 threshold. Use with caution."
        ),
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
        return 2
    except (neo4j_exceptions.ServiceUnavailable, OSError) as exc:
        print(
            f"ERROR: Could not connect to Neo4j at {args.uri}: {exc}",
            file=sys.stderr,
        )
        await driver.close()
        return 2

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
        return 2

    try:
        report = await backfill_temporal_edges(
            run_id=args.run_id,
            driver=driver,
            database=args.database,
            batch_size=args.batch_size,
            max_total=max_total,
            dry_run=args.dry_run,
            rate_limit_qps=args.rate_limit_qps,
            resume_from=args.resume_from,
            project_filter=args.project_filter,
            checkpoint_path=args.checkpoint_path,
            verbose=args.verbose,
            skip_coverage_gate=args.skip_coverage_gate,
            shutdown_event=shutdown_event,
        )
    except BackfillError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        await driver.close()
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Backfill failed unexpectedly: {exc}", file=sys.stderr)
        traceback.print_exc()
        await driver.close()
        return 3
    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, ValueError):
            pass

    await driver.close()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    if report.dry_run:
        print(
            f"DRY RUN — would create {report.edges_created} MEMORY_FOLLOWS "
            f"edges across {report.sessions_with_edges} sessions "
            f"({report.memories_in_sessions} memories, "
            f"{report.adjacent_pairs_considered} adjacent pairs, "
            f"{report.sessions_skipped_singleton} singleton sessions skipped).",
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
