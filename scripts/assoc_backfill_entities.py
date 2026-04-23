"""PLAN-0759 Phase 3 / Step 3.4 — backfill MENTIONS edges for existing memories.

This script mirrors :mod:`scripts.assoc_backfill_similarity` for the entity
linker. It walks every existing ``:base`` memory in Neo4j, runs the same
heuristic entity extractor that
:class:`app.services.associations.entity_linker.EntityLinker` would have
called at write-time, and ``MERGE``s ``(:Entity)`` nodes plus
``(:base)-[:MENTIONS]->(:Entity)`` edges retroactively.

It is **deliberately separate** from the write-time linker so a backfill
can use operator-supplied ``run_id`` tagging (rollback safety) and emit
a deterministic ``BackfillReport`` instead of fire-and-forget background
tasks.

Sources of truth (do not duplicate logic — call these directly)
---------------------------------------------------------------

- ``entity_extractor.extract_entities`` — Tier B heuristic extractor
- ``entity_extractor.canon_entity``     — canonicalization pipeline
- ``entity_extractor.MAX_ENTITIES_PER_MEMORY`` — fan-out cap
- ``memory_edges.EDGE_VERSION``         — schema version stamp

The MENTIONS upsert Cypher is *re-declared* here (not imported) because
:mod:`entity_linker` keeps it as a module-private constant. The shape is
identical and tracked alongside the source-of-truth in entity_linker.py;
any change there must be mirrored here. The shared template is small
enough that duplication is the lesser evil over reaching into private
internals.

Tier A (caller-provided ``metadata["entities"]``) is **not honored** by
this backfill. It only exists for the write-time path where a memory
ingestion may carry pre-extracted entities; backfill operates over
already-stored memories and reads ``text`` from Neo4j directly. If a
backfill needs to honor pre-extracted entities, those would need to live
on the ``:base`` node as a property — which they do not, today.

Usage
-----
Dry-run against a small subset::

    python -m scripts.assoc_backfill_entities \\
        --run-id phase6-eval-mentions-2026-04-15 --dry-run --max-total 50 \\
        --verbose

Full live run::

    python -m scripts.assoc_backfill_entities \\
        --run-id phase6-eval-mentions-2026-04-15 --rate-limit-qps 20.0

Rollback::

    python -m scripts.assoc_rollback --run-id backfill-phase6-eval-mentions-2026-04-15
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
    from app.observability.metrics import (
        record_backfill_record,
        set_backfill_progress,
    )
except ImportError:  # pragma: no cover — script may be run from non-package root
    def set_backfill_progress(script: str, phase: str, value: float) -> None:
        return None

    def record_backfill_record(script: str) -> None:
        return None

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


LOGGER = logging.getLogger("assoc_backfill_entities")


# ---------------------------------------------------------------------------
# Refusal constants
# ---------------------------------------------------------------------------

WILDCARD_RUN_IDS: Tuple[str, ...] = ("*", "%", "all", "ALL")

#: Sprint test scaffolding prefixes that must NEVER appear in a real backfill
#: run_id. Reused from assoc_backfill_similarity for consistency.
RESERVED_TEST_PREFIXES: Tuple[str, ...] = (
    "sprint2-",
    "sprint5-",
    "sprint6-",
    "sprint7-",
    "sprint8-",
    "sprint9-",
)

#: Write-time linker prefix owned by Sprint 9's EntityLinker. Always refused
#: so rollback can cleanly distinguish write-time-linker edges from backfill
#: edges. Also refuse the similarity write-time prefix as a defense in depth.
WT_ENTITY_PREFIX: str = "wt-entity-"
WT_LINK_PREFIX: str = "wt-link-"


class BackfillError(RuntimeError):
    """Raised when backfill input is invalid (safety refusal)."""


# ---------------------------------------------------------------------------
# Cypher templates
# ---------------------------------------------------------------------------

# Combined upsert: entity-node + mentions-edge in one round-trip per
# (memory_id, canon_name). Mirrors the constant in
# app/services/associations/entity_linker.py — see its module docstring
# for the full design rationale (mixed :base/:Entity endpoints, why edge
# service can't be reused, why metadata is literally None).
_MENTIONS_UPSERT_CYPHER: str = (
    "MATCH (m:base {entity_id: $memory_id})\n"
    "MERGE (e:Entity {project: $project, name: $canon_name})\n"
    "  ON CREATE SET e.created_at = $now\n"
    "MERGE (m)-[r:MENTIONS]->(e)\n"
    "  ON CREATE SET\n"
    "    r.weight = $weight,\n"
    "    r.created_at = $now,\n"
    "    r.last_seen_at = $now,\n"
    "    r.created_by = $created_by,\n"
    "    r.run_id = $run_id,\n"
    "    r.edge_version = $edge_version,\n"
    "    r.metadata = $metadata\n"
    "  ON MATCH SET\n"
    "    r.last_seen_at = $now\n"
    "RETURN e.name AS name"
)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class BackfillReport:
    """Structured summary of the backfill run."""

    run_id: str
    dry_run: bool
    started_at: str
    completed_at: Optional[str] = None
    memories_scanned: int = 0
    memories_processed: int = 0
    memories_skipped_no_text: int = 0
    memories_skipped_no_project: int = 0
    memories_skipped_no_entities: int = 0
    entities_extracted_total: int = 0
    edges_created: int = 0
    by_project: Dict[str, Dict[str, int]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    checkpoint_final: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dry_run": self.dry_run,
            "memories_scanned": self.memories_scanned,
            "memories_processed": self.memories_processed,
            "memories_skipped_no_text": self.memories_skipped_no_text,
            "memories_skipped_no_project": self.memories_skipped_no_project,
            "memories_skipped_no_entities": self.memories_skipped_no_entities,
            "entities_extracted_total": self.entities_extracted_total,
            "edges_created": self.edges_created,
            "by_project": {
                proj: dict(counts) for proj, counts in self.by_project.items()
            },
            "errors": list(self.errors),
            "checkpoint_final": self.checkpoint_final,
        }

    def bump_project(
        self, project: str, processed: int = 0, edges: int = 0, entities: int = 0
    ) -> None:
        bucket = self.by_project.setdefault(
            project, {"processed": 0, "edges": 0, "entities": 0}
        )
        bucket["processed"] += processed
        bucket["edges"] += edges
        bucket["entities"] += entities


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
    if stripped.startswith(WT_ENTITY_PREFIX):
        raise BackfillError(
            f"--run-id='{stripped}' starts with '{WT_ENTITY_PREFIX}' which is "
            "reserved for the EntityLinker write-time path. Pick a distinct "
            "prefix (e.g. 'phase6-eval-mentions-YYYYMMDD')."
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
                    "prefix (e.g. 'phase6-eval-mentions-YYYYMMDD')."
                )
    return stripped


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


def _default_checkpoint_path(run_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    return f"/tmp/assoc_backfill_entities_{safe}.checkpoint"


def _write_checkpoint_atomic(
    path: str, run_id: str, last_memory_id: Optional[str], count: int
) -> None:
    payload = {
        "run_id": run_id,
        "last_processed_memory_id": last_memory_id,
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
# Memory iteration — cursor-based over :base by entity_id
# ---------------------------------------------------------------------------


async def _iter_memories_with_text(
    driver: "AsyncDriver",
    *,
    database: str,
    project_filter: Optional[str],
    batch_size: int,
) -> "AsyncIterator[Tuple[str, Optional[str], Optional[str]]]":
    """Async generator yielding ``(entity_id, project, text)`` tuples.

    Paginates deterministically via ``entity_id`` ordering so a crashed
    run can resume from the last persisted cursor. Yields the text field
    too because the entity extractor needs the raw memory content.
    """
    cursor: str = ""
    while True:
        if project_filter is not None:
            query = (
                "MATCH (n:base) "
                "WHERE n.entity_id IS NOT NULL "
                "AND n.entity_id > $cursor "
                "AND n.project = $project_filter "
                "RETURN n.entity_id AS memory_id, n.project AS project, "
                "       n.text AS text "
                "ORDER BY n.entity_id "
                "LIMIT $batch_size"
            )
            params: Dict[str, Any] = {
                "cursor": cursor,
                "project_filter": project_filter,
                "batch_size": batch_size,
            }
        else:
            query = (
                "MATCH (n:base) "
                "WHERE n.entity_id IS NOT NULL "
                "AND n.entity_id > $cursor "
                "RETURN n.entity_id AS memory_id, n.project AS project, "
                "       n.text AS text "
                "ORDER BY n.entity_id "
                "LIMIT $batch_size"
            )
            params = {"cursor": cursor, "batch_size": batch_size}

        async with driver.session(database=database) as session:
            result = await session.run(query, params)
            batch = [dict(rec) async for rec in result]
            await result.consume()

        if not batch:
            return
        for row in batch:
            yield (row["memory_id"], row.get("project"), row.get("text"))
        cursor = batch[-1]["memory_id"]


# ---------------------------------------------------------------------------
# Core library entry point
# ---------------------------------------------------------------------------


async def backfill_mentions_edges(
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
    _allow_test_run_id: bool = False,
    shutdown_event: Optional[asyncio.Event] = None,
) -> BackfillReport:
    """Iterate existing ``:base`` memories and create ``MENTIONS`` edges.

    Parameters
    ----------
    run_id:
        Concrete run identifier. Wildcards / reserved prefixes refused.
    driver:
        Pre-opened async Neo4j driver. Caller owns lifecycle.
    database:
        Neo4j database name. Defaults to ``"neo4j"``.
    batch_size:
        Page size for the paginated memory scan. Default 100.
    max_total:
        Stop after processing this many memories. ``None`` = unlimited.
    dry_run:
        When True, run extraction + canonicalization but do NOT write any
        edges. Counts what would have been created.
    rate_limit_qps:
        Upper bound on per-memory upsert *batches* per second. Each batch
        issues up to ``MAX_ENTITIES_PER_MEMORY=20`` MENTIONS upserts.
        Default 20.0.
    resume_from:
        Skip all memories up to and including this ``entity_id``.
    project_filter:
        Only process memories where ``n.project = $filter``.
    checkpoint_path:
        Path to write the cursor file every 100 processed memories.
    verbose:
        Enable DEBUG-level logging.
    _allow_test_run_id:
        Bypass test-prefix refusal. Only used by integration tests.
    shutdown_event:
        Optional ``asyncio.Event``. If set mid-run, the loop finishes the
        current memory and exits cleanly. Wired to SIGINT in the CLI.
    """
    # Late imports keep the module top-level I/O free.
    from app.services.associations.entity_extractor import (
        MAX_ENTITIES_PER_MEMORY,
        canon_entity,
        extract_entities,
    )
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
    last_memory_id: Optional[str] = None

    try:
        async for memory_id, project, text in _iter_memories_with_text(
            driver=driver,
            database=database,
            project_filter=project_filter,
            batch_size=batch_size,
        ):
            if shutdown_event is not None and shutdown_event.is_set():
                LOGGER.warning(
                    "backfill.shutdown_requested memory_id=%s processed=%d",
                    memory_id,
                    processed_counter,
                )
                break

            report.memories_scanned += 1

            if not cursor_passed_resume:
                if memory_id == resume_from:
                    cursor_passed_resume = True
                continue

            if max_total is not None and processed_counter >= max_total:
                LOGGER.info(
                    "backfill.max_total_reached processed=%d max_total=%d",
                    processed_counter,
                    max_total,
                )
                break

            # Project scoping: EntityLinker refuses None project because
            # (project, name) is the Entity node primary key. Track and
            # skip the same way.
            if project is None:
                report.memories_skipped_no_project += 1
                LOGGER.debug(
                    "backfill.skip_no_project memory_id=%s", memory_id
                )
                last_memory_id = memory_id
                continue

            if not isinstance(text, str) or not text:
                report.memories_skipped_no_text += 1
                LOGGER.debug(
                    "backfill.skip_no_text memory_id=%s", memory_id
                )
                last_memory_id = memory_id
                continue

            # Run the same Tier B heuristic the EntityLinker would have
            # called at write time.
            try:
                raw_entities = extract_entities(
                    text, max_entities=MAX_ENTITIES_PER_MEMORY
                )
            except Exception as exc:  # noqa: BLE001 — fail-open per memory
                report.errors.append(
                    f"memory_id={memory_id}: extract_entities: "
                    f"{type(exc).__name__}: {exc}"
                )
                LOGGER.warning(
                    "backfill.extract_failed memory_id=%s error=%s",
                    memory_id,
                    exc,
                )
                last_memory_id = memory_id
                continue

            # Canonicalize + dedup. Preserve first occurrence so re-runs
            # produce a deterministic write order.
            canon_seen: Dict[str, str] = {}
            for raw in raw_entities:
                try:
                    canon = canon_entity(raw)
                except (ValueError, TypeError):
                    continue
                if canon not in canon_seen:
                    canon_seen[canon] = raw

            canonical_names = list(canon_seen.keys())[:MAX_ENTITIES_PER_MEMORY]

            if not canonical_names:
                report.memories_skipped_no_entities += 1
                LOGGER.debug(
                    "backfill.skip_no_entities memory_id=%s", memory_id
                )
                last_memory_id = memory_id
                processed_counter += 1
                report.memories_processed += 1
                report.bump_project(project, processed=1, edges=0, entities=0)
                continue

            report.entities_extracted_total += len(canonical_names)
            now_iso = datetime.now(tz=timezone.utc).isoformat()

            # Rate limit at the per-memory upsert batch granularity rather
            # than per-edge — one memory does up to 20 writes in one
            # session, and we want operators to think in "memories per
            # second" not "edges per second".
            await rate_limiter.wait()

            edges_for_this_memory = 0

            if dry_run:
                # In dry-run we still count what would be written, project-
                # tracked the same way as the live path.
                edges_for_this_memory = len(canonical_names)
                report.edges_created += edges_for_this_memory
                report.bump_project(
                    project,
                    processed=1,
                    edges=edges_for_this_memory,
                    entities=len(canonical_names),
                )
                processed_counter += 1
                report.memories_processed += 1
                last_memory_id = memory_id
            else:
                try:
                    async with driver.session(database=database) as session:
                        for canon_name in canonical_names:
                            params = {
                                "memory_id": memory_id,
                                "project": project,
                                "canon_name": canon_name,
                                "now": now_iso,
                                "weight": 1.0,
                                "created_by": "assoc_backfill_entities",
                                "run_id": tagged_run_id,
                                "edge_version": EDGE_VERSION,
                                "metadata": None,
                            }
                            result = await session.run(
                                _MENTIONS_UPSERT_CYPHER, params
                            )
                            records = [rec async for rec in result]
                            await result.consume()
                            if records:
                                edges_for_this_memory += 1
                except Exception as exc:  # noqa: BLE001 — fail-open per memory
                    report.errors.append(
                        f"memory_id={memory_id}: upsert: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    LOGGER.warning(
                        "backfill.upsert_failed memory_id=%s error=%s",
                        memory_id,
                        exc,
                    )
                    last_memory_id = memory_id
                    processed_counter += 1
                    report.memories_processed += 1
                    report.bump_project(project, processed=1, edges=0)
                    continue

                report.edges_created += edges_for_this_memory
                report.bump_project(
                    project,
                    processed=1,
                    edges=edges_for_this_memory,
                    entities=len(canonical_names),
                )
                processed_counter += 1
                report.memories_processed += 1
                last_memory_id = memory_id

            if processed_counter % 100 == 0:
                # Emit liveness counter unconditionally — batch-granular,
                # independent of max_total so full-graph runs still show
                # monotonic progress via rate(...[1m]).
                record_backfill_record("assoc_backfill_entities")
                if max_total and max_total > 0:
                    set_backfill_progress(
                        "assoc_backfill_entities",
                        "processing",
                        min(1.0, processed_counter / float(max_total)),
                    )
                _write_checkpoint_atomic(
                    effective_checkpoint_path,
                    normalized_run_id,
                    last_memory_id,
                    processed_counter,
                )
                LOGGER.info(
                    "backfill.checkpoint path=%s count=%d last=%s",
                    effective_checkpoint_path,
                    processed_counter,
                    last_memory_id,
                )
    except asyncio.CancelledError:
        LOGGER.warning("backfill.cancelled processed=%d", processed_counter)
        raise
    finally:
        try:
            _write_checkpoint_atomic(
                effective_checkpoint_path,
                normalized_run_id,
                last_memory_id,
                processed_counter,
            )
        except Exception as exc:  # noqa: BLE001 — don't mask original
            LOGGER.warning(
                "backfill.final_checkpoint_failed path=%s error=%s",
                effective_checkpoint_path,
                exc,
            )
        report.completed_at = datetime.now(tz=timezone.utc).isoformat()
        set_backfill_progress("assoc_backfill_entities", "processing", 1.0)

    LOGGER.info(
        "backfill.done run_id=%s scanned=%d processed=%d entities=%d "
        "edges=%d skipped_no_text=%d skipped_no_project=%d "
        "skipped_no_entities=%d errors=%d",
        normalized_run_id,
        report.memories_scanned,
        report.memories_processed,
        report.entities_extracted_total,
        report.edges_created,
        report.memories_skipped_no_text,
        report.memories_skipped_no_project,
        report.memories_skipped_no_entities,
        len(report.errors),
    )
    return report


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assoc_backfill_entities",
        description=(
            "PLAN-0759 Phase 3 / Step 3.4 — retroactively create MENTIONS "
            "edges for existing :base memories using Sprint 8's heuristic "
            "extractor and Sprint 9's upsert Cypher."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Concrete run identifier. Wildcards and reserved prefixes "
            "(wt-entity-, wt-link-, sprint{2,5,6,7,8,9}-) are refused."
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
        help="Memory pagination size. Default 100.",
    )
    parser.add_argument(
        "--max-total",
        type=str,
        default="unlimited",
        help="Max memories to process. Integer or 'unlimited'. Default unlimited.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction without writing any edges. Prints counts.",
    )
    parser.add_argument(
        "--rate-limit-qps",
        type=float,
        default=20.0,
        help="Per-memory upsert batch rate ceiling. Default 20.0. 0 disables.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="entity_id cursor. Skip memories up to and including this id.",
    )
    parser.add_argument(
        "--project-filter",
        default=None,
        help="Only process memories where n.project = <value>.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to write cursor checkpoint every 100 memories.",
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
        report = await backfill_mentions_edges(
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
            f"DRY RUN — would create {report.edges_created} MENTIONS edges "
            f"({report.entities_extracted_total} canonical entities) across "
            f"{report.memories_processed} memories in "
            f"{len(report.by_project)} projects.",
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
