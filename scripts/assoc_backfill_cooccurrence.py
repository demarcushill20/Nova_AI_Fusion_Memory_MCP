"""PLAN-0759 Phase 7a — backfill CO_OCCURS edges between memories sharing entities.

This script mirrors :mod:`scripts.assoc_backfill_similarity`,
:mod:`scripts.assoc_backfill_entities` and
:mod:`scripts.assoc_backfill_temporal` for the co-occurrence linker. It walks
every existing ``:base`` memory in Neo4j, finds memories that share ≥ 2
non-hub entities with it, and ``MERGE``s a single
``CO_OCCURS`` edge per canonical pair using an IDF-style weight.

It is **deliberately separate** from the write-time
:class:`CooccurrenceLinker` so a backfill can use operator-supplied
``run_id`` tagging (rollback safety), bulk-scan the graph efficiently,
canonicalize pairs for the bidirectional edge type, and emit a structured
:class:`BackfillReport` instead of fire-and-forget background tasks.

Semantic parity with the write-time linker
-------------------------------------------

Every constant and threshold mirrors ``cooccurrence_linker.py``:

- ``HUB_THRESHOLD = 50`` — entities mentioned by more than this many
  memories are skipped entirely.
- ``MAX_EDGES_PER_ENTITY = 30`` — at most this many co-mentioners are
  evaluated per entity (bounds fan-out on popular-but-non-hub entities).
- ``MIN_SHARED_ENTITIES = 2`` — a pair must share at least two non-hub
  entities before an edge is created.
- IDF weight: ``idf = log(max(max_degree / degree, 1.0))`` summed across
  shared entities, normalized via ``min(1.0, idf_sum / 10.0)``, floored
  at ``0.01`` so any qualifying pair gets a positive weight.

Bidirectional canonicalization (backfill-only correctness)
-----------------------------------------------------------

``CO_OCCURS`` is in :data:`memory_edges.BIDIRECTIONAL_EDGE_TYPES`. The
write-time linker fires from one endpoint only (the memory just stored),
so it emits at most one stored edge per (new, old) pair even if the same
pair would be symmetric. The backfill scans every memory, so without
canonicalization it would re-evaluate each pair from both endpoints and
produce two directed edges for the same logical fact. The backfill
therefore:

1. Computes the canonical (source, target) via
   :meth:`MemoryEdge.canonicalize_for_bidirectional` before emitting.
2. Tracks emitted pairs in an in-memory ``seen`` set keyed on the
   canonical tuple so the second pass of each pair is a no-op.

This is a **correctness improvement** over the write-time linker which
relies on ``MERGE`` + ordering luck to de-duplicate. The backfill's
canonicalized output is faithful to the linker's semantics (same weight
formula, same shared-entity rules) but avoids the double-edge hazard.

Usage
-----
Dry-run against a small subset::

    python -m scripts.assoc_backfill_cooccurrence \\
        --run-id phase6-cooccur-2026-04-15 --dry-run --max-total 50 --verbose

Full live run::

    python -m scripts.assoc_backfill_cooccurrence \\
        --run-id phase6-cooccur-2026-04-15 --rate-limit-qps 50.0

Rollback::

    python -m scripts.assoc_rollback --run-id backfill-phase6-cooccur-2026-04-15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
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


LOGGER = logging.getLogger("assoc_backfill_cooccurrence")


# ---------------------------------------------------------------------------
# Constants — mirror app/services/associations/cooccurrence_linker.py
# ---------------------------------------------------------------------------

HUB_THRESHOLD: int = 50
MAX_EDGES_PER_ENTITY: int = 30
MIN_SHARED_ENTITIES: int = 2

DEFAULT_WALLCLOCK_CEILING_SEC: float = 120.0


# ---------------------------------------------------------------------------
# Refusal constants
# ---------------------------------------------------------------------------

WILDCARD_RUN_IDS: Tuple[str, ...] = ("*", "%", "all", "ALL")

RESERVED_TEST_PREFIXES: Tuple[str, ...] = (
    "sprint2-",
    "sprint5-",
    "sprint6-",
    "sprint7-",
    "sprint8-",
    "sprint9-",
)

WT_COOCCUR_PREFIX: str = "wt-cooccur-"
WT_TEMPORAL_PREFIX: str = "wt-temporal-"
WT_ENTITY_PREFIX: str = "wt-entity-"
WT_LINK_PREFIX: str = "wt-link-"


class BackfillError(RuntimeError):
    """Raised when backfill input is invalid (safety refusal)."""


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class BackfillReport:
    """Structured summary of the co-occurrence backfill run."""

    run_id: str
    dry_run: bool
    started_at: str
    completed_at: Optional[str] = None
    memories_scanned: int = 0
    memories_processed: int = 0
    memories_skipped_no_project: int = 0
    memories_skipped_no_entities: int = 0
    memories_with_all_hub_entities: int = 0
    entities_evaluated: int = 0
    entities_hub_skipped: int = 0
    pairs_evaluated: int = 0
    pairs_below_min_shared: int = 0
    pairs_emitted: int = 0
    edges_created: int = 0
    edges_deduped_canonical: int = 0
    walltime_ceiling_hit: bool = False
    errors: List[str] = field(default_factory=list)
    by_project: Dict[str, Dict[str, int]] = field(default_factory=dict)
    checkpoint_final: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "dry_run": self.dry_run,
            "memories_scanned": self.memories_scanned,
            "memories_processed": self.memories_processed,
            "memories_skipped_no_project": self.memories_skipped_no_project,
            "memories_skipped_no_entities": self.memories_skipped_no_entities,
            "memories_with_all_hub_entities": self.memories_with_all_hub_entities,
            "entities_evaluated": self.entities_evaluated,
            "entities_hub_skipped": self.entities_hub_skipped,
            "pairs_evaluated": self.pairs_evaluated,
            "pairs_below_min_shared": self.pairs_below_min_shared,
            "pairs_emitted": self.pairs_emitted,
            "edges_created": self.edges_created,
            "edges_deduped_canonical": self.edges_deduped_canonical,
            "walltime_ceiling_hit": self.walltime_ceiling_hit,
            "errors": list(self.errors),
            "by_project": {
                proj: dict(counts) for proj, counts in self.by_project.items()
            },
            "checkpoint_final": self.checkpoint_final,
        }

    def bump_project(
        self,
        project: str,
        processed: int = 0,
        edges: int = 0,
        pairs: int = 0,
    ) -> None:
        bucket = self.by_project.setdefault(
            project, {"processed": 0, "edges": 0, "pairs": 0}
        )
        bucket["processed"] += processed
        bucket["edges"] += edges
        bucket["pairs"] += pairs


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
            f"--run-id='{stripped}' is a wildcard and is refused by policy."
        )
    for prefix, label in (
        (WT_COOCCUR_PREFIX, "CooccurrenceLinker"),
        (WT_TEMPORAL_PREFIX, "TemporalLinker"),
        (WT_ENTITY_PREFIX, "EntityLinker"),
        (WT_LINK_PREFIX, "SimilarityLinker"),
    ):
        if stripped.startswith(prefix):
            raise BackfillError(
                f"--run-id='{stripped}' starts with '{prefix}' which is "
                f"reserved for the {label} write-time path. Pick a distinct "
                "prefix (e.g. 'phase6-cooccur-YYYYMMDD')."
            )
    if not allow_test:
        for prefix in RESERVED_TEST_PREFIXES:
            if stripped.startswith(prefix):
                raise BackfillError(
                    f"--run-id='{stripped}' starts with '{prefix}' which is "
                    "reserved for sprint test scaffolding."
                )
    return stripped


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


def _default_checkpoint_path(run_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    return f"/tmp/assoc_backfill_cooccurrence_{safe}.checkpoint"


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
    """Token-spacing rate limiter — qps==0 disables."""

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
# Memory iteration — cursor-based over :base
# ---------------------------------------------------------------------------


async def _iter_memories(
    driver: "AsyncDriver",
    *,
    database: str,
    project_filter: Optional[str],
    batch_size: int,
) -> "AsyncIterator[Tuple[str, Optional[str]]]":
    """Async generator yielding ``(entity_id, project)`` tuples."""
    cursor: str = ""
    while True:
        if project_filter is not None:
            query = (
                "MATCH (n:base) "
                "WHERE n.entity_id IS NOT NULL "
                "AND n.entity_id > $cursor "
                "AND n.project = $project_filter "
                "RETURN n.entity_id AS memory_id, n.project AS project "
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
                "RETURN n.entity_id AS memory_id, n.project AS project "
                "ORDER BY n.entity_id "
                "LIMIT $batch_size"
            )
            params = {"cursor": cursor, "batch_size": batch_size}

        async with driver.session(database=database) as session:
            result = await session.run(query, params)
            batch = [(rec["memory_id"], rec.get("project")) async for rec in result]
            await result.consume()

        if not batch:
            return
        for row in batch:
            yield row
        cursor = batch[-1][0]


# ---------------------------------------------------------------------------
# Bulk co-mention fetch — one query per memory
# ---------------------------------------------------------------------------


#: Fetches one memory's non-hub entities alongside their co-mentioning
#: neighbor set, all in one round-trip. Pins both endpoints to
#: ``:base`` / ``:Entity`` so the walk can't leak into Session territory.
#:
#: Parameters:
#:   - ``$memory_id``      — the entity_id of the seed :base node
#:   - ``$project``        — the seed's project scope (required)
#:   - ``$hub_threshold``  — skip entities with degree > this value
#:   - ``$max_per_entity`` — cap co-mentioners returned per entity
_COMENTION_FETCH_CYPHER: str = (
    "MATCH (m:base {entity_id: $memory_id})-[:MENTIONS]->(e:Entity)\n"
    "WHERE e.project = $project\n"
    "WITH m, e, count { (e)<-[:MENTIONS]-(:base) } AS degree\n"
    "WHERE degree > 0 AND degree <= $hub_threshold\n"
    "OPTIONAL MATCH (other:base)-[:MENTIONS]->(e)\n"
    "WHERE other.entity_id <> $memory_id\n"
    "WITH e.name AS entity_name, degree, other.entity_id AS other_id\n"
    "ORDER BY other_id\n"
    "WITH entity_name, degree,\n"
    "     [x IN collect(DISTINCT other_id) WHERE x IS NOT NULL][0..$max_per_entity] AS comentioners\n"
    "RETURN entity_name, degree, comentioners"
)


#: Counts how many non-hub-passing entity slots exist for a memory. Used
#: to distinguish "no entities at all" from "all entities were hub-skipped"
#: in reporting.
_ENTITY_COUNT_CYPHER: str = (
    "MATCH (m:base {entity_id: $memory_id})-[:MENTIONS]->(e:Entity)\n"
    "WHERE e.project = $project\n"
    "RETURN count(e) AS total_entities"
)


async def _fetch_comentions(
    driver: "AsyncDriver",
    *,
    database: str,
    memory_id: str,
    project: str,
    hub_threshold: int,
    max_per_entity: int,
) -> Tuple[int, List[Tuple[str, int, List[str]]]]:
    """Return ``(total_entities, [(entity_name, degree, comentioners), ...])``.

    ``total_entities`` is the raw mention count before hub filtering — used
    by the caller to distinguish memories that have zero entities from
    memories whose entities were all hub-suppressed.
    """
    async with driver.session(database=database) as session:
        total_result = await session.run(
            _ENTITY_COUNT_CYPHER,
            {"memory_id": memory_id, "project": project},
        )
        total_rec = await total_result.single()
        await total_result.consume()
        total_entities = int(total_rec["total_entities"]) if total_rec else 0

        result = await session.run(
            _COMENTION_FETCH_CYPHER,
            {
                "memory_id": memory_id,
                "project": project,
                "hub_threshold": hub_threshold,
                "max_per_entity": max_per_entity,
            },
        )
        rows: List[Tuple[str, int, List[str]]] = []
        async for rec in result:
            rows.append(
                (
                    str(rec["entity_name"]),
                    int(rec["degree"]),
                    list(rec["comentioners"] or []),
                )
            )
        await result.consume()

    return total_entities, rows


# ---------------------------------------------------------------------------
# Core library entry point
# ---------------------------------------------------------------------------


async def backfill_cooccurrence_edges(
    *,
    run_id: str,
    driver: "AsyncDriver",
    database: str = "neo4j",
    batch_size: int = 100,
    max_total: Optional[int] = None,
    dry_run: bool = False,
    rate_limit_qps: float = 50.0,
    resume_from: Optional[str] = None,
    project_filter: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False,
    walltime_ceiling_sec: float = DEFAULT_WALLCLOCK_CEILING_SEC,
    _allow_test_run_id: bool = False,
    shutdown_event: Optional[asyncio.Event] = None,
) -> BackfillReport:
    """Iterate memories and create ``CO_OCCURS`` edges between pairs sharing entities.

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
        When True, run the co-mention scan but do NOT write edges.
    rate_limit_qps:
        Upper bound on per-memory query bursts per second. Default 50.0.
    resume_from:
        Skip all memories up to and including this ``entity_id``.
    project_filter:
        Only process memories where ``n.project = $filter``.
    checkpoint_path:
        Path to write the cursor file every 100 processed memories.
    verbose:
        Enable DEBUG-level logging.
    walltime_ceiling_sec:
        Hard wall-clock budget (default 120s). If exceeded the loop exits
        cleanly with ``walltime_ceiling_hit=True`` in the report.
    _allow_test_run_id:
        Bypass test-prefix refusal. Only used by integration tests.
    shutdown_event:
        Optional ``asyncio.Event``. If set mid-run, the loop finishes the
        current memory and exits cleanly.
    """
    # Late imports keep the module top-level I/O free.
    from app.services.associations.edge_cypher import build_merge_edge_cypher
    from app.services.associations.memory_edges import (
        EDGE_VERSION,
        MemoryEdge,
    )

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

    merge_cypher = build_merge_edge_cypher("CO_OCCURS")

    rate_limiter = _AsyncRateLimiter(rate_limit_qps)

    LOGGER.info(
        "backfill.start run_id=%s tagged=%s dry_run=%s batch_size=%d "
        "max_total=%s rate_limit_qps=%s project_filter=%s resume_from=%s "
        "walltime_ceiling=%.1fs",
        normalized_run_id,
        tagged_run_id,
        dry_run,
        batch_size,
        max_total,
        rate_limit_qps,
        project_filter,
        resume_from,
        walltime_ceiling_sec,
    )

    processed_counter = 0
    cursor_passed_resume = resume_from is None
    last_memory_id: Optional[str] = None

    # Canonical-pair dedup set. Stored as ``(src, tgt)`` with ``src <= tgt``
    # so re-encountering the pair from the other endpoint is a no-op.
    seen_pairs: set[Tuple[str, str]] = set()

    deadline = time.monotonic() + walltime_ceiling_sec

    try:
        async for memory_id, project in _iter_memories(
            driver=driver,
            database=database,
            project_filter=project_filter,
            batch_size=batch_size,
        ):
            if time.monotonic() >= deadline:
                report.walltime_ceiling_hit = True
                LOGGER.warning(
                    "backfill.walltime_ceiling_hit processed=%d ceiling_sec=%.1f",
                    processed_counter,
                    walltime_ceiling_sec,
                )
                break

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

            if project is None:
                report.memories_skipped_no_project += 1
                LOGGER.debug(
                    "backfill.skip_no_project memory_id=%s", memory_id
                )
                last_memory_id = memory_id
                continue

            await rate_limiter.wait()

            # Fetch this memory's co-mentions in one round-trip.
            try:
                total_entities, rows = await _fetch_comentions(
                    driver,
                    database=database,
                    memory_id=memory_id,
                    project=project,
                    hub_threshold=HUB_THRESHOLD,
                    max_per_entity=MAX_EDGES_PER_ENTITY,
                )
            except Exception as exc:  # noqa: BLE001 — fail-open per memory
                report.errors.append(
                    f"memory_id={memory_id}: fetch_comentions: "
                    f"{type(exc).__name__}: {exc}"
                )
                LOGGER.warning(
                    "backfill.fetch_failed memory_id=%s error=%s",
                    memory_id,
                    exc,
                )
                last_memory_id = memory_id
                continue

            if total_entities == 0:
                report.memories_skipped_no_entities += 1
                LOGGER.debug(
                    "backfill.skip_no_entities memory_id=%s", memory_id
                )
                last_memory_id = memory_id
                processed_counter += 1
                report.memories_processed += 1
                report.bump_project(project, processed=1)
                continue

            # Count how many entities were hub-suppressed (total - kept).
            kept_entity_count = len(rows)
            report.entities_evaluated += kept_entity_count
            report.entities_hub_skipped += max(
                0, total_entities - kept_entity_count
            )

            if kept_entity_count == 0:
                report.memories_with_all_hub_entities += 1
                LOGGER.debug(
                    "backfill.all_hub memory_id=%s total=%d",
                    memory_id,
                    total_entities,
                )
                last_memory_id = memory_id
                processed_counter += 1
                report.memories_processed += 1
                report.bump_project(project, processed=1)
                continue

            # Build the (other_id → shared entities) map and the
            # (entity_name → degree) map.
            comentions: Dict[str, List[str]] = {}
            entity_degrees: Dict[str, int] = {}
            for entity_name, degree, comentioners in rows:
                entity_degrees[entity_name] = max(degree, 1)
                for other_id in comentioners:
                    if not other_id or other_id == memory_id:
                        continue
                    comentions.setdefault(other_id, []).append(entity_name)

            if not comentions:
                last_memory_id = memory_id
                processed_counter += 1
                report.memories_processed += 1
                report.bump_project(project, processed=1)
                continue

            max_degree = max(entity_degrees.values(), default=1)
            max_degree = max(max_degree, 1)

            edges_this_memory = 0
            pairs_this_memory = 0
            dedup_hits = 0
            now_iso = datetime.now(tz=timezone.utc).isoformat()

            for other_id, shared in comentions.items():
                report.pairs_evaluated += 1
                pairs_this_memory += 1

                if len(shared) < MIN_SHARED_ENTITIES:
                    report.pairs_below_min_shared += 1
                    continue

                # Canonicalize the pair — CO_OCCURS is bidirectional, so
                # we want exactly one stored edge regardless of which
                # endpoint we walked from.
                src, tgt = MemoryEdge.canonicalize_for_bidirectional(
                    memory_id, other_id
                )
                pair_key = (src, tgt)
                if pair_key in seen_pairs:
                    dedup_hits += 1
                    continue
                seen_pairs.add(pair_key)

                # IDF-style weight: identical formula to
                # cooccurrence_linker._link_one.
                idf_sum = 0.0
                for ent_name in shared:
                    degree = entity_degrees.get(ent_name, 1)
                    idf_sum += math.log(max(max_degree / degree, 1.0))
                weight = min(1.0, idf_sum / 10.0)
                weight = max(weight, 0.01)
                if not math.isfinite(weight):
                    weight = 0.01

                report.pairs_emitted += 1

                if dry_run:
                    edges_this_memory += 1
                    continue

                try:
                    params = {
                        "src_id": src,
                        "dst_id": tgt,
                        "weight": round(float(weight), 4),
                        "created_at": now_iso,
                        "last_seen_at": now_iso,
                        "created_by": "assoc_backfill_cooccurrence",
                        "run_id": tagged_run_id,
                        "edge_version": EDGE_VERSION,
                        "metadata": None,
                    }
                    async with driver.session(database=database) as session:
                        result = await session.run(merge_cypher, params)
                        records = [rec async for rec in result]
                        await result.consume()
                        if records:
                            edges_this_memory += 1
                except Exception as exc:  # noqa: BLE001 — per-edge fail-open
                    report.errors.append(
                        f"memory_id={memory_id} -> {other_id}: upsert: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    LOGGER.warning(
                        "backfill.edge_failed %s -> %s error=%s",
                        memory_id,
                        other_id,
                        exc,
                    )
                    continue

            report.edges_created += edges_this_memory
            report.edges_deduped_canonical += dedup_hits
            report.bump_project(
                project,
                processed=1,
                edges=edges_this_memory,
                pairs=pairs_this_memory,
            )
            processed_counter += 1
            report.memories_processed += 1
            last_memory_id = memory_id

            LOGGER.debug(
                "backfill.memory_done memory_id=%s entities=%d pairs=%d "
                "edges=%d dedup=%d",
                memory_id,
                kept_entity_count,
                pairs_this_memory,
                edges_this_memory,
                dedup_hits,
            )

            if processed_counter % 100 == 0:
                _write_checkpoint_atomic(
                    effective_checkpoint_path,
                    normalized_run_id,
                    last_memory_id,
                    processed_counter,
                )
                LOGGER.info(
                    "backfill.checkpoint path=%s count=%d last=%s edges=%d",
                    effective_checkpoint_path,
                    processed_counter,
                    last_memory_id,
                    report.edges_created,
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

    LOGGER.info(
        "backfill.done run_id=%s scanned=%d processed=%d no_project=%d "
        "no_entities=%d all_hub=%d entities_eval=%d entities_hub=%d "
        "pairs_eval=%d pairs_below_min=%d pairs_emitted=%d edges=%d "
        "dedup=%d errors=%d walltime_hit=%s",
        normalized_run_id,
        report.memories_scanned,
        report.memories_processed,
        report.memories_skipped_no_project,
        report.memories_skipped_no_entities,
        report.memories_with_all_hub_entities,
        report.entities_evaluated,
        report.entities_hub_skipped,
        report.pairs_evaluated,
        report.pairs_below_min_shared,
        report.pairs_emitted,
        report.edges_created,
        report.edges_deduped_canonical,
        len(report.errors),
        report.walltime_ceiling_hit,
    )
    return report


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assoc_backfill_cooccurrence",
        description=(
            "PLAN-0759 Phase 7a — retroactively create CO_OCCURS edges "
            "between memories sharing >=2 non-hub entities, mirroring "
            "CooccurrenceLinker semantics with canonical deduplication."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Concrete run identifier. Wildcards and reserved prefixes "
            "(wt-cooccur-, wt-temporal-, wt-entity-, wt-link-, "
            "sprint{2,5,6,7,8,9}-) are refused."
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
        help="Max memories to process. Integer or 'unlimited'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the scan without writing any edges.",
    )
    parser.add_argument(
        "--rate-limit-qps",
        type=float,
        default=50.0,
        help="Per-memory query rate ceiling. Default 50.0. 0 disables.",
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
        "--walltime-ceiling-sec",
        type=float,
        default=DEFAULT_WALLCLOCK_CEILING_SEC,
        help="Hard wall-clock budget in seconds. Default 120.",
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
        report = await backfill_cooccurrence_edges(
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
            walltime_ceiling_sec=args.walltime_ceiling_sec,
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
            f"DRY RUN — would create {report.edges_created} CO_OCCURS edges "
            f"({report.pairs_emitted} canonical pairs, "
            f"{report.pairs_evaluated} pairs evaluated, "
            f"{report.entities_hub_skipped} hub entities skipped).",
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
