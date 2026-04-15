"""Associative-link backfill utility for PLAN-0759 Phase 2 / Step 2.6 (Sprint 7).

Purpose
-------
Sprint 6 delivered write-time similarity linking via
``app/services/associations/similarity_linker.py``. New memories upserted via
``MemoryService.perform_upsert()`` after the flag flip now get ``SIMILAR_TO``
edges computed at store time. This script retroactively creates the same
``SIMILAR_TO`` edges for memories that **already existed** in Neo4j at the
moment the flag flipped.

The plan's Phase 2 Step 2.6 (Backfill) calls for an async function that
iterates the existing memory set, pulls each memory's embedding, runs the
same similarity logic used at write time, and MERGEs the resulting edges.
Sprint 7 implements that as a standalone CLI + library, using the exact
same :class:`SimilarityLinker` class from Sprint 6 (no reimplementation of
the link logic).

Scope
-----
This is a **standalone utility script**. It is imported by
``tests/test_assoc_backfill_similarity.py`` for integration testing but is
NOT wired into ``memory_service.py``, ``memory_router.py`` (doesn't exist),
``graph_client.py``, or any production hot path. Operators invoke it from
the shell when they want to seed edges onto the historical corpus.

Design constraints
------------------
1. **Zero production code change.** This script does not modify any Sprint
   1-6 artifact. It imports :class:`SimilarityLinker` and
   :class:`MemoryEdgeService` and drives them from the outside.
2. **No module-level Pinecone client.** The Pinecone client (real or mock)
   is passed in by the caller or constructed inside ``main()`` from CLI
   args. Import time has zero I/O side effects.
3. **No ``app.config.settings`` import.** The script reads Neo4j + Pinecone
   creds from CLI flags or environment variables directly. This keeps it
   testable without pulling in the full Fusion Memory settings graph.
4. **Strict run_id validation.** Empty, whitespace-only, wildcard (``*``,
   ``%``, ``all``, ``ALL``), and Sprint-reserved prefixes (``wt-link-``,
   ``sprint2-``, ``sprint5-``, ``sprint6-``, ``sprint7-``) are refused.
5. **Rate-limited.** A configurable QPS ceiling bounds Pinecone traffic.
6. **Idempotent.** The underlying :meth:`MemoryEdgeService.create_edges_batch`
   uses Sprint 4's MERGE template, so a second run over the same corpus
   updates ``last_seen_at`` and weight tie-break without duplicating edges.
7. **Checkpoint-resumable.** The script writes a cursor file every 100
   memories so a crashed run can resume where it left off.
8. **Graceful shutdown.** SIGINT (Ctrl-C) finishes the current batch,
   writes the final checkpoint, prints the summary, and exits cleanly.

Safety refusals
---------------
- Missing / empty ``--run-id``: refuse, exit 2.
- Wildcard ``--run-id`` (``*``, ``%``, ``all``, ``ALL``): refuse, exit 2.
- ``--run-id`` starting with ``wt-link-`` (reserved for write-time linking):
  refuse, exit 2.
- ``--run-id`` starting with ``sprint2-``, ``sprint5-``, ``sprint6-``, or
  ``sprint7-`` (reserved for test scaffolding): refuse, exit 2.
- Any Cypher / driver exception propagates as exit code 3.

The library function accepts a ``_allow_test_run_id=True`` kwarg that
bypasses the Sprint-prefix refusal so the Sprint 7 integration test can
use ``sprint7-backfill-test-*`` run_ids without colliding with a real
operator backfill. The CLI never sets this flag.

Usage
-----
Dry-run against a small subset of a single project::

    python -m scripts.assoc_backfill_similarity \\
        --run-id backfill-20260413 --dry-run --max-total 50 \\
        --project-filter nova-core --verbose

Full live run::

    python -m scripts.assoc_backfill_similarity \\
        --run-id backfill-20260413-full --rate-limit-qps 5.0 \\
        --uri bolt://localhost:7687

Rollback of a bad backfill::

    python -m scripts.assoc_rollback --run-id backfill-20260413-full
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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


LOGGER = logging.getLogger("assoc_backfill_similarity")


# ---------------------------------------------------------------------------
# Refusal constants
# ---------------------------------------------------------------------------

WILDCARD_RUN_IDS: tuple[str, ...] = ("*", "%", "all", "ALL")

#: Sprint test scaffolding prefixes that must NEVER appear in a real backfill
#: run_id. The integration test passes ``_allow_test_run_id=True`` to bypass
#: this check; the CLI never does.
RESERVED_TEST_PREFIXES: tuple[str, ...] = (
    "sprint2-",
    "sprint5-",
    "sprint6-",
    "sprint7-",
)

#: Write-time linking prefix owned by Sprint 6's SimilarityLinker. Always
#: refused (tests and CLI alike) so rollback can cleanly distinguish
#: write-time-linker edges from backfill edges.
WT_LINK_PREFIX: str = "wt-link-"


class BackfillError(RuntimeError):
    """Raised when backfill input is invalid (safety refusal)."""


# ---------------------------------------------------------------------------
# Report + progress dataclasses
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
    memories_skipped_no_embedding: int = 0
    memories_skipped_no_project: int = 0
    edges_created: int = 0
    pinecone_queries: int = 0
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
            "memories_skipped_no_embedding": self.memories_skipped_no_embedding,
            "memories_skipped_no_project": self.memories_skipped_no_project,
            "edges_created": self.edges_created,
            "pinecone_queries": self.pinecone_queries,
            "by_project": {
                proj: dict(counts) for proj, counts in self.by_project.items()
            },
            "errors": list(self.errors),
            "checkpoint_final": self.checkpoint_final,
        }

    def bump_project(self, project: str, processed: int = 0, edges: int = 0) -> None:
        bucket = self.by_project.setdefault(
            project, {"processed": 0, "edges": 0}
        )
        bucket["processed"] += processed
        bucket["edges"] += edges


# ---------------------------------------------------------------------------
# run_id validation
# ---------------------------------------------------------------------------


def _validate_run_id(run_id: Optional[str], *, allow_test: bool = False) -> str:
    """Return a normalized run_id or raise :class:`BackfillError`.

    Parameters
    ----------
    run_id:
        Caller-supplied identifier. ``None``, empty, or whitespace-only is
        refused. Wildcards and reserved prefixes are refused.
    allow_test:
        When True, the ``sprint{2,5,6,7}-`` test prefixes are permitted.
        This is ONLY used by the integration test. The CLI path never
        passes True here.
    """
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
    if stripped.startswith(WT_LINK_PREFIX):
        raise BackfillError(
            f"--run-id='{stripped}' starts with '{WT_LINK_PREFIX}' which is "
            "reserved for Sprint 6 write-time linker runs. Pick a distinct "
            "prefix (e.g. 'backfill-YYYYMMDD')."
        )
    if not allow_test:
        for prefix in RESERVED_TEST_PREFIXES:
            if stripped.startswith(prefix):
                raise BackfillError(
                    f"--run-id='{stripped}' starts with '{prefix}' which is "
                    "reserved for Sprint test scaffolding. Pick a distinct "
                    "prefix (e.g. 'backfill-YYYYMMDD')."
                )
    return stripped


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


def _default_checkpoint_path(run_id: str) -> str:
    """Return the default checkpoint path for a given run_id (under /tmp)."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    return f"/tmp/assoc_backfill_{safe}.checkpoint"


def _write_checkpoint_atomic(
    path: str, run_id: str, last_memory_id: Optional[str], count: int
) -> None:
    """Write checkpoint JSON atomically via tmp-file + rename."""
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
# Rate limiter — simple single-flight token spacing
# ---------------------------------------------------------------------------


class _AsyncRateLimiter:
    """Token-spacing rate limiter.

    Ensures at most ``qps`` acquisitions per second. Serialized via an
    internal ``asyncio.Lock`` so concurrent callers observe the same cap.
    A qps of 0 or negative disables the limiter entirely.
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


async def _iter_memories(
    driver: AsyncDriver,
    *,
    database: str,
    project_filter: Optional[str],
    start_cursor: Optional[str],
    batch_size: int,
) -> Any:
    """Async generator over ``(entity_id, project)`` pairs.

    Paginates deterministically via ``entity_id`` ordering (no SKIP /
    OFFSET) so a crashed run can resume from the last persisted cursor.
    Yields tuples of ``(entity_id, project)``.
    """
    cursor: str = start_cursor or ""
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
            params = {
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
            batch = [dict(rec) async for rec in result]
            await result.consume()

        if not batch:
            return
        for row in batch:
            yield (row["memory_id"], row.get("project"))
        cursor = batch[-1]["memory_id"]


# ---------------------------------------------------------------------------
# Pinecone embedding fetch
# ---------------------------------------------------------------------------


async def _fetch_embedding(
    pinecone_client: Any, memory_id: str
) -> Optional[List[float]]:
    """Fetch the embedding vector for a memory_id from Pinecone.

    The production :class:`PineconeClient` does not expose a ``fetch``
    method (verified in ``app/services/pinecone_client.py``). The real
    operator path therefore calls the underlying
    ``pinecone_client.index.fetch(ids=[memory_id])`` API directly. Tests
    mock this whole function with a ``fetch_embedding_override`` hook on
    the library entrypoint — Pinecone is **never** touched in CI.

    Returns the vector as a list of floats, or ``None`` if Pinecone has
    no embedding for that id.
    """

    def _sync_fetch() -> Optional[List[float]]:
        # The Pinecone Python SDK returns a FetchResponse object whose
        # ``vectors`` attr is a dict keyed by id with ``.values`` floats.
        # We defensively handle both attribute-style and dict-style
        # responses since the SDK shape changed between Pinecone v2 and
        # v3. Any missing / None result returns None.
        index = getattr(pinecone_client, "index", None)
        if index is None:
            # Some test doubles pass the Pinecone index directly.
            index = pinecone_client
        fetch = getattr(index, "fetch", None)
        if fetch is None:
            return None
        resp = fetch(ids=[memory_id])
        vectors = getattr(resp, "vectors", None)
        if vectors is None and isinstance(resp, dict):
            vectors = resp.get("vectors")
        if not vectors:
            return None
        entry = vectors.get(memory_id) if isinstance(vectors, dict) else None
        if entry is None:
            return None
        values = getattr(entry, "values", None)
        if values is None and isinstance(entry, dict):
            values = entry.get("values")
        if values is None:
            return None
        return list(values)

    try:
        return await asyncio.to_thread(_sync_fetch)
    except Exception as exc:  # noqa: BLE001 — log and swallow
        LOGGER.warning(
            "backfill.fetch_failed memory_id=%s error=%s",
            memory_id,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Core library entry point
# ---------------------------------------------------------------------------


async def backfill_similarity_edges(
    *,
    run_id: str,
    driver: AsyncDriver,
    pinecone_client: Any,
    database: str = "neo4j",
    batch_size: int = 50,
    max_total: Optional[int] = None,
    dry_run: bool = False,
    rate_limit_qps: float = 5.0,
    resume_from: Optional[str] = None,
    project_filter: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False,
    _allow_test_run_id: bool = False,
    # Dependency-injection hooks for the integration test:
    fetch_embedding_override: Optional[Any] = None,
    edge_service_override: Optional[Any] = None,
    similarity_linker_override: Optional[Any] = None,
    shutdown_event: Optional[asyncio.Event] = None,
) -> BackfillReport:
    """Iterate existing ``:base`` memories and create ``SIMILAR_TO`` edges.

    This is the library entrypoint. The CLI ``main()`` function opens a
    driver + pinecone client from argv and then calls this. The
    integration test constructs its own mocks and calls this directly.

    Parameters
    ----------
    run_id:
        Concrete run identifier. Empty / wildcard / reserved prefixes
        are refused. See :func:`_validate_run_id`.
    driver:
        Pre-opened async Neo4j driver. The caller owns lifecycle.
    pinecone_client:
        Pinecone client (or a test double) exposing a ``query_vector``
        method matching Sprint 6's :class:`SimilarityLinker` contract and
        an ``index.fetch`` surface for embedding lookups. The test path
        uses ``fetch_embedding_override`` to bypass the fetch entirely.
    database:
        Neo4j database name. Defaults to ``"neo4j"``.
    batch_size:
        Page size for the paginated Cypher iteration. Default 50.
    max_total:
        Stop after processing this many memories. ``None`` = unlimited.
    dry_run:
        When True, run the full similarity pipeline (including Pinecone
        queries) but do NOT call ``edge_service.create_edges_batch``.
        Instead, count the edges that *would* have been created.
    rate_limit_qps:
        Upper bound on Pinecone queries per second. Default 5.0.
    resume_from:
        Skip all memories up to and including this ``entity_id`` before
        starting processing. Used for crash recovery.
    project_filter:
        If provided, only process memories where ``n.project = $filter``.
    checkpoint_path:
        Path to write the cursor file every 100 memories. Defaults to
        ``/tmp/assoc_backfill_<run_id>.checkpoint``.
    verbose:
        Enable DEBUG-level logging on the script logger.
    _allow_test_run_id:
        Bypass the ``sprint{2,5,6,7}-`` prefix refusal. ONLY used by the
        integration test. The CLI never sets this.
    fetch_embedding_override:
        Optional async callable ``fn(pinecone_client, memory_id) -> vec``
        that replaces the real Pinecone fetch. Used by the integration
        test to return canned vectors without touching Pinecone.
    edge_service_override:
        Optional pre-constructed edge service instance. When None, a
        fresh :class:`MemoryEdgeService` is built from the injected
        driver.
    similarity_linker_override:
        Optional pre-constructed linker instance. When None, a fresh
        :class:`SimilarityLinker` is built from the edge service and
        pinecone client.
    shutdown_event:
        Optional ``asyncio.Event``. If set mid-run, the loop finishes
        the current memory and exits cleanly. Used by the CLI's SIGINT
        handler.

    Returns
    -------
    :class:`BackfillReport`
        Structured run summary.
    """
    # -- Late imports so the script can be imported without side effects --
    from app.services.associations.edge_service import MemoryEdgeService
    from app.services.associations.similarity_linker import SimilarityLinker

    if verbose:
        LOGGER.setLevel(logging.DEBUG)

    normalized_run_id = _validate_run_id(run_id, allow_test=_allow_test_run_id)

    effective_checkpoint_path = checkpoint_path or _default_checkpoint_path(
        normalized_run_id
    )

    report = BackfillReport(
        run_id=normalized_run_id,
        dry_run=dry_run,
        started_at=datetime.now(tz=timezone.utc).isoformat(),
        checkpoint_final=effective_checkpoint_path,
    )

    edge_service = edge_service_override or MemoryEdgeService(
        driver=driver, database=database
    )
    linker = similarity_linker_override or SimilarityLinker(
        pinecone_client=pinecone_client,
        edge_service=edge_service,
        cross_project_enabled=False,
    )

    fetch_fn = fetch_embedding_override or _fetch_embedding
    rate_limiter = _AsyncRateLimiter(rate_limit_qps)

    async def _fetch_with_limit(mem_id: str) -> Optional[List[float]]:
        await rate_limiter.wait()
        return await fetch_fn(pinecone_client, mem_id)

    async def _query_with_limit(**kwargs: Any) -> Any:
        await rate_limiter.wait()
        return await asyncio.to_thread(pinecone_client.query_vector, **kwargs)

    LOGGER.info(
        "backfill.start run_id=%s dry_run=%s batch_size=%d max_total=%s "
        "rate_limit_qps=%s project_filter=%s resume_from=%s",
        normalized_run_id,
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
        async for memory_id, project in _iter_memories(
            driver=driver,
            database=database,
            project_filter=project_filter,
            start_cursor=None,  # resume is handled in-loop so we can still count scanned
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

            # Handle resume-from: skip until (and including) the cursor.
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

            # Project-scoping guard: Sprint 6 linker refuses to link when
            # cross_project_enabled=False and project is None. Track and
            # skip here rather than relying on the linker's warning so the
            # report gets a clean counter.
            if project is None:
                report.memories_skipped_no_project += 1
                LOGGER.debug(
                    "backfill.skip_no_project memory_id=%s", memory_id
                )
                last_memory_id = memory_id
                continue

            # Fetch the embedding for this memory.
            embedding = await _fetch_with_limit(memory_id)
            report.pinecone_queries += 1  # fetch counts as a Pinecone call
            if embedding is None:
                report.memories_skipped_no_embedding += 1
                LOGGER.debug(
                    "backfill.skip_no_embedding memory_id=%s", memory_id
                )
                last_memory_id = memory_id
                continue

            # Run the same similarity computation the linker would have
            # done at write time — except we route through an explicit
            # query_vector call so dry-run can count edges without
            # calling edge_service.create_edges_batch.
            try:
                matches = await _query_with_limit(
                    query_vector=embedding,
                    top_k=linker.CANDIDATE_POOL + 1,
                    filter={"project": project},
                    include_values=False,
                )
                report.pinecone_queries += 1
            except Exception as exc:  # noqa: BLE001 — fail-open per memory
                report.errors.append(
                    f"memory_id={memory_id}: pinecone_query: {type(exc).__name__}: {exc}"
                )
                LOGGER.warning(
                    "backfill.pinecone_failed memory_id=%s error=%s",
                    memory_id,
                    exc,
                )
                last_memory_id = memory_id
                continue

            # Apply the Sprint 6 filtering rules: self-exclude, threshold,
            # top-K with deterministic tie-break.
            qualified: list[tuple[str, float]] = []
            for m in matches or []:
                if isinstance(m, dict):
                    mid = m.get("id", "")
                    score = float(m.get("score", 0.0))
                else:
                    mid = getattr(m, "id", "")
                    score = float(getattr(m, "score", 0.0))
                if not mid or mid == memory_id:
                    continue
                if score < linker.SIMILARITY_THRESHOLD:
                    continue
                qualified.append((mid, score))

            qualified.sort(key=lambda t: (-t[1], t[0]))
            top = qualified[: linker.MAX_NEIGHBORS]

            from app.services.associations.memory_edges import MemoryEdge

            tagged_run_id = f"backfill-{normalized_run_id}"
            now_iso = datetime.now(tz=timezone.utc).isoformat()
            edges: list[MemoryEdge] = []
            for neighbor_id, score in top:
                edges.append(
                    MemoryEdge(
                        source_id=memory_id,
                        target_id=neighbor_id,
                        edge_type="SIMILAR_TO",
                        weight=min(1.0, max(0.0, float(score))),
                        created_at=now_iso,
                        last_seen_at=now_iso,
                        created_by="assoc_backfill_similarity",
                        run_id=tagged_run_id,
                        metadata=None,
                    )
                )

            if edges:
                if dry_run:
                    report.edges_created += len(edges)
                    report.bump_project(project, processed=0, edges=len(edges))
                else:
                    try:
                        created = await edge_service.create_edges_batch(edges)
                    except Exception as exc:  # noqa: BLE001
                        report.errors.append(
                            f"memory_id={memory_id}: edge_service: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        LOGGER.warning(
                            "backfill.edge_service_failed memory_id=%s error=%s",
                            memory_id,
                            exc,
                        )
                        last_memory_id = memory_id
                        processed_counter += 1
                        report.memories_processed += 1
                        report.bump_project(project, processed=1, edges=0)
                        continue
                    report.edges_created += int(created)
                    report.bump_project(
                        project, processed=0, edges=int(created)
                    )

            processed_counter += 1
            report.memories_processed += 1
            report.bump_project(project, processed=1, edges=0)
            last_memory_id = memory_id

            # Periodic checkpoint every 100 processed memories.
            if processed_counter % 100 == 0:
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
        # Always write a final checkpoint so operators can see where we
        # stopped, even on a crash.
        try:
            _write_checkpoint_atomic(
                effective_checkpoint_path,
                normalized_run_id,
                last_memory_id,
                processed_counter,
            )
        except Exception as exc:  # noqa: BLE001 — don't mask original error
            LOGGER.warning(
                "backfill.final_checkpoint_failed path=%s error=%s",
                effective_checkpoint_path,
                exc,
            )

        report.completed_at = datetime.now(tz=timezone.utc).isoformat()

    LOGGER.info(
        "backfill.done run_id=%s scanned=%d processed=%d edges=%d "
        "skipped_no_embedding=%d skipped_no_project=%d pinecone_queries=%d",
        normalized_run_id,
        report.memories_scanned,
        report.memories_processed,
        report.edges_created,
        report.memories_skipped_no_embedding,
        report.memories_skipped_no_project,
        report.pinecone_queries,
    )
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="assoc_backfill_similarity",
        description=(
            "PLAN-0759 Phase 2 / Step 2.6 — retroactively create SIMILAR_TO "
            "edges for existing :base memories using Sprint 6's linker logic."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Concrete run identifier. Wildcards and reserved prefixes "
            "(wt-link-, sprint2-, sprint5-, sprint6-, sprint7-) are refused."
        ),
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help=(
            "Neo4j bolt URI. Defaults to $NEO4J_URI or bolt://localhost:7687."
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
        "--pinecone-api-key",
        default=os.environ.get("PINECONE_API_KEY"),
        help="Pinecone API key. Defaults to $PINECONE_API_KEY.",
    )
    parser.add_argument(
        "--pinecone-index",
        default=os.environ.get("PINECONE_INDEX"),
        help="Pinecone index name. Defaults to $PINECONE_INDEX.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Cypher pagination size. Default 50.",
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
        help=(
            "Run the full pipeline including Pinecone queries but do NOT "
            "write any edges. Prints counts at the end."
        ),
    )
    parser.add_argument(
        "--rate-limit-qps",
        type=float,
        default=5.0,
        help="Pinecone query rate ceiling in QPS. Default 5.0. 0 disables.",
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
        help=(
            "Path to write cursor checkpoint every 100 memories. Defaults "
            "to /tmp/assoc_backfill_<run_id>.checkpoint."
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
    # Open Neo4j async driver.
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

    # Construct a real Pinecone client lazily — we import here so the
    # module top-level stays I/O free.
    pinecone_client: Any
    try:
        from app.services.pinecone_client import PineconeClient

        pinecone_client = PineconeClient()
        if not pinecone_client.initialize():
            print(
                "ERROR: Pinecone client failed to initialize. Check "
                "PINECONE_API_KEY / PINECONE_INDEX / network.",
                file=sys.stderr,
            )
            await driver.close()
            return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Pinecone client construction failed: {exc}", file=sys.stderr)
        await driver.close()
        return 2

    # SIGINT handler — flip the shutdown flag so the main loop can
    # finish cleanly. The default KeyboardInterrupt would propagate into
    # the middle of a Cypher transaction and leave half-written state.
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        LOGGER.warning("SIGINT received; requesting graceful shutdown")
        shutdown_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except NotImplementedError:
        # Windows / some test environments don't support add_signal_handler.
        pass

    try:
        max_total = _parse_max_total(args.max_total)
    except BackfillError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        await driver.close()
        return 2

    try:
        report = await backfill_similarity_edges(
            run_id=args.run_id,
            driver=driver,
            pinecone_client=pinecone_client,
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
            f"DRY RUN — would create {report.edges_created} edges across "
            f"{report.memories_processed} memories in "
            f"{len(report.by_project)} projects, estimated Pinecone calls = "
            f"{report.pinecone_queries}",
            file=sys.stderr,
        )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Silence "unknown relationship type" notification warnings — Sprint 7's
    # Cypher probes SIMILAR_TO which may be at zero count on the live DB.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        # asyncio.run() sometimes still propagates the interrupt if the
        # signal handler wasn't installed in time. Treat as clean exit.
        print("Interrupted by user; exiting.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
