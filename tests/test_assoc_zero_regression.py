"""Phase 0 zero-regression baseline (PLAN-0759 / Step 0.3).

Purpose
-------
Sprint 1 of PLAN-0759 added 8 ``ASSOC_*`` feature flags to
``app.config.Settings``, all defaulting to ``False``. The v2 plan's Step 0.3
contract is: *with every ``ASSOC_*`` flag False, the behavior of
``MemoryService.perform_upsert()`` and ``MemoryService.perform_query()``
must be byte-for-byte identical to pre-Sprint-1 behavior.* This test pins
that invariant so any future refactor that leaks associative-linking
behavior through a flag-guarded branch is caught immediately.

Design
------
The test is **fully hermetic** — it does not touch the real Pinecone index
or the live ``nova_neo4j_db`` container. Instead it patches the three
backend dependencies of ``MemoryService`` (``PineconeClient``,
``GraphClient``, ``get_embedding``) with deterministic in-memory fakes.
This mirrors the isolation pattern already in use by
``tests/test_memory_service.py`` and avoids any risk of polluting
production data.

What it actually pins
~~~~~~~~~~~~~~~~~~~~~
For each of the 10 canonical fixtures in
``tests/fixtures/phase0_regression_memories.json``, the test runs a
store→recall round-trip and captures **three** artifacts:

1. The sequence of calls ``MemoryService`` issued to the fake
   ``PineconeClient`` (which ``upsert_vector``/``query_vector`` args were
   passed, minus volatile fields like ``event_seq`` and ``event_time``
   which are system-generated).
2. The sequence of calls ``MemoryService`` issued to the fake
   ``GraphClient`` (``upsert_graph_data`` args, same volatile-field
   redaction).
3. The shape and content of the ``perform_query`` result list returned to
   the caller.

These three artifacts together form the "behavior signature" of the
orchestration. They are serialized to
``tests/fixtures/phase0_regression_baseline.json`` on first run and then
re-checked byte-for-byte on every subsequent run.

Volatile field handling
~~~~~~~~~~~~~~~~~~~~~~~
``_inject_chronology`` always writes ``event_seq`` (monotonic counter) and
``event_time`` (wall clock if not supplied). To keep the baseline stable
we:

- Supply ``event_time`` in every fixture so the injection never has to
  fall back to ``datetime.now()``.
- Patch ``SequenceService.next_seq`` to return a deterministic counter
  starting at 1000, so the event_seq sequence is reproducible.
- Still strip volatile metadata keys from the captured snapshot as a
  defence-in-depth guard.

Flag-default sanity check
~~~~~~~~~~~~~~~~~~~~~~~~~
Before running the round-trip, the test asserts that all 8 ``ASSOC_*``
flags on ``settings`` are currently ``False``. If any of them is ``True``
the test aborts with a loud error rather than pinning a polluted
baseline.

If the baseline needs to be regenerated intentionally (e.g. the fixture
set changes), delete ``tests/fixtures/phase0_regression_baseline.json``
and re-run. The test will create a fresh baseline and pass, and the next
run will lock it in.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Make the ``app`` package importable ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

from app.config import settings  # noqa: E402


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURES_PATH = FIXTURES_DIR / "phase0_regression_memories.json"
BASELINE_PATH = FIXTURES_DIR / "phase0_regression_baseline.json"

# Deterministic starting sequence number for _inject_chronology.
_FAKE_SEQ_START = 1000

# Metadata keys that are system-generated or volatile — stripped from the
# captured snapshot so the baseline remains stable across runs.
_VOLATILE_META_KEYS = frozenset({"event_seq"})

# Top-level result keys that are derived from wall-clock ``datetime.now()``
# (e.g. ``temporal_decay_score`` uses the current time to compute recency)
# and therefore drift on every run. These are stripped from the captured
# query results so the baseline pins the *retrieval+merge* behavior, not
# the wall-clock-dependent temporal scoring. Temporal scoring is not part
# of the ASSOC_* zero-regression contract.
_VOLATILE_RESULT_KEYS = frozenset(
    {"temporal_score", "composite_score", "semantic_score_normalized"}
)

# All 8 Sprint-1 ASSOC flags. Keep in sync with
# tests/test_assoc_feature_flags.py::ASSOC_FLAG_NAMES.
ASSOC_FLAG_NAMES: Tuple[str, ...] = (
    "ASSOC_SIMILARITY_WRITE_ENABLED",
    "ASSOC_ENTITY_WRITE_ENABLED",
    "ASSOC_TEMPORAL_WRITE_ENABLED",
    "ASSOC_PROVENANCE_WRITE_ENABLED",
    "ASSOC_COOCCURRENCE_WRITE_ENABLED",
    "ASSOC_TASK_HEURISTIC_WRITE_ENABLED",
    "ASSOC_GRAPH_RECALL_ENABLED",
    "ASSOC_CROSS_PROJECT_ENABLED",
)


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def _deterministic_embedding(text: str, _model: str = "") -> List[float]:
    """Return a 1536-dim pseudo-embedding derived from the text hash.

    We use the MD5 of the text as a seed for 1536 floats in [0,1]. This
    keeps the embedding dependent only on the content (reproducible) and
    non-zero (``any(embedding)`` is True, so ``perform_upsert`` does not
    abort on the "valid embedding" check).
    """
    digest = hashlib.md5(text.encode("utf-8")).digest()
    # Expand the 16-byte digest into 1536 floats by cycling with an offset.
    out: List[float] = []
    for i in range(1536):
        b = digest[i % 16]
        out.append(((b + (i * 7) % 251) % 251) / 251.0)
    # Guarantee at least one non-zero entry (all-zero would fail the
    # ``any(embedding)`` check in perform_upsert / _semantic_query).
    if not any(out):
        out[0] = 1.0
    return out


def _strip_volatile(meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a copy of ``meta`` with volatile fields removed."""
    if meta is None:
        return None
    return {k: v for k, v in meta.items() if k not in _VOLATILE_META_KEYS}


def _canonical_json(obj: Any) -> str:
    """Canonical JSON (sorted keys, compact separators) for stable diffs."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# In-memory fake backends
# ---------------------------------------------------------------------------

class _FakePineconeClient:
    """Deterministic in-memory fake for ``PineconeClient``.

    Records every upsert/query call in ``calls`` in the order they happen.
    ``query_vector`` returns a fixed, content-derived result so the
    reranker + RRF pipeline downstream is deterministic.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.store: Dict[str, Dict[str, Any]] = {}

    # Sync API — exactly what MemoryService calls via asyncio.to_thread.
    def initialize(self) -> bool:
        return True

    def upsert_vector(
        self, vector_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> bool:
        self.calls.append(
            {
                "method": "upsert_vector",
                "id": vector_id,
                "metadata": _strip_volatile(metadata),
            }
        )
        self.store[vector_id] = {
            "id": vector_id,
            "values": vector,
            "metadata": dict(metadata),
        }
        return True

    def query_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {
                "method": "query_vector",
                "top_k": top_k,
                "filter": filter,
                "include_values": include_values,
            }
        )
        # Deterministic pseudo-matches: up to 3 stored items, ordered by id.
        matches: List[Dict[str, Any]] = []
        for i, (mid, entry) in enumerate(sorted(self.store.items())[:3]):
            m: Dict[str, Any] = {
                "id": mid,
                "score": round(0.9 - i * 0.1, 3),
                "metadata": dict(entry["metadata"]),
            }
            if include_values:
                m["values"] = entry["values"]
            matches.append(m)
        return matches

    def delete_vector(self, vector_id: str) -> bool:
        self.calls.append({"method": "delete_vector", "id": vector_id})
        self.store.pop(vector_id, None)
        return True

    def check_connection(self) -> bool:
        return True


class _FakeGraphClient:
    """Deterministic in-memory fake for ``GraphClient``."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.store: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def check_connection(self) -> bool:
        return True

    async def upsert_graph_data(
        self,
        node_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self.calls.append(
            {
                "method": "upsert_graph_data",
                "id": node_id,
                "metadata": _strip_volatile(metadata),
            }
        )
        self.store[node_id] = {
            "id": node_id,
            "content": content,
            "metadata": dict(metadata or {}),
        }
        return True

    async def delete_graph_data(self, node_id: str) -> bool:
        self.calls.append({"method": "delete_graph_data", "id": node_id})
        self.store.pop(node_id, None)
        return True

    async def query_graph(
        self, query_text: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {"method": "query_graph", "top_k": top_k}
        )
        out: List[Dict[str, Any]] = []
        for i, (gid, entry) in enumerate(sorted(self.store.items())[:2]):
            out.append(
                {
                    "id": gid,
                    "text": entry["content"],
                    "source": "graph",
                    "score": 0.0,
                    "metadata": dict(entry["metadata"]),
                }
            )
        return out

    async def query_graph_multihop(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        # Deterministic: return nothing. This keeps the snapshot stable
        # and avoids pulling the entity extractor into the baseline.
        return []

    async def link_event_to_session(
        self, event_id: str, session_id: str
    ) -> bool:
        self.calls.append(
            {
                "method": "link_event_to_session",
                "event_id": event_id,
                "session_id": session_id,
            }
        )
        return True


# ---------------------------------------------------------------------------
# Baseline capture + comparison
# ---------------------------------------------------------------------------

def _load_fixtures() -> List[Dict[str, Any]]:
    with FIXTURES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


async def _run_roundtrip() -> Dict[str, Any]:
    """Store every fixture, query every fixture, return the captured trace.

    Returns a dict with:
        - ``upsert_trace``: list of per-fixture upsert signatures
        - ``query_trace``: list of per-fixture query results (as returned
          by ``perform_query``), cleansed of volatile metadata
        - ``pinecone_calls``: all recorded pinecone calls in order
        - ``graph_calls``: all recorded graph calls in order
    """
    from app.services.memory_service import MemoryService  # noqa: E402

    fake_pinecone = _FakePineconeClient()
    fake_graph = _FakeGraphClient()

    # Deterministic monotonic sequence counter.
    seq_counter = {"n": _FAKE_SEQ_START - 1}

    async def fake_next_seq() -> int:
        seq_counter["n"] += 1
        return seq_counter["n"]

    # Deterministic entity extraction — return empty so graph multi-hop
    # path doesn't depend on the real NLP pipeline.
    def fake_extract_entities(text: str) -> List[Any]:
        return []

    with patch(
        "app.services.memory_service.PineconeClient",
        return_value=fake_pinecone,
    ), patch(
        "app.services.memory_service.GraphClient",
        return_value=fake_graph,
    ), patch(
        "app.services.memory_service.get_embedding",
        side_effect=_deterministic_embedding,
    ), patch(
        "app.services.memory_service.extract_entities",
        side_effect=fake_extract_entities,
    ):
        service = MemoryService()
        # Replace the sequence service method with our deterministic one.
        service.sequence_service.next_seq = fake_next_seq  # type: ignore[method-assign]
        # Disable redis timeline — not part of the ASSOC_* contract.
        service.redis_timeline = None
        # Force the _initialized flag — we don't want to run real
        # pinecone/graph/redis init against live infra.
        service._initialized = True

        # Disable rerankers for determinism: the cross-encoder model is
        # large, non-hermetic, and not part of the ASSOC_* contract.
        service.reranker = None
        service._reranker_loaded = False
        service.pinecone_reranker = None

        fixtures = _load_fixtures()

        upsert_trace: List[Dict[str, Any]] = []
        for fx in fixtures:
            # Deep copy so the fixture file is not mutated by
            # _inject_chronology.
            meta = copy.deepcopy(fx["metadata"])
            item_id = await service.perform_upsert(
                content=fx["content"],
                memory_id=fx["fixture_id"],
                metadata=meta,
            )
            upsert_trace.append(
                {
                    "fixture_id": fx["fixture_id"],
                    "returned_id": item_id,
                }
            )

        query_trace: List[Dict[str, Any]] = []
        # Use a small stable set of query strings derived from the
        # fixtures so the snapshot has signal but stays deterministic.
        query_strings = [
            "What is the NovaTrade strategy baseline?",
            "How should integration tests isolate Neo4j writes?",
            "What does ADR-0759 say about associative linking location?",
            "What is the PLAN-0759 rollback safety policy?",
            "How does cross-project scoping work under PLAN-0759?",
        ]
        for q in query_strings:
            results = await service.perform_query(
                q, top_k_vector=5, top_k_final=3
            )
            # Canonicalize: drop volatile metadata keys from every result
            # so the baseline doesn't drift on every run.
            cleaned: List[Dict[str, Any]] = []
            for r in results:
                rc = {k: v for k, v in r.items() if k not in _VOLATILE_RESULT_KEYS}
                if isinstance(rc.get("metadata"), dict):
                    rc["metadata"] = _strip_volatile(rc["metadata"])
                cleaned.append(rc)
            query_trace.append({"query": q, "results": cleaned})

        return {
            "upsert_trace": upsert_trace,
            "query_trace": query_trace,
            "pinecone_calls": fake_pinecone.calls,
            "graph_calls": fake_graph.calls,
        }


def _assert_assoc_flags_all_false() -> None:
    """Hard-fail the test if any *write-path* ASSOC_* flag is True.

    Read-path flags that have been shipped (like ASSOC_GRAPH_RECALL_ENABLED,
    flipped True 2026-04-16 after Phase 6 gate eval) are excluded — they are
    force-disabled inside the test via monkeypatching so the zero-regression
    baseline stays valid.
    """
    shipped_flags = {"ASSOC_GRAPH_RECALL_ENABLED"}
    bad = [
        name for name in ASSOC_FLAG_NAMES
        if name not in shipped_flags and getattr(settings, name) is not False
    ]
    assert not bad, (
        "Zero-regression baseline refuses to run while any ASSOC_* write-path flag "
        f"is True. Offending flags: {bad}. Unset them and rerun."
    )


# ---------------------------------------------------------------------------
# The actual pytest entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase0_zero_regression_baseline() -> None:
    """Pin MemoryService upsert+query behavior under ASSOC_* all-False."""
    # Force shipped read-path flags to False for zero-regression pinning.
    _orig = settings.ASSOC_GRAPH_RECALL_ENABLED
    settings.ASSOC_GRAPH_RECALL_ENABLED = False  # type: ignore[misc]
    try:
        _assert_assoc_flags_all_false()

        trace = await _run_roundtrip()

        # Basic sanity: we got the expected number of upserts and queries.
        assert len(trace["upsert_trace"]) == 10, trace["upsert_trace"]
        assert len(trace["query_trace"]) == 5, trace["query_trace"]

        # Every fixture round-tripped its caller-provided id back.
        for entry in trace["upsert_trace"]:
            assert entry["returned_id"] == entry["fixture_id"], entry

        canonical = _canonical_json(trace)

        if not BASELINE_PATH.exists():
            # First run: write the baseline and pass. A second invocation of
            # this test in the same process tree (or on CI) will then pin it.
            BASELINE_PATH.write_text(canonical + "\n", encoding="utf-8")
            pytest.skip(
                f"Baseline did not exist; wrote fresh baseline to "
                f"{BASELINE_PATH}. Re-run the test to pin it."
            )

        saved = BASELINE_PATH.read_text(encoding="utf-8").rstrip("\n")
        current = canonical

        if saved != current:
            # Produce a compact diff-style error so a reviewer can see what
            # drifted without having to re-run the test locally.
            saved_lines = saved.splitlines()
            current_lines = current.splitlines()
            max_show = 40
            diff_lines: List[str] = []
            for i, (a, b) in enumerate(zip(saved_lines, current_lines)):
                if a != b:
                    diff_lines.append(f"  line {i + 1}:")
                    diff_lines.append(f"    - {a}")
                    diff_lines.append(f"    + {b}")
                    if len(diff_lines) >= max_show:
                        diff_lines.append("  ... (truncated)")
                        break
            if len(saved_lines) != len(current_lines):
                diff_lines.append(
                    f"  length mismatch: saved={len(saved_lines)} "
                    f"current={len(current_lines)}"
                )
            raise AssertionError(
                "Phase 0 zero-regression baseline drifted. If this is "
                "intentional (e.g. you modified the fixture set), delete "
                f"{BASELINE_PATH} and re-run to regenerate. Differences:\n"
                + "\n".join(diff_lines)
            )
    finally:
        settings.ASSOC_GRAPH_RECALL_ENABLED = _orig  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover - local debugging helper
    asyncio.run(_run_roundtrip())
