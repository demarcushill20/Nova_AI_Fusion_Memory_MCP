"""Phase 6 eval runner — PLAN-0759 hard gate.

Runs the associative-recall evaluation end-to-end:

  1. Load query fixture from tests/eval/fixtures/phase6_queries.json
  2. Bootstrap MemoryService against live Neo4j + Pinecone
  3. For each query, call perform_query(expand_graph=False) for baseline
  4. For each query, call perform_query(expand_graph=True)  for expanded
  5. Score both runs with ClaudeCLIJudge (opus-4-6, effort=high, batched)
  6. Compute recall@10 and MRR per query and in aggregate
  7. Emit tests/eval/results/phase6_eval.json with deltas and gate verdict

Gate (PLAN-0759 Phase 6): passes iff
    recall_delta >= +0.05  OR  mrr_delta >= +0.05

Usage
-----
Requires ``ASSOC_GRAPH_RECALL_ENABLED=true`` to be set for the expanded
run — the runner sets it via ``settings`` directly so this does not leak
to other processes.

Run from the Fusion MCP repo root::

    NEO4J_URI=bolt://localhost:7687 python -m tests.eval.run_phase6_eval
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.services.memory_service import MemoryService

from .associative_recall_eval import (
    EvalQuery,
    EvalResult,
    _compute_mrr,
    _compute_recall_at_k,
    compare_baselines,
    save_baseline,
)
from .cli_judge import ClaudeCLIJudge


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "eval" / "fixtures" / "phase6_queries.json"
BASELINE_DIR = REPO_ROOT / "tests" / "eval" / "baselines"
RESULTS_DIR = REPO_ROOT / "tests" / "eval" / "results"
BASELINE_PATH = BASELINE_DIR / "semantic_only_2026-04-15.json"
EXPANDED_PATH = RESULTS_DIR / "phase6_expanded_2026-04-15.json"
REPORT_PATH = RESULTS_DIR / "phase6_eval.json"

K = 10


def _load_queries() -> List[EvalQuery]:
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return [
        EvalQuery(
            query_id=entry["query_id"],
            query_text=entry["query_text"],
            project=entry["project"],
        )
        for entry in raw
    ]


def _shape_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a perform_query result row into the {id, content} shape
    that the judge rubric expects.

    perform_query returns rows with keys like ``id``/``entity_id``,
    ``text``/``content``, plus metadata. We prefer ``entity_id`` as the
    stable id and ``text`` as the primary content field (that's what the
    Neo4j :base nodes actually store).
    """
    cid = (
        item.get("entity_id")
        or item.get("id")
        or item.get("memory_id")
        or ""
    )
    text = item.get("text") or item.get("content") or ""
    if not text:
        meta = item.get("metadata") or {}
        text = meta.get("text") or meta.get("content") or ""
    return {"id": str(cid), "content": str(text)}


async def _retrieve(
    svc: MemoryService, query_text: str, k: int, expand_graph: bool
) -> List[Dict[str, Any]]:
    try:
        results = await svc.perform_query(
            query_text=query_text,
            top_k_vector=max(50, k * 3),
            top_k_final=k,
            expand_graph=expand_graph,
        )
    except Exception as exc:
        print(f"  [retrieve] perform_query crashed: {type(exc).__name__}: {exc}")
        return []
    # Defensive: filter out any non-dict entries returned by upstream
    # retrieval paths (a pre-existing Fusion Memory bug in the SESSION
    # routing fallback can place a None here — we treat it as no result
    # rather than let it crash the whole eval).
    shaped: List[Dict[str, Any]] = []
    for r in (results or []):
        if not isinstance(r, dict):
            continue
        shaped.append(_shape_candidate(r))
        if len(shaped) >= k:
            break
    return shaped


async def _run_one_pass(
    svc: MemoryService,
    queries: List[EvalQuery],
    judge: ClaudeCLIJudge,
    expand_graph: bool,
    label: str,
) -> List[EvalResult]:
    """Run retrieval + judge for every query in order and print progress."""
    results: List[EvalResult] = []
    total = len(queries)
    for i, q in enumerate(queries, start=1):
        candidates = await _retrieve(svc, q.query_text, K, expand_graph)
        if not candidates:
            print(f"  [{label}] {i}/{total} {q.query_id}: 0 candidates (skipping judge)")
            results.append(
                EvalResult(
                    query_id=q.query_id,
                    candidate_ids=[],
                    recall_at_k=0.0,
                    mrr=0.0,
                    judge_scores=[],
                )
            )
            continue

        judgements = judge.judge_batch(q.query_text, candidates)
        scores = [float(j["score"]) for j in judgements]
        recall = _compute_recall_at_k(scores, K)
        mrr = _compute_mrr(scores, K)

        relevant = sum(1 for s in scores if s >= 0.5)
        print(
            f"  [{label}] {i}/{total} {q.query_id}: "
            f"relevant={relevant}/{len(scores)} recall@{K}={recall:.2f} mrr={mrr:.2f}"
        )

        results.append(
            EvalResult(
                query_id=q.query_id,
                candidate_ids=[c["id"] for c in candidates],
                recall_at_k=recall,
                mrr=mrr,
                judge_scores=scores,
            )
        )
    return results


def _aggregate(results: List[EvalResult]) -> Dict[str, float]:
    if not results:
        return {"recall_at_k": 0.0, "mrr": 0.0}
    recall = sum(r.recall_at_k for r in results) / len(results)
    mrr = sum(r.mrr for r in results) / len(results)
    return {"recall_at_k": recall, "mrr": mrr}


async def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    queries = _load_queries()
    print(f"Loaded {len(queries)} queries from {FIXTURE_PATH}")

    judge = ClaudeCLIJudge()
    print(f"Judge: claude -p --model {judge.config.model} --effort {judge.config.effort}")

    print("Bootstrapping MemoryService...")
    svc = MemoryService()
    ok = await svc.initialize()
    if not ok:
        print("MemoryService initialization failed.", file=sys.stderr)
        return 2
    print("MemoryService ready.")

    # ------------------------------------------------------------------
    # Pass 1: baseline (expand_graph=False) — this path must be
    # byte-identical to current pre-plan behavior per the v2 recall merge
    # contract.
    # ------------------------------------------------------------------
    print("\n=== Pass 1/2: semantic-only baseline ===")
    settings.ASSOC_GRAPH_RECALL_ENABLED = False
    baseline_results = await _run_one_pass(
        svc, queries, judge, expand_graph=False, label="BASELINE"
    )
    save_baseline(baseline_results, str(BASELINE_PATH))
    baseline_agg = _aggregate(baseline_results)
    print(
        f"Baseline aggregate: recall@{K}={baseline_agg['recall_at_k']:.4f} "
        f"mrr={baseline_agg['mrr']:.4f}"
    )
    print(f"Baseline written to {BASELINE_PATH}")

    # ------------------------------------------------------------------
    # Pass 2: expanded (expand_graph=True, ASSOC_GRAPH_RECALL_ENABLED=True)
    # ------------------------------------------------------------------
    print("\n=== Pass 2/2: graph-expanded ===")
    settings.ASSOC_GRAPH_RECALL_ENABLED = True
    expanded_results = await _run_one_pass(
        svc, queries, judge, expand_graph=True, label="EXPANDED"
    )
    save_baseline(expanded_results, str(EXPANDED_PATH))
    expanded_agg = _aggregate(expanded_results)
    print(
        f"Expanded aggregate: recall@{K}={expanded_agg['recall_at_k']:.4f} "
        f"mrr={expanded_agg['mrr']:.4f}"
    )
    print(f"Expanded written to {EXPANDED_PATH}")

    # ------------------------------------------------------------------
    # Gate verdict
    # ------------------------------------------------------------------
    comparison = compare_baselines(baseline_results, expanded_results)
    report = {
        "plan_id": "PLAN-0759-assoc-linking",
        "phase": 6,
        "run_id": "phase6-eval-2026-04-15",
        "k": K,
        "judge": {
            "backend": "claude_cli",
            "model": judge.config.model,
            "effort": judge.config.effort,
        },
        "num_queries": len(queries),
        "baseline_aggregate": baseline_agg,
        "expanded_aggregate": expanded_agg,
        "recall_delta": comparison["recall_delta"],
        "mrr_delta": comparison["mrr_delta"],
        "gate_passed": comparison["gate_passed"],
        "gate_rule": "recall_delta >= 0.05 OR mrr_delta >= 0.05",
        "num_compared": comparison["num_compared"],
        "skipped_query_ids": comparison["skipped_query_ids"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_query": [
            {
                "query_id": q.query_id,
                "query_text": q.query_text,
                "project": q.project,
                "baseline_recall_at_k": b.recall_at_k,
                "baseline_mrr": b.mrr,
                "expanded_recall_at_k": e.recall_at_k,
                "expanded_mrr": e.mrr,
                "recall_delta": e.recall_at_k - b.recall_at_k,
                "mrr_delta": e.mrr - b.mrr,
            }
            for q, b, e in zip(queries, baseline_results, expanded_results)
        ],
    }
    REPORT_PATH.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== GATE VERDICT ===")
    print(f"recall_delta = {comparison['recall_delta']:+.4f}")
    print(f"mrr_delta    = {comparison['mrr_delta']:+.4f}")
    print(f"gate_passed  = {comparison['gate_passed']}")
    print(f"report       = {REPORT_PATH}")
    return 0 if comparison["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
