"""Associative-recall eval runner (PLAN-0759 / Step 0.4 — scaffolding).

This module is the Sprint 3 scaffolding for the Phase 6 associative-recall
evaluation described in ADR-0759 and PLAN-0759 Step 0.4. It wires up the
LLM-as-judge from ``llm_judge.py`` to a pluggable retrieval function,
computes ``recall@k`` and ``MRR`` for each query, and provides save /
load / compare helpers so Phase 6 can run a before/after A-B test and
gate on the rubric: ``recall@10 ≥ +5pp OR MRR ≥ +0.05``.

Scope of this sprint
--------------------
- **No initial query set is built here.** ``run_eval`` takes
  ``queries`` as an injected argument so the harness stays decoupled
  from any specific query list.
- **No live judge calls are made during the Sprint 3 test run.** The
  unit tests in ``test_associative_recall_eval.py`` pass a mock
  ``LLMJudge`` and a mock retrieval function so every code path is
  exercised hermetically.
- **No baseline is written on this sprint.** The ``save_baseline`` and
  ``load_baseline`` helpers exist but are intentionally unused in
  production until Phase 6.

Gate rubric
-----------
``compare_baselines`` returns ``gate_passed=True`` iff
``recall_delta ≥ +0.05`` OR ``mrr_delta ≥ +0.05``. The 5-percentage-point
threshold matches the PLAN-0759 Phase 6 acceptance criterion.

Relevance threshold
-------------------
A candidate is considered relevant iff its judge score is ``≥ 0.5``.
``recall@k`` is the fraction of top-k candidates meeting that threshold;
``MRR`` is the reciprocal rank of the first top-k candidate meeting
that threshold (``0.0`` if none do).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .llm_judge import LLMJudge


# A candidate is "relevant" iff judge score >= this threshold.
RELEVANCE_THRESHOLD = 0.5

# PLAN-0759 Phase 6 gate: recall_delta >= +0.05 OR mrr_delta >= +0.05.
GATE_DELTA = 0.05


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """A single evaluation query.

    Attributes:
        query_id: Stable identifier for the query (used as a primary
            key in baseline snapshots so reruns can be diffed).
        query_text: The actual query string passed to the retrieval
            function and the judge prompt.
        project: Project scope for the query. PLAN-0759 defaults to
            single-project isolation, so the retrieval function is
            expected to honor this scope.
    """

    query_id: str
    query_text: str
    project: str


@dataclass
class EvalResult:
    """Result of evaluating a single query against a retrieval function.

    Attributes:
        query_id: Matches ``EvalQuery.query_id``.
        candidate_ids: The top-k candidate ids returned by the retrieval
            function, in their original rank order.
        recall_at_k: Fraction of the top-k candidates with
            judge_score >= ``RELEVANCE_THRESHOLD``.
        mrr: Reciprocal rank of the first relevant candidate, or ``0.0``
            if none of the top-k are relevant.
        judge_scores: Per-candidate judge scores in rank order.
        timestamp: ISO-8601 UTC timestamp at which the result was
            computed (not the moment the judge scored any individual
            pair — that's in the judge response records).
    """

    query_id: str
    candidate_ids: List[str]
    recall_at_k: float
    mrr: float
    judge_scores: List[float]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _compute_recall_at_k(judge_scores: Sequence[float], k: int) -> float:
    """Fraction of top-k scores meeting ``RELEVANCE_THRESHOLD``.

    Defined as ``len(relevant_in_topk) / k`` so an empty result list
    (retrieval returned nothing) scores 0.0, and a list shorter than k
    is treated as "missing candidates are irrelevant".
    """
    if k <= 0:
        return 0.0
    topk = list(judge_scores)[:k]
    relevant = sum(1 for s in topk if s >= RELEVANCE_THRESHOLD)
    return relevant / k


def _compute_mrr(judge_scores: Sequence[float], k: int) -> float:
    """Reciprocal rank of the first top-k score meeting the threshold."""
    if k <= 0:
        return 0.0
    for i, s in enumerate(list(judge_scores)[:k]):
        if s >= RELEVANCE_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Run / save / load / compare
# ---------------------------------------------------------------------------

def run_eval(
    queries: List[EvalQuery],
    retrieval_fn: Callable[[str, int], List[Dict[str, Any]]],
    judge: LLMJudge,
    k: int = 10,
) -> List[EvalResult]:
    """Run each query through retrieval + judge and collect the metrics.

    Args:
        queries: The set of queries to evaluate. Callers are expected to
            pass a stable, deterministic list — reordering affects
            nothing in the metrics but makes diffs harder to read.
        retrieval_fn: A callable ``(query_text, k) -> list[dict]`` that
            returns up to ``k`` candidate memories in rank order. Each
            candidate dict must carry an ``id`` key; ``content`` or
            ``text`` is used as the judge payload.
        judge: An ``LLMJudge`` instance (or any duck-typed object with
            a ``judge_relevance(query, candidate) -> dict`` method).
        k: Top-k cutoff for both ``recall@k`` and the MRR horizon.

    Returns:
        A list of ``EvalResult``, one per query, in the same order as
        the input ``queries``.
    """
    results: List[EvalResult] = []
    for q in queries:
        candidates = retrieval_fn(q.query_text, k) or []
        # Clip to k in case the retrieval_fn returned more — the metrics
        # are defined over the top-k window only.
        candidates = candidates[:k]

        candidate_ids: List[str] = []
        judge_scores: List[float] = []
        for cand in candidates:
            cid = cand.get("id", "")
            candidate_ids.append(str(cid))
            judgement = judge.judge_relevance(q.query_text, cand)
            score = float(judgement.get("score", 0.0))
            judge_scores.append(score)

        results.append(
            EvalResult(
                query_id=q.query_id,
                candidate_ids=candidate_ids,
                recall_at_k=_compute_recall_at_k(judge_scores, k),
                mrr=_compute_mrr(judge_scores, k),
                judge_scores=judge_scores,
            )
        )
    return results


def save_baseline(results: List[EvalResult], path: str) -> None:
    """Serialize a list of ``EvalResult`` to a JSON file.

    The file is written with sorted keys and 2-space indentation so
    future diffs are stable and reviewable.
    """
    payload = [asdict(r) for r in results]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")


def load_baseline(path: str) -> List[EvalResult]:
    """Deserialize a baseline file back into a list of ``EvalResult``."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out: List[EvalResult] = []
    for entry in payload:
        out.append(
            EvalResult(
                query_id=entry["query_id"],
                candidate_ids=list(entry["candidate_ids"]),
                recall_at_k=float(entry["recall_at_k"]),
                mrr=float(entry["mrr"]),
                judge_scores=[float(s) for s in entry["judge_scores"]],
                timestamp=entry.get("timestamp", ""),
            )
        )
    return out


def compare_baselines(
    before: List[EvalResult],
    after: List[EvalResult],
) -> Dict[str, Any]:
    """Compute mean deltas between two baselines and evaluate the gate.

    The deltas are averaged over the set of ``query_id``s that appear in
    BOTH baselines — query_ids that only exist in one side are skipped
    with a note in the returned dict, so drift caused by the query set
    changing is visible rather than hidden.

    Returns:
        A dict with keys:
            - ``recall_delta``: mean(after.recall - before.recall)
            - ``mrr_delta``: mean(after.mrr - before.mrr)
            - ``gate_passed``: True iff recall_delta >= GATE_DELTA
              OR mrr_delta >= GATE_DELTA
            - ``num_compared``: number of query_ids compared
            - ``skipped_query_ids``: query_ids present in only one side
    """
    before_map = {r.query_id: r for r in before}
    after_map = {r.query_id: r for r in after}

    common_ids = sorted(set(before_map) & set(after_map))
    only_before = sorted(set(before_map) - set(after_map))
    only_after = sorted(set(after_map) - set(before_map))

    if not common_ids:
        return {
            "recall_delta": 0.0,
            "mrr_delta": 0.0,
            "gate_passed": False,
            "num_compared": 0,
            "skipped_query_ids": {
                "only_before": only_before,
                "only_after": only_after,
            },
        }

    recall_deltas = [
        after_map[qid].recall_at_k - before_map[qid].recall_at_k
        for qid in common_ids
    ]
    mrr_deltas = [
        after_map[qid].mrr - before_map[qid].mrr for qid in common_ids
    ]

    recall_delta = sum(recall_deltas) / len(recall_deltas)
    mrr_delta = sum(mrr_deltas) / len(mrr_deltas)
    gate_passed = (recall_delta >= GATE_DELTA) or (mrr_delta >= GATE_DELTA)

    return {
        "recall_delta": recall_delta,
        "mrr_delta": mrr_delta,
        "gate_passed": gate_passed,
        "num_compared": len(common_ids),
        "skipped_query_ids": {
            "only_before": only_before,
            "only_after": only_after,
        },
    }
