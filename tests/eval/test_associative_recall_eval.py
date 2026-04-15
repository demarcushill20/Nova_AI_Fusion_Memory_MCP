"""Hermetic unit tests for ``tests.eval.associative_recall_eval``.

No Anthropic API calls, no real retrieval backend — every test injects
a trivial fake retrieval function and a fake judge with a fixed score
lookup table. The goal is to pin the metric math and the save/load/
compare contracts so Phase 6 can rely on them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Make the Fusion MCP repo root importable so ``tests.eval.*`` resolves
# when pytest is invoked from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.eval.associative_recall_eval import (  # noqa: E402
    GATE_DELTA,
    RELEVANCE_THRESHOLD,
    EvalQuery,
    EvalResult,
    _compute_mrr,
    _compute_recall_at_k,
    compare_baselines,
    load_baseline,
    run_eval,
    save_baseline,
)


# ---------------------------------------------------------------------------
# Toy fixtures — a fake judge and a fake retrieval_fn
# ---------------------------------------------------------------------------


class _FakeJudge:
    """Duck-typed judge: look up a score by candidate id from a table.

    Any id not in the table scores 0.0. The judge records every call so
    tests can assert on the exact (query, candidate) pairs sent.
    """

    def __init__(self, score_by_id: Dict[str, float]) -> None:
        self.score_by_id = dict(score_by_id)
        self.calls: List[Dict[str, Any]] = []

    def judge_relevance(
        self, query: str, candidate_memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        cid = str(candidate_memory.get("id", ""))
        self.calls.append({"query": query, "id": cid})
        return {
            "score": self.score_by_id.get(cid, 0.0),
            "reasoning": "fake",
            "model": "fake-model",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }


def _make_retrieval_fn(
    candidates_by_query: Dict[str, List[Dict[str, Any]]]
):
    """Return a ``(query, k) -> list[dict]`` closure over a table."""

    def retrieval_fn(query: str, k: int) -> List[Dict[str, Any]]:
        return list(candidates_by_query.get(query, []))[:k]

    return retrieval_fn


# ---------------------------------------------------------------------------
# Pin the metric constants
# ---------------------------------------------------------------------------


def test_relevance_threshold_is_point_five() -> None:
    assert RELEVANCE_THRESHOLD == 0.5


def test_gate_delta_is_five_percent() -> None:
    assert GATE_DELTA == 0.05


# ---------------------------------------------------------------------------
# _compute_recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_all_relevant() -> None:
    assert _compute_recall_at_k([0.9, 0.8, 0.7], k=3) == pytest.approx(1.0)


def test_recall_at_k_none_relevant() -> None:
    assert _compute_recall_at_k([0.1, 0.2, 0.3], k=3) == pytest.approx(0.0)


def test_recall_at_k_mixed() -> None:
    # 2 out of 4 >= 0.5
    assert _compute_recall_at_k([0.9, 0.1, 0.6, 0.2], k=4) == pytest.approx(0.5)


def test_recall_at_k_threshold_is_inclusive() -> None:
    # Exactly 0.5 counts as relevant.
    assert _compute_recall_at_k([0.5, 0.5, 0.5], k=3) == pytest.approx(1.0)


def test_recall_at_k_empty_scores() -> None:
    assert _compute_recall_at_k([], k=5) == pytest.approx(0.0)


def test_recall_at_k_clips_to_k() -> None:
    # 10 relevant candidates but k=3 -> denominator is k, not len(scores)
    assert _compute_recall_at_k([0.9] * 10, k=3) == pytest.approx(1.0)


def test_recall_at_k_handles_short_list() -> None:
    # Only 2 scores, both relevant, k=5 -> 2/5 = 0.4
    assert _compute_recall_at_k([0.9, 0.9], k=5) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# _compute_mrr
# ---------------------------------------------------------------------------


def test_mrr_first_position() -> None:
    assert _compute_mrr([0.9, 0.1, 0.1], k=3) == pytest.approx(1.0)


def test_mrr_second_position() -> None:
    assert _compute_mrr([0.1, 0.9, 0.1], k=3) == pytest.approx(0.5)


def test_mrr_third_position() -> None:
    assert _compute_mrr([0.1, 0.1, 0.9], k=3) == pytest.approx(1.0 / 3.0)


def test_mrr_no_relevant() -> None:
    assert _compute_mrr([0.1, 0.2, 0.3], k=3) == pytest.approx(0.0)


def test_mrr_respects_k_window() -> None:
    # First relevant is at position 4, but k=3, so MRR=0.
    assert _compute_mrr([0.1, 0.2, 0.3, 0.9], k=3) == pytest.approx(0.0)


def test_mrr_empty() -> None:
    assert _compute_mrr([], k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_eval end-to-end toy example
# ---------------------------------------------------------------------------


def test_run_eval_computes_recall_and_mrr_on_toy_example() -> None:
    """Three queries, three candidates each, known-good expected metrics."""
    queries = [
        EvalQuery(query_id="q1", query_text="alpha", project="novacore"),
        EvalQuery(query_id="q2", query_text="beta", project="novacore"),
        EvalQuery(query_id="q3", query_text="gamma", project="novacore"),
    ]

    candidates_by_query = {
        # q1: all three relevant in order -> recall=1.0, MRR=1.0
        "alpha": [
            {"id": "a1", "content": "about alpha"},
            {"id": "a2", "content": "alpha again"},
            {"id": "a3", "content": "more alpha"},
        ],
        # q2: only second candidate relevant -> recall=1/3, MRR=0.5
        "beta": [
            {"id": "b1", "content": "off topic"},
            {"id": "b2", "content": "on topic"},
            {"id": "b3", "content": "off topic again"},
        ],
        # q3: none relevant -> recall=0.0, MRR=0.0
        "gamma": [
            {"id": "c1", "content": "unrelated"},
            {"id": "c2", "content": "also unrelated"},
        ],
    }
    retrieval_fn = _make_retrieval_fn(candidates_by_query)

    judge = _FakeJudge(
        score_by_id={
            "a1": 0.9, "a2": 0.8, "a3": 0.7,
            "b1": 0.1, "b2": 0.9, "b3": 0.2,
            "c1": 0.1, "c2": 0.2,
        }
    )

    results = run_eval(queries=queries, retrieval_fn=retrieval_fn, judge=judge, k=3)

    assert len(results) == 3

    # q1: 3/3 relevant at k=3, MRR=1.0
    assert results[0].query_id == "q1"
    assert results[0].candidate_ids == ["a1", "a2", "a3"]
    assert results[0].recall_at_k == pytest.approx(1.0)
    assert results[0].mrr == pytest.approx(1.0)
    assert results[0].judge_scores == [0.9, 0.8, 0.7]

    # q2: 1/3 relevant (b2), MRR=1/2
    assert results[1].query_id == "q2"
    assert results[1].candidate_ids == ["b1", "b2", "b3"]
    assert results[1].recall_at_k == pytest.approx(1.0 / 3.0)
    assert results[1].mrr == pytest.approx(0.5)

    # q3: 0/3 relevant (only 2 candidates returned) — recall=0/3, MRR=0
    assert results[2].query_id == "q3"
    assert results[2].candidate_ids == ["c1", "c2"]
    assert results[2].recall_at_k == pytest.approx(0.0)
    assert results[2].mrr == pytest.approx(0.0)

    # Sanity: judge received exactly 3 + 3 + 2 = 8 calls.
    assert len(judge.calls) == 8


def test_run_eval_empty_retrieval_returns_zero_metrics() -> None:
    """A retrieval_fn that returns nothing yields recall=0.0 and MRR=0.0."""
    queries = [EvalQuery(query_id="q1", query_text="x", project="p")]
    retrieval_fn = _make_retrieval_fn({})
    judge = _FakeJudge({})
    results = run_eval(queries, retrieval_fn, judge, k=5)
    assert len(results) == 1
    assert results[0].candidate_ids == []
    assert results[0].recall_at_k == pytest.approx(0.0)
    assert results[0].mrr == pytest.approx(0.0)
    assert results[0].judge_scores == []
    # Judge was never called.
    assert judge.calls == []


def test_run_eval_clips_retrieval_to_k() -> None:
    """Retrieval that returns more than k candidates is clipped to k."""
    queries = [EvalQuery(query_id="q1", query_text="x", project="p")]
    retrieval_fn = _make_retrieval_fn(
        {"x": [{"id": f"c{i}", "content": "c"} for i in range(20)]}
    )
    judge = _FakeJudge({f"c{i}": 0.9 for i in range(20)})
    results = run_eval(queries, retrieval_fn, judge, k=5)
    assert len(results[0].candidate_ids) == 5
    assert len(results[0].judge_scores) == 5


# ---------------------------------------------------------------------------
# save_baseline / load_baseline
# ---------------------------------------------------------------------------


def test_save_and_load_baseline_round_trips(tmp_path: Path) -> None:
    """A baseline written to disk loads back into identical EvalResult."""
    before = [
        EvalResult(
            query_id="q1",
            candidate_ids=["a1", "a2", "a3"],
            recall_at_k=0.67,
            mrr=0.5,
            judge_scores=[0.9, 0.1, 0.7],
            timestamp="2026-04-13T00:00:00+00:00",
        ),
        EvalResult(
            query_id="q2",
            candidate_ids=["b1"],
            recall_at_k=1.0,
            mrr=1.0,
            judge_scores=[0.95],
            timestamp="2026-04-13T00:00:01+00:00",
        ),
    ]
    path = tmp_path / "baseline.json"
    save_baseline(before, str(path))

    assert path.exists()
    # File is valid JSON and has expected shape.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 2

    loaded = load_baseline(str(path))
    assert len(loaded) == 2
    assert loaded[0].query_id == "q1"
    assert loaded[0].candidate_ids == ["a1", "a2", "a3"]
    assert loaded[0].recall_at_k == pytest.approx(0.67)
    assert loaded[0].mrr == pytest.approx(0.5)
    assert loaded[0].judge_scores == [0.9, 0.1, 0.7]
    assert loaded[1].query_id == "q2"
    assert loaded[1].recall_at_k == pytest.approx(1.0)


def test_save_baseline_creates_parent_directory(tmp_path: Path) -> None:
    """save_baseline mkdir-p's the parent, so nested paths just work."""
    path = tmp_path / "a" / "b" / "c" / "nested.json"
    save_baseline([], str(path))
    assert path.exists()


# ---------------------------------------------------------------------------
# compare_baselines + gate
# ---------------------------------------------------------------------------


def _mk(query_id: str, recall: float, mrr: float) -> EvalResult:
    return EvalResult(
        query_id=query_id,
        candidate_ids=[],
        recall_at_k=recall,
        mrr=mrr,
        judge_scores=[],
        timestamp="2026-04-13T00:00:00+00:00",
    )


def test_compare_baselines_gate_passes_on_recall_delta() -> None:
    """gate_passed=True when mean recall improves by >= 0.05."""
    before = [_mk("q1", 0.40, 0.30), _mk("q2", 0.20, 0.10)]
    after = [_mk("q1", 0.50, 0.30), _mk("q2", 0.30, 0.10)]
    # mean recall delta = (0.10 + 0.10) / 2 = 0.10 >= 0.05
    out = compare_baselines(before, after)
    assert out["recall_delta"] == pytest.approx(0.10)
    assert out["mrr_delta"] == pytest.approx(0.0)
    assert out["gate_passed"] is True
    assert out["num_compared"] == 2


def test_compare_baselines_gate_passes_on_mrr_delta() -> None:
    """gate_passed=True when mean MRR improves by >= 0.05 even if recall is flat."""
    before = [_mk("q1", 0.40, 0.20), _mk("q2", 0.40, 0.20)]
    after = [_mk("q1", 0.40, 0.30), _mk("q2", 0.40, 0.30)]
    # mean mrr delta = 0.10 >= 0.05
    out = compare_baselines(before, after)
    assert out["recall_delta"] == pytest.approx(0.0)
    assert out["mrr_delta"] == pytest.approx(0.10)
    assert out["gate_passed"] is True


def test_compare_baselines_gate_fails_when_both_deltas_below_threshold() -> None:
    """Neither delta reaches 0.05 -> gate_passed=False."""
    before = [_mk("q1", 0.40, 0.20), _mk("q2", 0.40, 0.20)]
    after = [_mk("q1", 0.42, 0.22), _mk("q2", 0.42, 0.22)]
    out = compare_baselines(before, after)
    assert out["recall_delta"] == pytest.approx(0.02)
    assert out["mrr_delta"] == pytest.approx(0.02)
    assert out["gate_passed"] is False


def test_compare_baselines_boundary_recall_delta_at_gate() -> None:
    """A delta of exactly ``GATE_DELTA`` must satisfy the gate (>= not >).

    Uses the ``GATE_DELTA`` constant directly rather than a float literal
    so the arithmetic is exact to IEEE 754: ``(0.0 + GATE_DELTA) - 0.0``
    trivially equals ``GATE_DELTA`` and the comparison is unambiguous.
    """
    before = [_mk("q1", 0.0, 0.0)]
    after = [_mk("q1", GATE_DELTA, 0.0)]
    out = compare_baselines(before, after)
    assert out["recall_delta"] == pytest.approx(GATE_DELTA)
    assert out["gate_passed"] is True


def test_compare_baselines_just_below_gate_fails() -> None:
    """A delta strictly below ``GATE_DELTA`` must not pass the gate."""
    before = [_mk("q1", 0.0, 0.0)]
    after = [_mk("q1", GATE_DELTA / 2.0, 0.0)]
    out = compare_baselines(before, after)
    assert out["gate_passed"] is False


def test_compare_baselines_skips_missing_query_ids() -> None:
    """Queries present on only one side are reported in skipped_query_ids."""
    before = [_mk("q1", 0.4, 0.2), _mk("q2", 0.3, 0.1)]
    after = [_mk("q1", 0.5, 0.3), _mk("q3", 0.9, 0.9)]
    out = compare_baselines(before, after)
    # Only q1 is in both.
    assert out["num_compared"] == 1
    assert out["skipped_query_ids"]["only_before"] == ["q2"]
    assert out["skipped_query_ids"]["only_after"] == ["q3"]
    # Deltas from q1 alone: recall +0.10, mrr +0.10
    assert out["recall_delta"] == pytest.approx(0.10)
    assert out["mrr_delta"] == pytest.approx(0.10)
    assert out["gate_passed"] is True


def test_compare_baselines_no_overlap_returns_not_passed() -> None:
    """Zero overlap between baselines -> gate_passed=False, zero deltas."""
    before = [_mk("q1", 0.4, 0.2)]
    after = [_mk("q2", 0.9, 0.9)]
    out = compare_baselines(before, after)
    assert out["num_compared"] == 0
    assert out["gate_passed"] is False
    assert out["recall_delta"] == pytest.approx(0.0)
    assert out["mrr_delta"] == pytest.approx(0.0)
