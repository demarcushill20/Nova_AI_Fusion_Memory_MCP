"""Measure MemoryService.perform_upsert orchestration overhead (PLAN-0759).

Captures the Phase 2 latency-gate baseline. Every backend is mocked so the
metric is orchestration cost only. Usage: ``python3 measure_perform_upsert_latency.py``.
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "phase0_regression_memories.json"
DEFAULT_OUT = PROJECT_ROOT / "tests" / "eval" / "baselines" / "latency_baseline_2026-04-13.json"
WARMUP, ITERS = 3, 30


def _fake_embedding(text: str, _model: str = "") -> list[float]:
    d = hashlib.md5(text.encode("utf-8")).digest()
    out = [((d[i % 16] + i * 7) % 251) / 251.0 for i in range(1536)]
    if not any(out):
        out[0] = 1.0
    return out


def _sync_noop(*a, **kw): return True
async def _async_noop(*a, **kw): return True


_FakePinecone = type("_FakePinecone", (), {
    "initialize": _sync_noop, "check_connection": _sync_noop,
    "upsert_vector": _sync_noop, "delete_vector": _sync_noop,
    "query_vector": lambda *a, **kw: [],
})
_FakeGraph = type("_FakeGraph", (), {
    "initialize": _async_noop, "close": _async_noop, "check_connection": _async_noop,
    "upsert_graph_data": _async_noop, "delete_graph_data": _async_noop,
    "link_event_to_session": _async_noop,
})


def _pctile(sorted_samples: list[float], p: float) -> float:
    if not sorted_samples:
        return 0.0
    k = (len(sorted_samples) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_samples) - 1)
    return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * (k - lo)


async def _measure(iterations: int, warmup: int) -> dict[str, Any]:
    from app.services.memory_service import MemoryService

    fixture = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))[0]
    seq = {"n": 1000}

    async def _next_seq() -> int:
        seq["n"] += 1
        return seq["n"]

    with patch("app.services.memory_service.PineconeClient", return_value=_FakePinecone()), \
         patch("app.services.memory_service.GraphClient", return_value=_FakeGraph()), \
         patch("app.services.memory_service.get_embedding", side_effect=_fake_embedding), \
         patch("app.services.memory_service.extract_entities", side_effect=lambda _t: []):
        svc = MemoryService()
        svc.sequence_service.next_seq = _next_seq  # type: ignore[method-assign]
        svc.redis_timeline = None
        svc._initialized = True
        svc.reranker = None
        svc._reranker_loaded = False
        svc.pinecone_reranker = None

        for _ in range(warmup):
            await svc.perform_upsert(fixture["content"], fixture["fixture_id"], dict(fixture["metadata"]))

        samples_ns: list[int] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            await svc.perform_upsert(fixture["content"], fixture["fixture_id"], dict(fixture["metadata"]))
            samples_ns.append(time.perf_counter_ns() - t0)

    samples = sorted(ns / 1_000_000 for ns in samples_ns)
    return {
        "measured_at": datetime.now(tz=timezone.utc).isoformat(),
        "measurement_type": "perform_upsert_orchestration_overhead",
        "iterations": iterations, "warmup_iterations": warmup,
        "fixture_memory_id": fixture["fixture_id"],
        "mocks_used": ["PineconeClient", "GraphClient", "get_embedding", "extract_entities", "SequenceService.next_seq"],
        "p50_ms": _pctile(samples, 0.50), "p95_ms": _pctile(samples, 0.95), "p99_ms": _pctile(samples, 0.99),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": samples[0], "max_ms": samples[-1], "all_samples_ms": samples,
        "phase2_gate": "Phase 2 p95 overhead must not regress by more than 10% over this baseline",
        "notes": "perform_upsert() under ASSOC_* all-False with every backend mocked; isolates orchestration cost.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--iterations", type=int, default=ITERS)
    p.add_argument("--warmup", type=int, default=WARMUP)
    args = p.parse_args()
    result = asyncio.run(_measure(args.iterations, args.warmup))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote baseline to {args.out}")
    print(f"p50={result['p50_ms']:.3f}ms  p95={result['p95_ms']:.3f}ms  p99={result['p99_ms']:.3f}ms  mean={result['mean_ms']:.3f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
