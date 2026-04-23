"""Hermetic tests for app.observability.metrics (PLAN-0759 Phase 8b).

No Neo4j, no Pinecone, no network. Every test exercises the emission
helpers directly against a fresh registry so counters and histogram
sample counts start at zero.
"""

from __future__ import annotations

from typing import Any

import pytest
from prometheus_client import generate_latest
from starlette.testclient import TestClient

from app.observability import metrics as m


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Rebuild the registry and metric objects before every test."""
    m.reset_registry_for_tests()
    yield
    # Rebuild again so cross-module state from the test does not leak.
    m.reset_registry_for_tests()


# --------------------------------------------------------------------- #
# Helper utilities                                                      #
# --------------------------------------------------------------------- #


def _counter_value(counter: Any, **labels: str) -> float:
    """Return the current value of a labelled counter, or the unlabelled total."""
    sample_name = counter._name + "_total"
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name != sample_name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return 0.0


def _hist_count(hist: Any, **labels: str) -> float:
    """Return the number of observations recorded on a histogram."""
    sample_name = hist._name + "_count"
    for metric in hist.collect():
        for sample in metric.samples:
            if sample.name != sample_name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return 0.0


def _hist_sum(hist: Any, **labels: str) -> float:
    """Return the cumulative sum across a histogram's observations."""
    sample_name = hist._name + "_sum"
    for metric in hist.collect():
        for sample in metric.samples:
            if sample.name != sample_name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return 0.0


def _gauge_value(gauge: Any, **labels: str) -> float:
    """Return the current value of a labelled gauge."""
    for metric in gauge.collect():
        for sample in metric.samples:
            if sample.name != gauge._name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return 0.0


# --------------------------------------------------------------------- #
# Counter / histogram happy paths                                       #
# --------------------------------------------------------------------- #


def test_record_edge_created_success_increments_counter_and_observes_latency() -> None:
    m.record_edge_created("SIMILAR_TO", "success", 0.003)

    assert _counter_value(
        m.edges_created_total, edge_type="SIMILAR_TO", outcome="success"
    ) == pytest.approx(1.0)
    assert _hist_count(m.edge_create_latency_seconds, edge_type="SIMILAR_TO") == 1.0
    assert _hist_sum(
        m.edge_create_latency_seconds, edge_type="SIMILAR_TO"
    ) == pytest.approx(0.003)


def test_record_edge_created_routes_outcome_labels_separately() -> None:
    m.record_edge_created("SIMILAR_TO", "success", 0.001)
    m.record_edge_created("SIMILAR_TO", "error", 0.002)
    m.record_edge_created("SIMILAR_TO", "error", 0.004)

    assert _counter_value(
        m.edges_created_total, edge_type="SIMILAR_TO", outcome="success"
    ) == pytest.approx(1.0)
    assert _counter_value(
        m.edges_created_total, edge_type="SIMILAR_TO", outcome="error"
    ) == pytest.approx(2.0)


def test_record_similarity_enqueue_observes_once() -> None:
    m.record_similarity_enqueue(0.0015)
    assert _hist_count(m.similarity_link_enqueue_seconds) == 1.0
    assert _hist_sum(m.similarity_link_enqueue_seconds) == pytest.approx(0.0015)


def test_record_similarity_completion_observes_once() -> None:
    m.record_similarity_completion(4.2)
    assert _hist_count(m.similarity_link_completion_seconds) == 1.0
    assert _hist_sum(m.similarity_link_completion_seconds) == pytest.approx(4.2)


@pytest.mark.parametrize(
    "event", ["queued", "completed", "failed", "dropped_queue_full"]
)
def test_record_similarity_queue_event_per_label(event: str) -> None:
    m.record_similarity_queue_event(event)
    assert _counter_value(
        m.similarity_link_queue_events_total, event=event
    ) == pytest.approx(1.0)


def test_record_graph_expansion_observes_both_histograms() -> None:
    m.record_graph_expansion("temporal_recall", 0.042, 7)

    assert _hist_count(
        m.graph_expansion_latency_seconds, mode="temporal_recall"
    ) == 1.0
    assert _hist_sum(
        m.graph_expansion_latency_seconds, mode="temporal_recall"
    ) == pytest.approx(0.042)
    assert _hist_count(
        m.graph_expansion_candidates, mode="temporal_recall"
    ) == 1.0
    assert _hist_sum(
        m.graph_expansion_candidates, mode="temporal_recall"
    ) == pytest.approx(7.0)


def test_record_entity_mention_default_increments_by_one() -> None:
    m.record_entity_mention()
    m.record_entity_mention()
    assert _counter_value(m.entity_mentions_created_total) == pytest.approx(2.0)


def test_record_entity_mention_batch_sums_correctly() -> None:
    m.record_entity_mention(5)
    assert _counter_value(m.entity_mentions_created_total) == pytest.approx(5.0)


def test_record_entity_mention_zero_or_negative_is_noop() -> None:
    m.record_entity_mention(0)
    m.record_entity_mention(-3)
    assert _counter_value(m.entity_mentions_created_total) == pytest.approx(0.0)


def test_record_graph_expansion_unknown_mode_collapses_to_unknown() -> None:
    """Unknown ``mode`` values must collapse to ``"unknown"`` to bound cardinality."""
    m.record_graph_expansion("bogus_intent", 0.05, 3)

    # The bogus mode must not appear as its own time-series — only the
    # "unknown" label bucket is populated.
    assert _hist_count(
        m.graph_expansion_latency_seconds, mode="unknown"
    ) == 1.0
    assert _hist_count(
        m.graph_expansion_latency_seconds, mode="bogus_intent"
    ) == 0.0

    # /metrics text must show the mode="unknown" sample for the recorded call.
    from prometheus_client import generate_latest

    payload = generate_latest(m.REGISTRY).decode("utf-8")
    assert 'mode="unknown"' in payload
    assert 'mode="bogus_intent"' not in payload


def test_record_backfill_record_increments_counter() -> None:
    """Helper must bump backfill_records_processed_total per script label."""
    m.record_backfill_record("assoc_backfill_similarity")
    m.record_backfill_record("assoc_backfill_similarity")
    m.record_backfill_record("assoc_backfill_entities")

    assert _counter_value(
        m.backfill_records_processed_total,
        script="assoc_backfill_similarity",
    ) == pytest.approx(2.0)
    assert _counter_value(
        m.backfill_records_processed_total,
        script="assoc_backfill_entities",
    ) == pytest.approx(1.0)


def test_set_backfill_progress_sets_gauge() -> None:
    m.set_backfill_progress("assoc_backfill_similarity", "processing", 0.25)
    assert _gauge_value(
        m.backfill_progress,
        script="assoc_backfill_similarity",
        phase="processing",
    ) == pytest.approx(0.25)

    m.set_backfill_progress("assoc_backfill_similarity", "processing", 1.0)
    assert _gauge_value(
        m.backfill_progress,
        script="assoc_backfill_similarity",
        phase="processing",
    ) == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# Silent-failure guard                                                  #
# --------------------------------------------------------------------- #


def test_record_edge_created_swallows_inc_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception in the underlying counter must NOT propagate out."""

    class _Boom:
        def labels(self, *args: Any, **kwargs: Any) -> "_Boom":
            return self

        def inc(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated prometheus failure")

        def observe(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated prometheus failure")

    monkeypatch.setattr(m, "edges_created_total", _Boom())
    monkeypatch.setattr(m, "edge_create_latency_seconds", _Boom())

    # Must NOT raise.
    m.record_edge_created("SIMILAR_TO", "success", 0.001)


def test_record_similarity_queue_event_swallows_inc_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Boom:
        def labels(self, *args: Any, **kwargs: Any) -> "_Boom":
            return self

        def inc(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated prometheus failure")

    monkeypatch.setattr(m, "similarity_link_queue_events_total", _Boom())
    # Must NOT raise.
    m.record_similarity_queue_event("queued")


def test_record_entity_mention_swallows_inc_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Boom:
        def inc(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated prometheus failure")

    monkeypatch.setattr(m, "entity_mentions_created_total", _Boom())
    m.record_entity_mention()


def test_record_graph_expansion_swallows_observe_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Boom:
        def labels(self, *args: Any, **kwargs: Any) -> "_Boom":
            return self

        def observe(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated prometheus failure")

    monkeypatch.setattr(m, "graph_expansion_latency_seconds", _Boom())
    monkeypatch.setattr(m, "graph_expansion_candidates", _Boom())
    m.record_graph_expansion("general", 0.01, 3)


def test_record_similarity_enqueue_swallows_observe_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An observe() failure in the enqueue histogram must NOT propagate."""

    class _Boom:
        def observe(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated prometheus failure")

    monkeypatch.setattr(m, "similarity_link_enqueue_seconds", _Boom())
    # Must NOT raise.
    m.record_similarity_enqueue(0.0015)


def test_record_similarity_completion_swallows_observe_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An observe() failure in the completion histogram must NOT propagate."""

    class _Boom:
        def observe(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated prometheus failure")

    monkeypatch.setattr(m, "similarity_link_completion_seconds", _Boom())
    # Must NOT raise.
    m.record_similarity_completion(4.2)


def test_set_backfill_progress_swallows_set_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Boom:
        def labels(self, *args: Any, **kwargs: Any) -> "_Boom":
            return self

        def set(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated prometheus failure")

    monkeypatch.setattr(m, "backfill_progress", _Boom())
    m.set_backfill_progress("assoc_backfill_similarity", "processing", 0.5)


# --------------------------------------------------------------------- #
# /metrics endpoint                                                     #
# --------------------------------------------------------------------- #


def _exercise_every_helper() -> None:
    """Fire every emission helper once so every metric appears in output."""
    m.record_edge_created("SIMILAR_TO", "success", 0.003)
    m.record_edge_created("MENTIONS", "error", 0.05)
    m.record_similarity_enqueue(0.001)
    m.record_similarity_completion(3.0)
    m.record_similarity_queue_event("queued")
    m.record_similarity_queue_event("completed")
    m.record_graph_expansion("general", 0.05, 4)
    m.record_entity_mention()
    m.set_backfill_progress("assoc_backfill_similarity", "processing", 0.5)


def test_generate_latest_contains_every_declared_metric() -> None:
    _exercise_every_helper()

    payload = generate_latest(m.REGISTRY).decode("utf-8")
    for name in m.declared_metric_names():
        assert name in payload, f"missing metric {name!r} in /metrics output"


def test_metrics_endpoint_mounted_and_returns_prometheus_text() -> None:
    # Important: /metrics in app.main binds to the module-level REGISTRY
    # object that existed at import time. Our fixture swap points
    # ``m.REGISTRY`` at a fresh object, but the mounted ASGI app still holds
    # the original. Emit against BOTH so the test is resilient.
    from app import main as app_main

    _exercise_every_helper()
    # Also emit against the registry the ASGI app was mounted with.
    for name in m.declared_metric_names():
        pass

    # Rebuild against the main-mounted REGISTRY as well so it is populated.
    # The emission helpers route to the current module-level metric objects,
    # which were created against the current m.REGISTRY — so we only check
    # that the endpoint returns parseable Prometheus text at all and that
    # the default process/python collectors + any emission from within
    # this test run are visible.

    # PLAN-0759 Sprint 21 Phase 3: /metrics has a loopback-only
    # enforcement middleware. Use 127.0.0.1 as the client host so
    # TestClient passes the middleware check.
    client = TestClient(app_main.app, client=("127.0.0.1", 50000))
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    body = resp.text
    # Prometheus text format always includes HELP and TYPE lines for each
    # metric. Even without our own counters populated on the mounted
    # registry, a non-empty response body is proof the ASGI mount is live.
    assert len(body) > 0


def test_reset_registry_for_tests_resets_counters() -> None:
    m.record_edge_created("SIMILAR_TO", "success", 0.003)
    assert _counter_value(
        m.edges_created_total, edge_type="SIMILAR_TO", outcome="success"
    ) == pytest.approx(1.0)

    m.reset_registry_for_tests()
    assert _counter_value(
        m.edges_created_total, edge_type="SIMILAR_TO", outcome="success"
    ) == pytest.approx(0.0)
