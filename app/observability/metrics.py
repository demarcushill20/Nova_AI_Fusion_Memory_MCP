"""Prometheus metric declarations and emission helpers for PLAN-0759 Phase 8.

Overview
--------

This module owns every Prometheus metric exported by the Fusion Memory MCP.
All metrics are registered against a single module-level
:data:`REGISTRY` (a :class:`prometheus_client.CollectorRegistry`) so tests
can reset state between runs without touching the default global registry.

Emission contract
-----------------

Every helper defined here wraps its :meth:`Counter.inc` or
:meth:`Histogram.observe` call in a ``try / except Exception`` envelope and
logs a warning on failure. Metric emission is best-effort — it must never
break the write path. This is the silent-failure guard the Phase 8
validator flagged (see v2 plan §8 amendment B).

Async-safety note
-----------------

``prometheus-client`` is thread-safe via its global lock, but
:meth:`Histogram.time` is a synchronous context manager and is **not** safe
inside ``async def`` coroutines. Hot paths in this project always pass a
pre-computed ``time.perf_counter()`` delta into :meth:`Histogram.observe`
via the helpers below — direct use of ``with hist.time():`` inside async
code is forbidden.

Metric names and labels match v2-plan-spec Phase 8.3.
"""

from __future__ import annotations

import logging
from typing import Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Bucket definitions                                                    #
# --------------------------------------------------------------------- #

_EDGE_CREATE_BUCKETS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
)
# SLO: p95 enqueue < 5 ms.
_SIMILARITY_ENQUEUE_BUCKETS: tuple[float, ...] = (
    0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1,
)
# SLO: p95 background completion < 10 s.
_SIMILARITY_COMPLETION_BUCKETS: tuple[float, ...] = (
    0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0,
)
# SLO: p95 graph expansion (2-hop) < 200 ms.
_GRAPH_EXPANSION_LATENCY_BUCKETS: tuple[float, ...] = (
    0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5,
)
_GRAPH_EXPANSION_CANDIDATE_BUCKETS: tuple[float, ...] = (
    0, 1, 2, 5, 10, 20, 50, 100,
)


# --------------------------------------------------------------------- #
# Registry + metric objects (rebuildable for test isolation)            #
# --------------------------------------------------------------------- #

REGISTRY: CollectorRegistry
edges_created_total: Counter
edge_create_latency_seconds: Histogram
similarity_link_enqueue_seconds: Histogram
similarity_link_completion_seconds: Histogram
similarity_link_queue_events_total: Counter
graph_expansion_latency_seconds: Histogram
graph_expansion_candidates: Histogram
entity_mentions_created_total: Counter
backfill_progress: Gauge
backfill_records_processed_total: Counter


# Allow-list for the ``mode`` label on graph_expansion_* metrics. Any value
# outside this set is collapsed to ``"unknown"`` in record_graph_expansion()
# to bound Prometheus label cardinality — unvalidated upstream ``intent``
# strings could otherwise spawn a new time-series per distinct value.
#
# NOTE: this set MUST stay in sync with
# ``app.services.associations.associative_recall.INTENT_EDGE_FILTER.keys()``
# plus the literal ``"general"`` default. We hard-code it here (instead of
# importing INTENT_EDGE_FILTER) so the metrics module has no dependency on
# the recall module. If a new recall intent is added upstream, update this
# set too.
_ALLOWED_EXPANSION_MODES: frozenset[str] = frozenset(
    {
        "temporal_recall",
        "procedural_recall",
        "decision_recall",
        "debug_recall",
        "entity_recall",
        "provenance_recall",
        "general",
    }
)


def _build_registry_and_metrics() -> None:
    """Create a fresh registry and (re)bind every metric to it.

    Called once at import time and by :func:`reset_registry_for_tests`.
    """
    global REGISTRY
    global edges_created_total
    global edge_create_latency_seconds
    global similarity_link_enqueue_seconds
    global similarity_link_completion_seconds
    global similarity_link_queue_events_total
    global graph_expansion_latency_seconds
    global graph_expansion_candidates
    global entity_mentions_created_total
    global backfill_progress
    global backfill_records_processed_total

    REGISTRY = CollectorRegistry()

    edges_created_total = Counter(
        "edges_created_total",
        "Associative edges written to Neo4j, by edge type and outcome.",
        labelnames=("edge_type", "outcome"),
        registry=REGISTRY,
    )
    edge_create_latency_seconds = Histogram(
        "edge_create_latency_seconds",
        "Wall-clock latency of a single edge MERGE, by edge type.",
        labelnames=("edge_type",),
        buckets=_EDGE_CREATE_BUCKETS,
        registry=REGISTRY,
    )
    similarity_link_enqueue_seconds = Histogram(
        "similarity_link_enqueue_seconds",
        "Time spent inside SimilarityLinker.enqueue_link before returning.",
        buckets=_SIMILARITY_ENQUEUE_BUCKETS,
        registry=REGISTRY,
    )
    similarity_link_completion_seconds = Histogram(
        "similarity_link_completion_seconds",
        "End-to-end duration of a background similarity-link task.",
        buckets=_SIMILARITY_COMPLETION_BUCKETS,
        registry=REGISTRY,
    )
    similarity_link_queue_events_total = Counter(
        "similarity_link_queue_events_total",
        "Lifecycle events for the similarity linker queue.",
        labelnames=("event",),
        registry=REGISTRY,
    )
    graph_expansion_latency_seconds = Histogram(
        "graph_expansion_latency_seconds",
        "AssociativeRecall.expand wall-clock latency, by intent mode.",
        labelnames=("mode",),
        buckets=_GRAPH_EXPANSION_LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    graph_expansion_candidates = Histogram(
        "graph_expansion_candidates",
        "Number of expansion candidates returned per expand() call.",
        labelnames=("mode",),
        buckets=_GRAPH_EXPANSION_CANDIDATE_BUCKETS,
        registry=REGISTRY,
    )
    entity_mentions_created_total = Counter(
        "entity_mentions_created_total",
        "MENTIONS edges written by the entity linker.",
        registry=REGISTRY,
    )
    backfill_progress = Gauge(
        "backfill_progress",
        "Fractional progress (0.0–1.0) of a backfill script run.",
        labelnames=("script", "phase"),
        registry=REGISTRY,
    )
    backfill_records_processed_total = Counter(
        "backfill_records_processed_total",
        (
            "Records processed by a backfill script, emitted at batch "
            "boundaries. Rises monotonically during a full-graph backfill "
            "(where max_total is unset and backfill_progress stays at 0). "
            "Use rate(backfill_records_processed_total[1m]) for liveness."
        ),
        labelnames=("script",),
        registry=REGISTRY,
    )


_build_registry_and_metrics()


def reset_registry_for_tests() -> None:
    """Rebuild the module-level registry and metrics.

    Hermetic tests call this in a function-scope fixture so each test sees
    zeroed counters and empty histograms. Production code must not call
    this.
    """
    _build_registry_and_metrics()


def declared_metric_names() -> Iterable[str]:
    """Return the canonical metric names exported by this module.

    Used by the hermetic smoke test to assert every declared metric shows
    up in ``/metrics`` after at least one emission.
    """
    return (
        "edges_created_total",
        "edge_create_latency_seconds",
        "similarity_link_enqueue_seconds",
        "similarity_link_completion_seconds",
        "similarity_link_queue_events_total",
        "graph_expansion_latency_seconds",
        "graph_expansion_candidates",
        "entity_mentions_created_total",
        "backfill_progress",
        "backfill_records_processed_total",
    )


# --------------------------------------------------------------------- #
# Emission helpers — every caller in the codebase routes through here.  #
# --------------------------------------------------------------------- #


def record_edge_created(edge_type: str, outcome: str, latency_s: float) -> None:
    """Record a single edge-write outcome + its latency.

    ``outcome`` must be ``"success"`` or ``"error"``. The helper swallows
    any exception raised while emitting metrics.
    """
    try:
        edges_created_total.labels(edge_type=edge_type, outcome=outcome).inc()
        edge_create_latency_seconds.labels(edge_type=edge_type).observe(
            float(latency_s)
        )
    except Exception as exc:  # noqa: BLE001 — metrics must never break writes
        logger.warning(
            "metrics.record_edge_created failed: edge_type=%s outcome=%s err=%s",
            edge_type,
            outcome,
            exc,
        )


def record_similarity_enqueue(latency_s: float) -> None:
    """Record the wall-clock cost of SimilarityLinker.enqueue_link."""
    try:
        similarity_link_enqueue_seconds.observe(float(latency_s))
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics.record_similarity_enqueue failed: err=%s", exc)


def record_similarity_completion(latency_s: float) -> None:
    """Record the end-to-end duration of a background similarity link."""
    try:
        similarity_link_completion_seconds.observe(float(latency_s))
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics.record_similarity_completion failed: err=%s", exc)


def record_similarity_queue_event(event: str) -> None:
    """Record a lifecycle event on the similarity linker queue.

    ``event`` is one of ``"queued"``, ``"completed"``, ``"failed"``, or
    ``"dropped_queue_full"``.
    """
    try:
        similarity_link_queue_events_total.labels(event=event).inc()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "metrics.record_similarity_queue_event failed: event=%s err=%s",
            event,
            exc,
        )


def record_graph_expansion(
    mode: str, latency_s: float, candidate_count: int
) -> None:
    """Record an AssociativeRecall.expand() call's latency and fan-out.

    ``mode`` is clamped to :data:`_ALLOWED_EXPANSION_MODES`; unknown values
    collapse to ``"unknown"`` so unvalidated upstream intent strings cannot
    blow up Prometheus label cardinality.
    """
    safe_mode = mode if mode in _ALLOWED_EXPANSION_MODES else "unknown"
    try:
        graph_expansion_latency_seconds.labels(mode=safe_mode).observe(
            float(latency_s)
        )
        graph_expansion_candidates.labels(mode=safe_mode).observe(
            float(candidate_count)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "metrics.record_graph_expansion failed: mode=%s err=%s",
            safe_mode,
            exc,
        )


def record_entity_mention(n: int = 1) -> None:
    """Increment the MENTIONS-edge counter.

    Pass ``n`` when batching multiple mentions from a single call site. The
    default single-increment form is what the per-edge call site uses.
    """
    try:
        if n <= 0:
            return
        entity_mentions_created_total.inc(n)
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics.record_entity_mention failed: err=%s", exc)


def record_backfill_record(script: str) -> None:
    """Increment the records-processed counter at a backfill batch boundary.

    Called once per batch (not per record) by backfill scripts so operators
    can observe in-flight liveness via
    ``rate(backfill_records_processed_total[1m])`` even on full-graph runs
    where ``max_total`` is unset and :data:`backfill_progress` stays at 0
    until completion.
    """
    try:
        backfill_records_processed_total.labels(script=script).inc()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "metrics.record_backfill_record failed: script=%s err=%s",
            script,
            exc,
        )


def set_backfill_progress(script: str, phase: str, value: float) -> None:
    """Set the fractional progress gauge for a backfill script.

    ``value`` should land in ``[0.0, 1.0]``. Callers are expected to emit
    this at batch boundaries, not per-record, to keep cardinality bounded.
    """
    try:
        backfill_progress.labels(script=script, phase=phase).set(float(value))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "metrics.set_backfill_progress failed: script=%s phase=%s err=%s",
            script,
            phase,
            exc,
        )


__all__ = [
    "REGISTRY",
    "declared_metric_names",
    "record_backfill_record",
    "record_edge_created",
    "record_entity_mention",
    "record_graph_expansion",
    "record_similarity_completion",
    "record_similarity_enqueue",
    "record_similarity_queue_event",
    "reset_registry_for_tests",
    "set_backfill_progress",
]
