"""Observability surface for PLAN-0759 Phase 8 — Prometheus metrics."""

from .metrics import (
    REGISTRY,
    record_edge_created,
    record_entity_mention,
    record_graph_expansion,
    record_similarity_completion,
    record_similarity_enqueue,
    record_similarity_queue_event,
    reset_registry_for_tests,
    set_backfill_progress,
)

__all__ = [
    "REGISTRY",
    "record_edge_created",
    "record_entity_mention",
    "record_graph_expansion",
    "record_similarity_completion",
    "record_similarity_enqueue",
    "record_similarity_queue_event",
    "reset_registry_for_tests",
    "set_backfill_progress",
]
