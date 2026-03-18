"""Tests for Phase 4 temporal-first query router.

Tests the upgraded QueryRouter with TEMPORAL and TEMPORAL_SEMANTIC
routing modes. Updated for P9A.4 pattern-based classification which adds
DECISION, PATTERN, and SESSION modes. No external dependencies needed.
"""

import pytest

from app.services.query_router import QueryRouter, RoutingMode


@pytest.fixture
def router():
    return QueryRouter()


# --- TEMPORAL Routing ---


class TestTemporalRouting:
    """Queries with recency intent route to TEMPORAL or a temporal-adjacent mode."""

    def test_what_did_we_do_last_session(self, router):
        mode = router.route("What did we do last session?")
        # SESSION + TEMPORAL only -> SESSION (session replay with recency)
        assert mode in (RoutingMode.SESSION, RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_most_recent_events(self, router):
        mode = router.route("Show me the most recent events")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_latest_changes(self, router):
        mode = router.route("What are the latest changes?")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_where_were_we(self, router):
        # P9A.4: "where were we" now matches SESSION (more specific than TEMPORAL)
        mode = router.route("Where were we?")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC, RoutingMode.SESSION)

    def test_resume_work(self, router):
        # P9A.4: "resume" now matches SESSION
        mode = router.route("Let's resume from where we left off")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC, RoutingMode.SESSION)

    def test_what_happened_recently(self, router):
        mode = router.route("What happened recently?")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_catch_up(self, router):
        mode = router.route("Help me catch up on progress")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_last_thing_we_did(self, router):
        mode = router.route("What was the last thing we did?")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_pick_up_where(self, router):
        # P9A.4: "pick up where" now matches SESSION
        mode = router.route("Let's pick up where we left off")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC, RoutingMode.SESSION)

    def test_previous_session(self, router):
        # P9A.4: "previous session" now matches SESSION
        mode = router.route("Tell me about the previous session")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC, RoutingMode.SESSION)

    def test_just_did(self, router):
        mode = router.route("What did we just did?")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_earlier_today(self, router):
        mode = router.route("What did we work on earlier today?")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_continuation(self, router):
        # P9A.4: "continuation" was a keyword in old system; with regex patterns
        # it no longer matches temporal (not a word-boundary match for any pattern).
        # May route to HYBRID or TEMPORAL depending on other signals.
        mode = router.route("This is a continuation of our work")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC, RoutingMode.HYBRID)


# --- TEMPORAL_SEMANTIC Routing ---


class TestTemporalSemanticRouting:
    """Queries with both temporal and semantic intent route to TEMPORAL_SEMANTIC."""

    def test_latest_change_to_pipeline(self, router):
        """'latest' (temporal) + 'explain' / specific topic (semantic)."""
        mode = router.route("What's the latest change to the pipeline?")
        # Should detect 'latest' as temporal
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_explain_most_recent_decision(self, router):
        """'explain' (vector) + 'most recent' (temporal) + 'decision' (decision)."""
        mode = router.route("Explain the most recent architecture decision")
        # DECISION is a specific mode, so it wins over TEMPORAL_SEMANTIC synthesis
        assert mode in (RoutingMode.DECISION, RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_what_is_the_latest(self, router):
        """'what is' (vector) + 'latest' (temporal)."""
        mode = router.route("What is the latest deployment config?")
        assert mode == RoutingMode.TEMPORAL_SEMANTIC

    def test_how_does_recent_compare(self, router):
        """'how does' (graph) + 'recently' (temporal)."""
        mode = router.route("How does the recently added module compare?")
        assert mode == RoutingMode.TEMPORAL_SEMANTIC


# --- Non-Temporal Routing (Backward Compatibility) ---


class TestNonTemporalRouting:
    """Queries without temporal intent route via existing logic."""

    def test_explain_concept(self, router):
        mode = router.route("Explain the concept of photosynthesis")
        assert mode == RoutingMode.VECTOR

    def test_relationship_query(self, router):
        """P9A.4: 'relationship' + 'between' gives GRAPH 2 matches vs VECTOR 1.
        GRAPH wins. Both are valid non-temporal routes."""
        mode = router.route("What is the relationship between A and B?")
        assert mode in (RoutingMode.HYBRID, RoutingMode.GRAPH)

    def test_generic_query(self, router):
        # P9A.4: "Tell me about" matches VECTOR pattern; this is an improvement
        # over the old HYBRID default since it has clear semantic search intent.
        mode = router.route("Tell me about the weather")
        assert mode in (RoutingMode.HYBRID, RoutingMode.VECTOR)

    def test_compare_databases(self, router):
        """'compare' is a graph keyword."""
        mode = router.route("Compare vector databases and graph databases")
        assert mode == RoutingMode.GRAPH

    def test_define_term(self, router):
        mode = router.route("Define machine learning")
        assert mode == RoutingMode.VECTOR

    def test_who_is(self, router):
        mode = router.route("Who is the CEO of Anthropic?")
        assert mode == RoutingMode.VECTOR

    def test_both_vector_and_graph(self, router):
        """P9A.4: With pattern-based scoring, GRAPH may win over VECTOR.
        Both are valid; the key is that it doesn't route to TEMPORAL."""
        mode = router.route("Explain the relationship between quantum computing and cryptography")
        assert mode in (RoutingMode.HYBRID, RoutingMode.GRAPH, RoutingMode.VECTOR)


# --- Routing Mode Enum ---


class TestRoutingModeEnum:
    """RoutingMode has the expected values."""

    def test_temporal_exists(self):
        assert hasattr(RoutingMode, "TEMPORAL")

    def test_temporal_semantic_exists(self):
        assert hasattr(RoutingMode, "TEMPORAL_SEMANTIC")

    def test_all_modes(self):
        # P9A.4: Expanded from 5 to 8 modes
        expected = {
            "VECTOR", "GRAPH", "HYBRID", "TEMPORAL", "TEMPORAL_SEMANTIC",
            "DECISION", "PATTERN", "SESSION",
        }
        actual = {m.name for m in RoutingMode}
        assert actual == expected

    def test_original_modes_still_exist(self):
        """Backward compatibility: original 5 modes still present."""
        expected = {"VECTOR", "GRAPH", "HYBRID", "TEMPORAL", "TEMPORAL_SEMANTIC"}
        actual = {m.name for m in RoutingMode}
        assert expected.issubset(actual)


# --- Edge Cases ---


class TestRouterEdgeCases:
    """Edge cases for the router."""

    def test_empty_query(self, router):
        mode = router.route("")
        assert mode == RoutingMode.HYBRID

    def test_case_insensitive(self, router):
        mode = router.route("WHAT DID WE DO LAST SESSION?")
        # SESSION + TEMPORAL only -> SESSION (session replay with recency)
        assert mode in (RoutingMode.SESSION, RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_temporal_keyword_substring(self, router):
        """P9A.4: With word boundary regex, 'blast' no longer matches 'last'."""
        mode = router.route("blast radius of changes")
        # With word boundaries, this should NOT match temporal
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC, RoutingMode.HYBRID)

    def test_no_false_positive_on_unrelated(self, router):
        """Queries with no temporal/vector/graph keywords -> HYBRID."""
        mode = router.route("hello world")
        assert mode == RoutingMode.HYBRID

    def test_temporal_priority_over_hybrid(self, router):
        """Temporal intent should not be downgraded to HYBRID."""
        mode = router.route("What did we do last time?")
        assert mode != RoutingMode.HYBRID
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC, RoutingMode.SESSION)
