"""Tests for Phase P9A.4 Intelligent Query Router.

Tests the upgraded QueryRouter with pattern-based intent classification
and the new DECISION, PATTERN, SESSION routing modes.
No external dependencies needed.
"""

import os
import pytest

from app.services.query_router import (
    INTENT_PATTERNS,
    QueryRouter,
    RoutingMode,
)


@pytest.fixture
def router():
    return QueryRouter()


# --- Temporal Intent ---


class TestTemporalIntent:
    """Queries with recency intent route to TEMPORAL."""

    def test_what_did_we_do_last_session(self, router):
        mode = router.route("what did we do last session")
        # "last" matches TEMPORAL, "session" matches SESSION -> TEMPORAL_SEMANTIC
        # because temporal + another mode = TEMPORAL_SEMANTIC
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_most_recent_events(self, router):
        mode = router.route("show me the most recent events")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_what_happened_yesterday(self, router):
        mode = router.route("what happened yesterday")
        assert mode in (RoutingMode.TEMPORAL, RoutingMode.TEMPORAL_SEMANTIC)

    def test_pure_temporal_today(self, router):
        mode = router.route("today")
        assert mode == RoutingMode.TEMPORAL


# --- Decision Intent ---


class TestDecisionIntent:
    """Queries recalling decisions route to DECISION."""

    def test_why_did_we_choose_pinecone(self, router):
        mode = router.route("why did we choose Pinecone over Weaviate")
        assert mode == RoutingMode.DECISION

    def test_decided_on_auth(self, router):
        mode = router.route("What did we decide about auth?")
        # "decided" matches DECISION; "what did" may match TEMPORAL
        # temporal + non-temporal = TEMPORAL_SEMANTIC
        assert mode in (RoutingMode.DECISION, RoutingMode.TEMPORAL_SEMANTIC)

    def test_rationale_for_approach(self, router):
        mode = router.route("what is the rationale for this approach")
        assert mode in (RoutingMode.DECISION, RoutingMode.TEMPORAL_SEMANTIC)

    def test_what_was_the_plan(self, router):
        mode = router.route("what was the plan for deployment")
        assert mode in (RoutingMode.DECISION, RoutingMode.TEMPORAL_SEMANTIC)

    def test_chose_strategy(self, router):
        mode = router.route("We chose a microservices strategy")
        assert mode == RoutingMode.DECISION

    def test_trade_off(self, router):
        mode = router.route("What were the trade-offs of using Redis?")
        assert mode in (RoutingMode.DECISION, RoutingMode.TEMPORAL_SEMANTIC)


# --- Pattern Intent ---


class TestPatternIntent:
    """Queries about patterns/workflows route to PATTERN."""

    def test_how_do_we_handle_errors(self, router):
        mode = router.route("how do we handle error recovery")
        assert mode == RoutingMode.PATTERN

    def test_best_practice_for_testing(self, router):
        mode = router.route("what is the best practice for testing")
        assert mode in (RoutingMode.PATTERN, RoutingMode.TEMPORAL_SEMANTIC)

    def test_standard_deployment(self, router):
        mode = router.route("what is the standard for deployment")
        assert mode in (RoutingMode.PATTERN, RoutingMode.TEMPORAL_SEMANTIC)

    def test_typical_workflow(self, router):
        mode = router.route("what is our typical workflow for code review")
        assert mode in (RoutingMode.PATTERN, RoutingMode.TEMPORAL_SEMANTIC)

    def test_how_should_we(self, router):
        mode = router.route("how should we structure the API layer")
        assert mode == RoutingMode.PATTERN

    def test_convention(self, router):
        mode = router.route("what is the convention for naming files")
        assert mode in (RoutingMode.PATTERN, RoutingMode.TEMPORAL_SEMANTIC)


# --- Session Intent ---


class TestSessionIntent:
    """Queries about sessions/checkpoints route to SESSION."""

    def test_resume_from_checkpoint(self, router):
        mode = router.route("resume from last checkpoint")
        # "resume" + "checkpoint" -> SESSION, "last" -> TEMPORAL
        # temporal + session -> TEMPORAL_SEMANTIC (by temporal combo rule)
        assert mode in (RoutingMode.SESSION, RoutingMode.TEMPORAL_SEMANTIC)

    def test_previous_session(self, router):
        mode = router.route("tell me about the previous session")
        # "previous session" matches SESSION, "last time" could match TEMPORAL
        assert mode in (RoutingMode.SESSION, RoutingMode.TEMPORAL_SEMANTIC)

    def test_pick_up_where_we_left_off(self, router):
        mode = router.route("let's pick up where we left off")
        # "pick up where" matches SESSION, "left off" matches SESSION
        assert mode in (RoutingMode.SESSION, RoutingMode.TEMPORAL_SEMANTIC)

    def test_continue_from_last(self, router):
        mode = router.route("continue from where we stopped")
        assert mode in (RoutingMode.SESSION, RoutingMode.TEMPORAL_SEMANTIC)

    def test_where_were_we(self, router):
        mode = router.route("where were we in the migration")
        assert mode in (RoutingMode.SESSION, RoutingMode.TEMPORAL_SEMANTIC)


# --- Ambiguous / Hybrid Default ---


class TestAmbiguousDefaultsHybrid:
    """Unclear or unmatched queries default to HYBRID."""

    def test_generic_query(self, router):
        mode = router.route("hello world")
        assert mode == RoutingMode.HYBRID

    def test_random_text(self, router):
        mode = router.route("the quick brown fox jumps over the lazy dog")
        assert mode == RoutingMode.HYBRID

    def test_numbers_only(self, router):
        mode = router.route("12345")
        assert mode == RoutingMode.HYBRID


# --- Routing Mode Coverage ---


class TestRoutingModeCoverage:
    """Every mode in INTENT_PATTERNS has at least one matching pattern."""

    def test_all_modes_have_patterns(self):
        for mode in INTENT_PATTERNS:
            patterns = INTENT_PATTERNS[mode]
            assert len(patterns) > 0, f"RoutingMode.{mode.name} has no patterns defined"

    def test_new_modes_exist_in_enum(self):
        expected_new = {"DECISION", "PATTERN", "SESSION"}
        actual = {m.name for m in RoutingMode}
        assert expected_new.issubset(actual), f"Missing modes: {expected_new - actual}"

    def test_backward_compatible_modes(self):
        """Original modes still exist."""
        expected_original = {"VECTOR", "GRAPH", "HYBRID", "TEMPORAL", "TEMPORAL_SEMANTIC"}
        actual = {m.name for m in RoutingMode}
        assert expected_original.issubset(actual), f"Missing modes: {expected_original - actual}"

    def test_all_eight_modes(self):
        expected = {
            "VECTOR", "GRAPH", "HYBRID", "TEMPORAL", "TEMPORAL_SEMANTIC",
            "DECISION", "PATTERN", "SESSION",
        }
        actual = {m.name for m in RoutingMode}
        assert actual == expected


# --- LLM Fallback ---


class TestLLMFallback:
    """LLM classifier behavior when enabled/disabled."""

    def test_llm_fallback_disabled_by_default(self, router):
        """LLM route returns None when QUERY_ROUTER_LLM_ENABLED is not set."""
        result = router.route_with_llm("some ambiguous query")
        assert result is None

    def test_llm_fallback_returns_none_when_disabled(self, router, monkeypatch):
        """Explicitly setting env to false should return None."""
        monkeypatch.setenv("QUERY_ROUTER_LLM_ENABLED", "false")
        result = router.route_with_llm("some query")
        assert result is None

    def test_llm_fallback_returns_none_when_enabled_but_not_wired(self, router, monkeypatch):
        """Even when enabled, the stub returns None (not wired yet)."""
        monkeypatch.setenv("QUERY_ROUTER_LLM_ENABLED", "true")
        result = router.route_with_llm("some query")
        assert result is None

    def test_llm_cache_populated_after_call(self, router):
        """Cache should remain empty since LLM is not wired."""
        router.route_with_llm("test query")
        # Since LLM is disabled, nothing gets cached
        assert len(router._llm_cache) == 0


# --- Empty Query ---


class TestEmptyQuery:
    """Empty or whitespace-only queries default to HYBRID."""

    def test_empty_string(self, router):
        assert router.route("") == RoutingMode.HYBRID

    def test_whitespace_only(self, router):
        assert router.route("   ") == RoutingMode.HYBRID

    def test_tab_only(self, router):
        assert router.route("\t") == RoutingMode.HYBRID


# --- Temporal + Semantic Combo ---


class TestTemporalSemanticCombo:
    """Queries with both temporal and non-temporal intent route to TEMPORAL_SEMANTIC."""

    def test_recent_decisions_about_auth(self, router):
        mode = router.route("recent decisions about auth")
        assert mode == RoutingMode.TEMPORAL_SEMANTIC

    def test_latest_pattern_for_deployment(self, router):
        mode = router.route("what is the latest best practice for deployment")
        assert mode == RoutingMode.TEMPORAL_SEMANTIC

    def test_yesterday_session_events(self, router):
        mode = router.route("what happened in yesterday's session")
        assert mode == RoutingMode.TEMPORAL_SEMANTIC

    def test_recent_workflow_changes(self, router):
        mode = router.route("what recent workflow changes were made")
        assert mode == RoutingMode.TEMPORAL_SEMANTIC

    def test_latest_decision(self, router):
        mode = router.route("what was the latest decision we made")
        assert mode == RoutingMode.TEMPORAL_SEMANTIC


# --- Graph Intent ---


class TestGraphIntent:
    """Queries about relationships route to GRAPH."""

    def test_relationship_between(self, router):
        mode = router.route("relationship between service A and service B")
        assert mode == RoutingMode.GRAPH

    def test_connection_between(self, router):
        mode = router.route("what is the connection between these modules")
        assert mode == RoutingMode.GRAPH

    def test_depends_on(self, router):
        mode = router.route("what does the auth service depend on")
        assert mode == RoutingMode.GRAPH

    def test_compare_databases(self, router):
        mode = router.route("compare vector databases and graph databases")
        assert mode == RoutingMode.GRAPH


# --- Vector Intent ---


class TestVectorIntent:
    """Pure semantic queries route to VECTOR."""

    def test_define_term(self, router):
        mode = router.route("define machine learning")
        assert mode == RoutingMode.VECTOR

    def test_explain_concept(self, router):
        mode = router.route("explain the concept of photosynthesis")
        assert mode == RoutingMode.VECTOR

    def test_summarize(self, router):
        mode = router.route("summarize the architecture document")
        assert mode == RoutingMode.VECTOR

    def test_who_is(self, router):
        mode = router.route("who is the lead engineer")
        assert mode == RoutingMode.VECTOR


# --- Edge Cases ---


class TestEdgeCases:
    """Edge cases for the router."""

    def test_case_insensitive(self, router):
        mode = router.route("WHY DID WE CHOOSE PINECONE?")
        assert mode == RoutingMode.DECISION

    def test_mixed_case(self, router):
        mode = router.route("What Was The Decision About Auth?")
        assert mode in (RoutingMode.DECISION, RoutingMode.TEMPORAL_SEMANTIC)

    def test_no_false_positive_on_blast(self, router):
        """'last' as substring in 'blast' should NOT match with word boundaries."""
        mode = router.route("blast radius of changes")
        # With regex word boundaries, 'blast' should NOT trigger TEMPORAL
        assert mode == RoutingMode.HYBRID

    def test_long_query(self, router):
        """Very long query still routes correctly."""
        mode = router.route(
            "I want to understand why we decided to go with the current approach "
            "for the authentication system and what the rationale was behind it"
        )
        assert mode in (RoutingMode.DECISION, RoutingMode.TEMPORAL_SEMANTIC)

    def test_multiple_intent_signals(self, router):
        """Query with many intents should still produce a valid mode."""
        mode = router.route("explain the latest decision about our standard workflow")
        assert mode in (
            RoutingMode.TEMPORAL_SEMANTIC,
            RoutingMode.DECISION,
            RoutingMode.PATTERN,
            RoutingMode.VECTOR,
        )
        # Most likely TEMPORAL_SEMANTIC since "latest" hits temporal + other modes
