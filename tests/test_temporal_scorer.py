"""Tests for Phase P9A.2: Temporal Decay Scoring & Composite Scoring.

All tests are self-contained with no external service dependencies.
"""

import math
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# --- Adjust sys.path so we can import from the 'app' directory ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from app.services.temporal_scorer import (
    HALF_LIVES,
    temporal_decay_score,
    load_half_lives,
)
from app.services.composite_scorer import (
    composite_score,
    normalize_semantic_score,
    DEFAULT_WEIGHTS,
)


# Fixed reference time for deterministic tests
NOW = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)


class TestTemporalDecayScore(unittest.TestCase):
    """Tests for temporal_decay_score()."""

    def test_fresh_memory_high_score(self):
        """A memory from 1 hour ago should score > 0.99."""
        one_hour_ago = (NOW - timedelta(hours=1)).isoformat()
        score = temporal_decay_score(one_hour_ago, category="default", now=NOW)
        self.assertGreater(score, 0.99)

    def test_old_debug_decays_fast(self):
        """A debug memory from 14 days ago should score < 0.25.

        Debug half-life = 7 days.  14 days = 2 half-lives.
        Expected: exp(-ln2 * 14/7) = exp(-2*ln2) = 0.25.
        Due to max(0.01, ...) floor this should be ~0.25.
        """
        two_weeks_ago = (NOW - timedelta(days=14)).isoformat()
        score = temporal_decay_score(two_weeks_ago, category="debug", now=NOW)
        self.assertLess(score, 0.26)
        # Also verify it's close to expected 0.25
        self.assertAlmostEqual(score, 0.25, places=2)

    def test_old_decision_decays_slow(self):
        """A decision from 14 days ago should score > 0.85.

        Decision half-life = 90 days.
        Expected: exp(-ln2 * 14/90) ≈ exp(-0.1078) ≈ 0.8977.
        """
        two_weeks_ago = (NOW - timedelta(days=14)).isoformat()
        score = temporal_decay_score(two_weeks_ago, category="decision", now=NOW)
        self.assertGreater(score, 0.85)

    def test_unknown_time_neutral(self):
        """An invalid timestamp should return 0.5 (neutral score)."""
        score = temporal_decay_score("not-a-timestamp", category="debug", now=NOW)
        self.assertEqual(score, 0.5)

        score_none = temporal_decay_score(None, category="debug", now=NOW)
        self.assertEqual(score_none, 0.5)

        score_empty = temporal_decay_score("", category="debug", now=NOW)
        self.assertEqual(score_empty, 0.5)

    def test_decay_floor(self):
        """A very old memory should score >= 0.01, never zero."""
        very_old = (NOW - timedelta(days=3650)).isoformat()  # 10 years ago
        score = temporal_decay_score(very_old, category="debug", now=NOW)
        self.assertGreaterEqual(score, 0.01)
        # Debug half-life = 7 days, 3650/7 = 521 half-lives → effectively 0
        # but floor is 0.01
        self.assertAlmostEqual(score, 0.01, places=4)

    def test_no_category_uses_default(self):
        """When category is None, should use the 'default' half-life (30 days)."""
        thirty_days_ago = (NOW - timedelta(days=30)).isoformat()
        score = temporal_decay_score(thirty_days_ago, category=None, now=NOW)
        # 1 half-life → 0.5
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_future_timestamp_full_score(self):
        """A future timestamp should get full score (1.0)."""
        future = (NOW + timedelta(days=5)).isoformat()
        score = temporal_decay_score(future, category="debug", now=NOW)
        self.assertEqual(score, 1.0)

    def test_pattern_category_long_lived(self):
        """A pattern memory from 90 days ago should still score well.

        Pattern half-life = 180 days.
        Expected: exp(-ln2 * 90/180) = exp(-0.5*ln2) ≈ 0.707.
        """
        ninety_days_ago = (NOW - timedelta(days=90)).isoformat()
        score = temporal_decay_score(ninety_days_ago, category="pattern", now=NOW)
        self.assertAlmostEqual(score, 0.707, places=2)

    def test_custom_half_lives(self):
        """Custom half-lives passed directly should be used."""
        custom_hl = {"debug": 1}  # Very short half-life
        one_day_ago = (NOW - timedelta(days=1)).isoformat()
        score = temporal_decay_score(
            one_day_ago, category="debug", now=NOW, half_lives=custom_hl
        )
        # 1 half-life → 0.5
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_naive_datetime_assumed_utc(self):
        """A naive ISO string (no timezone) should be treated as UTC."""
        one_hour_ago = (NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat()
        score = temporal_decay_score(one_hour_ago, category="default", now=NOW)
        self.assertGreater(score, 0.99)


class TestLoadHalfLives(unittest.TestCase):
    """Tests for load_half_lives()."""

    def test_default_half_lives(self):
        """Without override, returns the default table."""
        hl = load_half_lives(None)
        self.assertEqual(hl, HALF_LIVES)

    def test_json_override(self):
        """A valid JSON override replaces specific keys."""
        hl = load_half_lives('{"debug": 3, "custom_cat": 60}')
        self.assertEqual(hl["debug"], 3)
        self.assertEqual(hl["custom_cat"], 60)
        # Unchanged keys remain
        self.assertEqual(hl["decision"], 90)

    def test_invalid_json_returns_defaults(self):
        """Invalid JSON should return the defaults with a warning."""
        hl = load_half_lives("not valid json")
        self.assertEqual(hl["debug"], HALF_LIVES["debug"])

    def test_configurable_half_lives_from_env(self):
        """Half-lives loaded from TEMPORAL_DECAY_HALF_LIVES env var."""
        env_val = '{"debug": 2, "decision": 180}'
        hl = load_half_lives(env_val)
        self.assertEqual(hl["debug"], 2)
        self.assertEqual(hl["decision"], 180)
        # Other categories unchanged
        self.assertEqual(hl["pattern"], 180)


class TestCompositeScore(unittest.TestCase):
    """Tests for composite_score()."""

    def test_composite_weighting(self):
        """Verify the weighted sum math is correct."""
        semantic = 0.8
        temporal = 0.6
        frequency = 0.5
        importance = 0.5

        expected = (
            DEFAULT_WEIGHTS["semantic"] * semantic
            + DEFAULT_WEIGHTS["temporal"] * temporal
            + DEFAULT_WEIGHTS["frequency"] * frequency
            + DEFAULT_WEIGHTS["importance"] * importance
        )
        result = composite_score(semantic, temporal, frequency, importance)
        self.assertAlmostEqual(result, expected, places=6)

    def test_all_ones(self):
        """All signals at 1.0 should produce a score of 1.0."""
        result = composite_score(1.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_all_zeros(self):
        """All signals at 0.0 should produce a score of 0.0."""
        result = composite_score(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_custom_weights(self):
        """Custom weight dict should override defaults."""
        custom_w = {
            "semantic": 0.50,
            "temporal": 0.50,
            "frequency": 0.0,
            "importance": 0.0,
        }
        result = composite_score(0.8, 0.4, weights=custom_w)
        expected = 0.50 * 0.8 + 0.50 * 0.4
        self.assertAlmostEqual(result, expected, places=6)

    def test_semantic_dominant(self):
        """With high semantic and low temporal, composite should lean semantic."""
        result = composite_score(1.0, 0.0)
        # Should be 0.55 * 1.0 + 0.30 * 0.0 + 0.10 * 0.5 + 0.05 * 0.5 = 0.625
        self.assertAlmostEqual(result, 0.625, places=3)


class TestNormalizeSemanticScore(unittest.TestCase):
    """Tests for normalize_semantic_score()."""

    def test_rerank_positive_logit(self):
        """Positive rerank logit should be > 0.5 after sigmoid."""
        result = normalize_semantic_score(5.0, "rerank")
        self.assertGreater(result, 0.99)

    def test_rerank_negative_logit(self):
        """Negative rerank logit should be < 0.5 after sigmoid."""
        result = normalize_semantic_score(-5.0, "rerank")
        self.assertLess(result, 0.01)

    def test_rerank_zero_logit(self):
        """Zero logit should map to 0.5."""
        result = normalize_semantic_score(0.0, "rerank")
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_rrf_score_normalization(self):
        """RRF scores should be scaled relative to ceiling of ~0.035."""
        # Typical RRF score for rank 1 from both sources ≈ 0.0328
        result = normalize_semantic_score(0.0328, "rrf")
        self.assertGreater(result, 0.9)
        self.assertLessEqual(result, 1.0)

    def test_rrf_zero(self):
        """RRF score of 0 should normalize to 0."""
        result = normalize_semantic_score(0.0, "rrf")
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_unknown_score_type_clamped(self):
        """Unknown score type should just clamp to [0,1]."""
        result = normalize_semantic_score(0.7, "unknown")
        self.assertAlmostEqual(result, 0.7, places=4)

        result_over = normalize_semantic_score(1.5, "unknown")
        self.assertAlmostEqual(result_over, 1.0, places=4)

    def test_rerank_overflow_protection(self):
        """Extremely large/small logits should not raise OverflowError."""
        result_big = normalize_semantic_score(1000.0, "rerank")
        self.assertAlmostEqual(result_big, 1.0, places=4)

        result_small = normalize_semantic_score(-1000.0, "rerank")
        self.assertAlmostEqual(result_small, 0.0, places=4)


class TestIntegrationScenarios(unittest.TestCase):
    """End-to-end scenarios combining temporal + composite scoring."""

    def test_recent_decision_beats_old_debug(self):
        """A recent decision should outscore an old debug note with
        the same semantic relevance."""
        recent_decision_time = (NOW - timedelta(hours=2)).isoformat()
        old_debug_time = (NOW - timedelta(days=14)).isoformat()

        # Same semantic score
        semantic = 0.8

        decision_temporal = temporal_decay_score(
            recent_decision_time, "decision", now=NOW
        )
        debug_temporal = temporal_decay_score(old_debug_time, "debug", now=NOW)

        decision_composite = composite_score(semantic, decision_temporal)
        debug_composite = composite_score(semantic, debug_temporal)

        self.assertGreater(decision_composite, debug_composite)

    def test_highly_relevant_old_memory_can_still_win(self):
        """A semantically strong old memory can still beat a weak recent one."""
        old_time = (NOW - timedelta(days=60)).isoformat()
        recent_time = (NOW - timedelta(hours=1)).isoformat()

        old_semantic = 1.0  # Perfect semantic match
        recent_semantic = 0.1  # Weak match

        old_temporal = temporal_decay_score(old_time, "decision", now=NOW)
        recent_temporal = temporal_decay_score(recent_time, "debug", now=NOW)

        old_composite = composite_score(old_semantic, old_temporal)
        recent_composite = composite_score(recent_semantic, recent_temporal)

        # Old memory with strong semantic should still win
        self.assertGreater(old_composite, recent_composite)


if __name__ == "__main__":
    unittest.main()
