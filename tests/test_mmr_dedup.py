"""Tests for MMR deduplication (Phase P9A.3).

All tests use synthetic embeddings — no external API calls required.
"""

import numpy as np
import pytest

from app.services.mmr_dedup import deduplicate_exact, mmr_rerank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    rid: str,
    text: str,
    composite_score: float = 0.5,
    rerank_score: float = None,
    rrf_score: float = None,
) -> dict:
    """Create a synthetic result dict."""
    r = {
        "id": rid,
        "text": text,
        "metadata": {"text": text},
        "composite_score": composite_score,
    }
    if rerank_score is not None:
        r["rerank_score"] = rerank_score
    if rrf_score is not None:
        r["rrf_score"] = rrf_score
    return r


def _random_embedding(dim: int = 128, seed: int = 0) -> list:
    """Deterministic random unit-ish embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim)
    v = v / (np.linalg.norm(v) + 1e-10)
    return v.tolist()


def _near_duplicate_embedding(base: list, noise_scale: float = 0.01, seed: int = 1) -> list:
    """Create an embedding that is very similar to *base* (near-duplicate)."""
    rng = np.random.RandomState(seed)
    arr = np.array(base) + rng.randn(len(base)) * noise_scale
    arr = arr / (np.linalg.norm(arr) + 1e-10)
    return arr.tolist()


# ---------------------------------------------------------------------------
# test_exact_dedup_removes_duplicates
# ---------------------------------------------------------------------------


class TestExactDedup:
    def test_exact_dedup_removes_duplicates(self):
        """Same text with different IDs should be deduplicated."""
        results = [
            _make_result("id-1", "Decision: use PostgreSQL for persistence"),
            _make_result("id-2", "Decision: use PostgreSQL for persistence"),
            _make_result("id-3", "Research: Redis vs Memcached comparison"),
        ]
        deduped = deduplicate_exact(results)
        assert len(deduped) == 2
        assert deduped[0]["id"] == "id-1"
        assert deduped[1]["id"] == "id-3"

    def test_exact_dedup_preserves_order(self):
        """First occurrence of each unique text should be kept."""
        results = [
            _make_result("a", "alpha"),
            _make_result("b", "beta"),
            _make_result("c", "alpha"),
            _make_result("d", "gamma"),
            _make_result("e", "beta"),
        ]
        deduped = deduplicate_exact(results)
        assert [r["id"] for r in deduped] == ["a", "b", "d"]

    def test_exact_dedup_empty_list(self):
        """Empty input returns empty output."""
        assert deduplicate_exact([]) == []

    def test_exact_dedup_no_duplicates(self):
        """All unique texts should pass through unchanged."""
        results = [
            _make_result("1", "first"),
            _make_result("2", "second"),
            _make_result("3", "third"),
        ]
        deduped = deduplicate_exact(results)
        assert len(deduped) == 3

    def test_exact_dedup_metadata_text_fallback(self):
        """Falls back to metadata.text when top-level text is missing."""
        results = [
            {"id": "a", "metadata": {"text": "shared content"}},
            {"id": "b", "metadata": {"text": "shared content"}},
            {"id": "c", "metadata": {"text": "unique content"}},
        ]
        deduped = deduplicate_exact(results)
        assert len(deduped) == 2


# ---------------------------------------------------------------------------
# test_mmr_selects_diverse_results
# ---------------------------------------------------------------------------


class TestMMRRerank:
    def test_mmr_selects_diverse_results(self):
        """Near-duplicate results should be demoted in favour of diverse ones."""
        base_emb = _random_embedding(dim=64, seed=42)
        near_dup = _near_duplicate_embedding(base_emb, noise_scale=0.01, seed=43)
        diverse_emb = _random_embedding(dim=64, seed=99)

        results = [
            _make_result("original", "Original finding about caching", composite_score=0.9),
            _make_result("near_dup", "Original finding about caching (v2)", composite_score=0.88),
            _make_result("diverse", "Unrelated finding about auth tokens", composite_score=0.85),
        ]
        embeddings = [base_emb, near_dup, diverse_emb]

        selected = mmr_rerank(results, embeddings, lambda_param=0.5, top_n=2)

        assert len(selected) == 2
        # First should be highest relevance
        assert selected[0]["id"] == "original"
        # Second should be the diverse one (near-dup penalized by similarity)
        assert selected[1]["id"] == "diverse"

    def test_mmr_preserves_top_result(self):
        """The highest-relevance result should always be selected first."""
        embs = [_random_embedding(dim=64, seed=i) for i in range(5)]
        results = [
            _make_result(f"r{i}", f"text {i}", composite_score=0.9 - i * 0.1)
            for i in range(5)
        ]
        selected = mmr_rerank(results, embs, lambda_param=0.7, top_n=3)
        assert selected[0]["id"] == "r0"

    def test_mmr_lambda_1_pure_relevance(self):
        """lambda=1.0 should produce the same ordering as sorting by score."""
        embs = [_random_embedding(dim=64, seed=i) for i in range(5)]
        results = [
            _make_result(f"r{i}", f"text {i}", composite_score=score)
            for i, score in enumerate([0.3, 0.9, 0.7, 0.5, 0.1])
        ]

        selected = mmr_rerank(results, embs, lambda_param=1.0, top_n=5)

        # Should be in descending score order (lambda=1.0 ignores diversity entirely)
        scores = [r["composite_score"] for r in selected]
        assert scores == sorted(scores, reverse=True)

    def test_mmr_lambda_0_pure_diversity(self):
        """lambda=0.0 should maximize diversity (minimize max-sim to selected)."""
        # Create embeddings where some are near-duplicates and others are diverse
        base = _random_embedding(dim=64, seed=10)
        dup1 = _near_duplicate_embedding(base, noise_scale=0.005, seed=11)
        dup2 = _near_duplicate_embedding(base, noise_scale=0.005, seed=12)
        far1 = _random_embedding(dim=64, seed=100)
        far2 = _random_embedding(dim=64, seed=200)

        results = [
            _make_result("base", "base text", composite_score=0.95),
            _make_result("dup1", "dup1 text", composite_score=0.90),
            _make_result("dup2", "dup2 text", composite_score=0.85),
            _make_result("far1", "far1 text", composite_score=0.50),
            _make_result("far2", "far2 text", composite_score=0.40),
        ]
        embeddings = [base, dup1, dup2, far1, far2]

        selected = mmr_rerank(results, embeddings, lambda_param=0.0, top_n=3)

        # With pure diversity, after selecting base (highest score goes first
        # because normalized scores are all 1.0 when lambda=0, so argmax picks first),
        # the next picks should favour far1/far2 over dup1/dup2
        selected_ids = {r["id"] for r in selected}
        # At least one of the far results should be picked over both duplicates
        assert len(selected_ids & {"far1", "far2"}) >= 1

    def test_single_result_passthrough(self):
        """A single result should be returned unchanged."""
        result = _make_result("solo", "only result", composite_score=0.8)
        emb = _random_embedding(dim=64, seed=0)

        selected = mmr_rerank([result], [emb], lambda_param=0.7, top_n=5)
        assert len(selected) == 1
        assert selected[0]["id"] == "solo"
        assert selected[0]["composite_score"] == 0.8

    def test_empty_results(self):
        """Empty input should return empty output."""
        selected = mmr_rerank([], [], lambda_param=0.7, top_n=5)
        assert selected == []

    def test_top_n_limits_output(self):
        """Output should not exceed top_n even if more results are available."""
        embs = [_random_embedding(dim=64, seed=i) for i in range(10)]
        results = [
            _make_result(f"r{i}", f"text {i}", composite_score=0.5)
            for i in range(10)
        ]

        selected = mmr_rerank(results, embs, lambda_param=0.7, top_n=3)
        assert len(selected) == 3

    def test_mmr_with_different_score_fields(self):
        """MMR should work with rerank_score when composite_score is absent."""
        embs = [_random_embedding(dim=64, seed=i) for i in range(3)]
        results = [
            {"id": "a", "text": "text a", "metadata": {"text": "text a"}, "rerank_score": 0.9},
            {"id": "b", "text": "text b", "metadata": {"text": "text b"}, "rerank_score": 0.7},
            {"id": "c", "text": "text c", "metadata": {"text": "text c"}, "rerank_score": 0.5},
        ]

        selected = mmr_rerank(results, embs, lambda_param=1.0, top_n=3)
        # With lambda=1.0, pure relevance — should be sorted by rerank_score
        assert selected[0]["id"] == "a"

    def test_mmr_with_rrf_score_fallback(self):
        """MMR should fall back to rrf_score when no other scores are present."""
        embs = [_random_embedding(dim=64, seed=i) for i in range(3)]
        results = [
            {"id": "a", "text": "text a", "metadata": {"text": "text a"}, "rrf_score": 0.03},
            {"id": "b", "text": "text b", "metadata": {"text": "text b"}, "rrf_score": 0.02},
            {"id": "c", "text": "text c", "metadata": {"text": "text c"}, "rrf_score": 0.01},
        ]

        selected = mmr_rerank(results, embs, lambda_param=1.0, top_n=3)
        assert selected[0]["id"] == "a"

    def test_mmr_identical_scores_still_works(self):
        """When all relevance scores are identical, diversity should drive selection."""
        # Create two clusters of similar embeddings
        cluster_a = _random_embedding(dim=64, seed=10)
        cluster_a2 = _near_duplicate_embedding(cluster_a, noise_scale=0.01, seed=11)
        cluster_b = _random_embedding(dim=64, seed=50)
        cluster_b2 = _near_duplicate_embedding(cluster_b, noise_scale=0.01, seed=51)

        results = [
            _make_result("a1", "cluster A item 1", composite_score=0.5),
            _make_result("a2", "cluster A item 2", composite_score=0.5),
            _make_result("b1", "cluster B item 1", composite_score=0.5),
            _make_result("b2", "cluster B item 2", composite_score=0.5),
        ]
        embeddings = [cluster_a, cluster_a2, cluster_b, cluster_b2]

        selected = mmr_rerank(results, embeddings, lambda_param=0.5, top_n=2)

        # With equal scores and lambda=0.5, after picking first item,
        # second should be from the OTHER cluster (more diverse)
        ids = [r["id"] for r in selected]
        assert len(ids) == 2
        # Verify one from each cluster
        has_a = any(i.startswith("a") for i in ids)
        has_b = any(i.startswith("b") for i in ids)
        assert has_a and has_b


# ---------------------------------------------------------------------------
# Integration-style test: exact dedup + MMR together
# ---------------------------------------------------------------------------


class TestDeduplicationPipeline:
    def test_full_pipeline_exact_then_mmr(self):
        """Exact dedup followed by MMR should remove both exact and near dupes."""
        base_emb = _random_embedding(dim=64, seed=1)
        near_emb = _near_duplicate_embedding(base_emb, noise_scale=0.01, seed=2)
        diverse_emb = _random_embedding(dim=64, seed=99)

        results = [
            _make_result("orig", "The system uses Redis for caching", composite_score=0.9),
            _make_result("exact_dup", "The system uses Redis for caching", composite_score=0.88),
            _make_result("near_dup", "System relies on Redis as its cache layer", composite_score=0.85),
            _make_result("diverse", "Authentication flow uses JWT tokens", composite_score=0.80),
        ]
        embeddings_map = {
            "orig": base_emb,
            "exact_dup": base_emb,  # exact dup
            "near_dup": near_emb,
            "diverse": diverse_emb,
        }

        # Step 1: exact dedup
        deduped = deduplicate_exact(results)
        assert len(deduped) == 3  # exact dup removed

        # Step 2: MMR
        embs = [embeddings_map[r["id"]] for r in deduped]
        final = mmr_rerank(deduped, embs, lambda_param=0.5, top_n=2)

        assert len(final) == 2
        assert final[0]["id"] == "orig"
        # The diverse result should beat the near-duplicate
        assert final[1]["id"] == "diverse"
