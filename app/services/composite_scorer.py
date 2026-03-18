"""Composite scoring for memory retrieval (Phase P9A.2).

Combines semantic similarity, temporal decay, frequency, and importance
signals into a single composite score — inspired by Mem0's scoring approach.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default weight distribution
DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic": 0.55,
    "temporal": 0.30,
    "frequency": 0.10,
    "importance": 0.05,
}


def composite_score(
    semantic_score: float,
    temporal_score: float,
    frequency_score: float = 0.5,
    importance_score: float = 0.5,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute a weighted composite score from multiple signals.

    Args:
        semantic_score: Similarity score from the reranker or RRF pipeline,
            expected in [0, 1].
        temporal_score: Decay factor from :func:`temporal_decay_score`,
            in [0.01, 1.0].
        frequency_score: How often this memory has been retrieved (future use).
            Defaults to 0.5 (neutral).
        importance_score: Manual importance flag (future use). Defaults to 0.5.
        weights: Optional dict of signal weights.  Must contain keys
            ``"semantic"``, ``"temporal"``, ``"frequency"``, ``"importance"``.
            Defaults to :data:`DEFAULT_WEIGHTS`.

    Returns:
        A float representing the composite relevance score.
    """
    w = weights or DEFAULT_WEIGHTS

    return (
        w.get("semantic", 0.55) * semantic_score
        + w.get("temporal", 0.30) * temporal_score
        + w.get("frequency", 0.10) * frequency_score
        + w.get("importance", 0.05) * importance_score
    )


def normalize_semantic_score(
    raw_score: float,
    score_type: str = "rerank",
) -> float:
    """Normalize a raw semantic score to [0, 1].

    Cross-encoder rerank scores are logits (unbounded, often in [-10, 10]).
    RRF scores are small positive floats (typically [0, ~0.03]).

    Args:
        raw_score: The raw score value.
        score_type: One of ``"rerank"`` or ``"rrf"``.

    Returns:
        A float in [0, 1].
    """
    if score_type == "rerank":
        # Sigmoid squash for cross-encoder logits
        import math

        try:
            return 1.0 / (1.0 + math.exp(-raw_score))
        except OverflowError:
            return 0.0 if raw_score < 0 else 1.0

    elif score_type == "rrf":
        # RRF scores: 1/(k+rank).  With k=60, rank 1 → 0.0164, rank 60 → 0.0083
        # Scale to [0,1]: multiply by k to get roughly [0,1] range
        # Max possible single-source RRF = 1/(60+1) ≈ 0.0164
        # Max possible dual-source RRF ≈ 0.0328
        # We use 0.035 as practical ceiling for normalization
        ceiling = 0.035
        return min(1.0, max(0.0, raw_score / ceiling))

    else:
        # Unknown type — clamp to [0,1]
        return min(1.0, max(0.0, raw_score))
