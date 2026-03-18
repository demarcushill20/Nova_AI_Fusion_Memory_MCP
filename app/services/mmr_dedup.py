"""Maximal Marginal Relevance deduplication for memory retrieval (Phase P9A.3).

Removes exact-text duplicates (MD5) and then applies MMR reranking to
balance relevance against diversity — penalizing results that are too
similar to already-selected ones.

This prevents the agent's context window from being wasted on near-duplicate
memories (same decision stored multiple times with slightly different wording,
overlapping research notes, etc.).
"""

import hashlib
import logging
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def deduplicate_exact(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove exact text duplicates using MD5 hashing.

    Two results with identical ``text`` (from ``metadata.text`` or the
    top-level ``text`` field) are considered duplicates regardless of their
    IDs, scores, or other metadata.

    Args:
        results: List of result dicts, each expected to contain a ``text``
            key or ``metadata.text``.

    Returns:
        De-duplicated list preserving the original order.
    """
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in results:
        text = r.get("text", "") or r.get("metadata", {}).get("text", "")
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(r)
    return deduped


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def mmr_rerank(
    results: List[Dict[str, Any]],
    embeddings: List[List[float]],
    lambda_param: float = 0.7,
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance selection.

    Iteratively selects the result that maximizes:

        MMR(i) = lambda * relevance(i) - (1 - lambda) * max_sim(i, selected)

    where ``relevance`` is the composite/rerank/rrf score and ``max_sim``
    is the maximum cosine similarity between the candidate embedding and
    any already-selected embedding.

    Args:
        results: Scored result dicts.  Each should contain at least one of
            ``composite_score``, ``rerank_score``, or ``rrf_score``.
        embeddings: Pre-computed embedding vectors aligned 1:1 with *results*.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
            Default 0.7 favours relevance while still suppressing near-dupes.
        top_n: Maximum number of results to return.

    Returns:
        Re-ordered subset of *results* selected by MMR.
    """
    if len(results) <= 1:
        return results

    n_results = len(results)
    top_n = min(top_n, n_results)

    # Build embedding matrix
    emb_matrix = np.array(embeddings, dtype=np.float64)

    # Extract relevance scores (prefer composite > rerank > rrf)
    def _pick_score(r: Dict[str, Any]) -> float:
        cs = r.get("composite_score")
        if cs is not None:
            return float(cs)
        rs = r.get("rerank_score")
        if rs is not None:
            return float(rs)
        return float(r.get("rrf_score", 0.5))

    rel_scores = np.array([
        _pick_score(r) for r in results
    ], dtype=np.float64)

    # Normalize relevance scores to [0, 1]
    score_range = rel_scores.max() - rel_scores.min()
    if score_range > 1e-10:
        rel_scores = (rel_scores - rel_scores.min()) / score_range
    else:
        # All scores identical — set all to 1.0 so diversity drives selection
        rel_scores = np.ones(n_results, dtype=np.float64)

    # Pre-compute norms for efficient cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-10, 1e-10, norms)
    emb_normed = emb_matrix / norms

    # Greedy MMR selection
    selected_indices: List[int] = []
    remaining = set(range(n_results))

    # First pick: highest relevance
    first = int(np.argmax(rel_scores))
    selected_indices.append(first)
    remaining.discard(first)

    while remaining and len(selected_indices) < top_n:
        best_idx = -1
        best_mmr = -float("inf")

        # Pre-compute max similarity to selected set for all remaining
        selected_embs = emb_normed[selected_indices]  # (n_selected, dim)

        for idx in remaining:
            # Cosine similarities to all selected results
            sims = selected_embs @ emb_normed[idx]  # dot products of normed vectors
            max_sim = float(np.max(sims))

            mmr = lambda_param * rel_scores[idx] - (1 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [results[i] for i in selected_indices]
