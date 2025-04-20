import logging
from typing import List, Dict, Any, Optional, Union
from hashlib import md5
from collections import defaultdict

logger = logging.getLogger(__name__)

# Placeholder for result types - adjust as needed
VectorResult = Dict[str, Any] # Example: {'id': 'vec1', 'score': 0.85, 'metadata': {...}}
GraphResult = Dict[str, Any]  # Example: {'id': 'node1', 'score': 15.2, 'metadata': {...}} or {'text': '...', 'score': ...}

def normalize_vector_score(cosine_similarity: float) -> float:
    """
    Normalizes a cosine similarity score (typically [-1, 1]) to a [0, 1] range.

    Args:
        cosine_similarity: The raw cosine similarity score.

    Returns:
        The normalized score between 0 and 1.
    """
    # Clamp the score just in case it's slightly outside the [-1, 1] range due to float precision
    clamped_score = max(-1.0, min(1.0, cosine_similarity))
    return (clamped_score + 1.0) / 2.0

def normalize_graph_score(
    raw_graph_score: float,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None
) -> float:
    """
    Normalizes a raw graph score to a [0, 1] range using min-max scaling.

    Requires knowledge of the potential minimum and maximum scores for the specific
    graph scoring method used. If min/max are not provided, it assumes the score
    is already somewhat normalized or attempts a basic scaling (less reliable).

    Args:
        raw_graph_score: The raw score from the graph retrieval system.
        min_score: The expected minimum possible score.
        max_score: The expected maximum possible score.

    Returns:
        The normalized score between 0 and 1. Returns 0.5 if min/max are equal or not provided.
    """
    if min_score is None or max_score is None:
        logger.warning("Min/max scores not provided for graph score normalization. Returning 0.5.")
        # TODO: Implement a more robust fallback? Or require min/max always?
        # For now, return a neutral score if range is unknown.
        return 0.5

    if max_score == min_score:
        logger.warning("Max score equals min score for graph normalization. Returning 0.5.")
        return 0.5 # Avoid division by zero, return neutral score

    # Clamp the score to be within the provided min/max range before scaling
    clamped_score = max(min_score, min(max_score, raw_graph_score))

    return (clamped_score - min_score) / (max_score - min_score)

# --- Hybrid Merger Implementation (Task P2.T3) ---
class HybridMerger:
    """
    Handles the merging, normalization, fusion, and deduplication of results
    from vector and graph retrieval pipelines.
    """
    def __init__(
        self,
        vector_weight: float = 0.6, # Default weight favoring vector results slightly
        graph_weight: float = 0.4,
        graph_score_min: Optional[float] = None, # Optional: Define expected graph score range
        graph_score_max: Optional[float] = None,
        rrf_k: int = 60 # Constant for RRF calculation
    ):
        """
        Initializes the HybridMerger.

        Args:
            vector_weight: Weight assigned to normalized vector scores during fusion.
            graph_weight: Weight assigned to normalized graph scores during fusion.
            graph_score_min: Expected minimum score from the graph source for normalization.
            graph_score_max: Expected maximum score from the graph source for normalization.
        """
        # if not (0 <= vector_weight <= 1 and 0 <= graph_weight <= 1):
        #     raise ValueError("Weights must be between 0 and 1.")
        # Weights are not used in RRF, but kept for potential future use or alternative modes
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.graph_score_min = graph_score_min
        self.graph_score_max = graph_score_max
        self.rrf_k = rrf_k
        logger.info(f"HybridMerger initialized with RRF k={rrf_k}. Weights (Vector={vector_weight}, Graph={graph_weight}) are currently unused by RRF.")

    # Removed _calculate_fusion_score as it's replaced by RRF logic

    def merge_results(
        self,
        vector_results: List[VectorResult],
        graph_results: List[GraphResult]
    ) -> List[Dict[str, Any]]:
        """
        Merges results from vector and graph searches using Reciprocal Rank Fusion (RRF)
        and performs basic MD5-based deduplication.

        Args:
            vector_results: List of results from the vector store (sorted by relevance, expecting 'score' and 'metadata').
            graph_results: List of results from the graph store (sorted by relevance, expecting 'score' and 'metadata' or 'text').

        Returns:
            A sorted list of merged, deduplicated results with RRF scores.
        """
        logger.info(f"Starting RRF merge process with {len(vector_results)} vector results and {len(graph_results)} graph results.")
        # --- RRF Implementation ---
        # Dictionary to store RRF scores and the best representation of each unique item
        # Key: text_hash, Value: {'rrf_score': float, 'best_result': Dict}
        rrf_scores: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'rrf_score': 0.0, 'best_result': None})

        # Process vector results (assuming they are pre-sorted by relevance)
        for rank, res in enumerate(vector_results):
            try:
                metadata = res.get('metadata', {})
                text = metadata.get('text', '')
                # MODIFICATION: Process even if text is missing, but log warning. Use ID as placeholder text.
                if not text:
                    logger.warning(f"Vector result missing 'text' in metadata: {res.get('id', 'N/A')}. Using ID as placeholder.")
                    text = f"Vector Result ID: {res.get('id', 'N/A')}" # Use ID as placeholder text

                text_hash = md5(text.encode()).hexdigest()
                rank_score = 1.0 / (self.rrf_k + rank + 1) # RRF formula (rank is 0-based)
                rrf_scores[text_hash]['rrf_score'] += rank_score

                # Store the first encountered (highest rank) result data for this hash
                if rrf_scores[text_hash]['best_result'] is None:
                     # Add normalized score for potential later use/inspection
                     raw_score = float(res.get('score', 0.0))
                     norm_score = normalize_vector_score(raw_score)
                     rrf_scores[text_hash]['best_result'] = {
                         'id': res.get('id'),
                         'text': text,
                         'source': 'vector',
                         'raw_score': raw_score,
                         'normalized_score': norm_score,
                         'metadata': metadata,
                         'initial_rank_vector': rank + 1 # Store 1-based rank
                     }
                     logger.debug(f"Added/Updated vector result hash {text_hash} with rank {rank+1}, score {rank_score:.4f}")
            except (ValueError, TypeError) as e:
                 logger.warning(f"Skipping invalid vector result due to score error: {res}. Error: {e}")
            except Exception as e:
                 logger.warning(f"Skipping vector result due to unexpected error: {res}. Error: {e}", exc_info=True)

        # Process graph results (assuming they are pre-sorted by relevance)
        for rank, res in enumerate(graph_results):
             try:
                 metadata = res.get('metadata', {})
                 text = res.get('text') or metadata.get('text', '')
                 # MODIFICATION: Process even if text is missing, but log warning. Use ID as placeholder text.
                 if not text:
                     logger.warning(f"Graph result missing 'text': {res.get('id', 'N/A')}. Using ID as placeholder.")
                     text = f"Graph Result ID: {res.get('id', 'N/A')}" # Use ID as placeholder text

                 text_hash = md5(text.encode()).hexdigest()
                 rank_score = 1.0 / (self.rrf_k + rank + 1) # RRF formula (rank is 0-based)
                 rrf_scores[text_hash]['rrf_score'] += rank_score

                 # Store the result data if it's the first time seeing this hash
                 # or if this graph result has a higher rank than a previously seen vector result
                 if rrf_scores[text_hash]['best_result'] is None:
                      raw_score = float(res.get('score', 0.0))
                      norm_score = normalize_graph_score(raw_score, self.graph_score_min, self.graph_score_max)
                      rrf_scores[text_hash]['best_result'] = {
                          'id': res.get('id'),
                          'text': text,
                          'source': 'graph',
                          'raw_score': raw_score,
                          'normalized_score': norm_score,
                          'metadata': metadata,
                          'initial_rank_graph': rank + 1 # Store 1-based rank
                      }
                      logger.debug(f"Added/Updated graph result hash {text_hash} with rank {rank+1}, score {rank_score:.4f}")
                 elif 'initial_rank_graph' not in rrf_scores[text_hash]['best_result']: # If vector result was stored first
                      # Update the source rank if this is the first time seeing it from graph
                      logger.debug(f"Updating existing result hash {text_hash} with graph rank {rank+1}")
                      rrf_scores[text_hash]['best_result']['initial_rank_graph'] = rank + 1


             except (ValueError, TypeError) as e:
                  logger.warning(f"Skipping invalid graph result due to score error: {res}. Error: {e}")
             except Exception as e:
                  logger.warning(f"Skipping graph result due to unexpected error: {res}. Error: {e}", exc_info=True)

        # Prepare final list of results from the rrf_scores dictionary
        final_results = []
        for data in rrf_scores.values():
            if data['best_result']: # Ensure we have stored result data
                # Add the final RRF score to the result dictionary
                data['best_result']['rrf_score'] = data['rrf_score']
                final_results.append(data['best_result'])

        # Sort the final list by RRF score in descending order
        final_results.sort(key=lambda x: x.get('rrf_score', 0.0), reverse=True)

        logger.info(f"RRF merge process complete. Returning {len(final_results)} unique sorted results.")
        return final_results

if __name__ == '__main__':
    # Example Usage/Testing
    print("Testing Normalization Functions:")

    # Vector Score Normalization
    print(f"Cosine 1.0 -> Normalized: {normalize_vector_score(1.0)}")
    print(f"Cosine 0.5 -> Normalized: {normalize_vector_score(0.5)}")
    print(f"Cosine 0.0 -> Normalized: {normalize_vector_score(0.0)}")
    print(f"Cosine -0.5 -> Normalized: {normalize_vector_score(-0.5)}")
    print(f"Cosine -1.0 -> Normalized: {normalize_vector_score(-1.0)}")
    print(f"Cosine 1.1 (Clamped) -> Normalized: {normalize_vector_score(1.1)}")
    print(f"Cosine -1.1 (Clamped) -> Normalized: {normalize_vector_score(-1.1)}")

    print("\nGraph Score Normalization:")
    print(f"Score 75 (Min 0, Max 100) -> Normalized: {normalize_graph_score(75, 0, 100)}")
    print(f"Score 100 (Min 0, Max 100) -> Normalized: {normalize_graph_score(100, 0, 100)}")
    print(f"Score 0 (Min 0, Max 100) -> Normalized: {normalize_graph_score(0, 0, 100)}")
    print(f"Score 120 (Clamped, Min 0, Max 100) -> Normalized: {normalize_graph_score(120, 0, 100)}")
    print(f"Score -10 (Clamped, Min 0, Max 100) -> Normalized: {normalize_graph_score(-10, 0, 100)}")
    print(f"Score 50 (Min 50, Max 50) -> Normalized: {normalize_graph_score(50, 50, 50)}") # Edge case
    print(f"Score 50 (No Min/Max) -> Normalized: {normalize_graph_score(50)}") # Edge case