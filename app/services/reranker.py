import logging
import torch
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers.cross_encoder import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks candidate documents based on relevance to a query using a Cross-Encoder model.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', device: Optional[str] = None):
        """
        Initializes the CrossEncoderReranker, deferring model loading.

        Args:
            model_name: The name of the Cross-Encoder model to load from sentence-transformers.
            device: The device to run the model on ('cuda', 'cpu', or None for auto-detection).
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[CrossEncoder] = None # Model is loaded later
        logger.info(f"CrossEncoderReranker initialized for model '{self.model_name}' on device '{self.device}'. Model loading deferred.")

    async def load_model(self):
        """Loads the CrossEncoder model asynchronously."""
        if self.model:
            logger.info("CrossEncoder model already loaded.")
            return True # Already loaded

        logger.info(f"Attempting to load CrossEncoder model '{self.model_name}' asynchronously...")
        try:
            # Loading models can be blocking, run in a thread
            import asyncio
            self.model = await asyncio.to_thread(
                CrossEncoder, self.model_name, device=self.device
            )
            logger.info(f"Successfully loaded CrossEncoder model '{self.model_name}'.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Cross-Encoder model '{self.model_name}' asynchronously: {e}", exc_info=True)
            self.model = None # Ensure model is None on failure
            # Optionally raise the error or return False
            # raise RuntimeError(f"Could not load CrossEncoder model: {self.model_name}") from e
            return False

    async def rerank(self, query: str, results: List[Dict[str, Any]], top_n: int = 15) -> List[Dict[str, Any]]:
        """
        Reranks a list of results based on their relevance to the query.

        Args:
            query: The user's query string.
            results: A list of dictionaries, where each dictionary represents a retrieved item
                     and must contain a 'text' key for reranking. Example:
                     [{'id': 'doc1', 'text': '...', 'source': 'vector', 'fusion_score': 0.8}, ...]
            top_n: The maximum number of reranked results to return.

        Returns:
            A list of the top_n results, sorted by the cross-encoder relevance score.
            Each dictionary will have an added 'rerank_score' key.
        """
        if not results:
            logger.info("No results provided to rerank.")
            return []
        if not query:
             logger.warning("Empty query provided for reranking. Returning original results.")
             return results[:top_n]

        logger.info(f"Starting reranking for {len(results)} results. Query: '{query[:100]}...'")

        # Prepare pairs for the cross-encoder: [query, passage_text]
        sentence_pairs: List[Tuple[str, str]] = []
        valid_results_indices: List[int] = [] # Keep track of indices of results with text

        for i, res in enumerate(results):
            text = res.get('text')
            if isinstance(text, str) and text.strip():
                sentence_pairs.append((query, text))
                valid_results_indices.append(i)
            else:
                logger.warning(f"Skipping result index {i} for reranking due to missing/invalid 'text': ID={res.get('id', 'N/A')}")

        if not sentence_pairs:
             logger.warning("No valid results with text found to rerank.")
             return []

        # Ensure model is loaded before prediction
        if not self.model:
             logger.error("❌ Reranker model is not loaded. Cannot perform reranking.")
             # Fallback: return original results sorted by fusion_score if available
             results.sort(key=lambda x: x.get('fusion_score', 0.0), reverse=True)
             return results[:top_n]

        # Predict scores using the cross-encoder model
        try:
            logger.info(f"Predicting rerank scores for {len(sentence_pairs)} valid pairs...")
            # The predict method handles batching internally based on model config
            # Run prediction in a thread as it can be CPU/GPU intensive
            import asyncio
            scores = await asyncio.to_thread(
                 self.model.predict, sentence_pairs, show_progress_bar=False, batch_size=32
            )
            # scores = self.model.predict(sentence_pairs, show_progress_bar=False, batch_size=32) # Original sync call
            logger.info(f"Predicted {len(scores)} scores successfully.")

            if len(scores) != len(valid_results_indices):
                 logger.error(f"Mismatch between number of scores ({len(scores)}) and valid results ({len(valid_results_indices)}). Cannot proceed with reranking.")
                 # Return original results as fallback? Or raise error? Returning original for now.
                 return results[:top_n]

            # Add scores back to the corresponding valid results
            reranked_results = []
            for i, score in enumerate(scores):
                original_index = valid_results_indices[i]
                result_copy = results[original_index].copy() # Avoid modifying original list items directly
                result_copy['rerank_score'] = float(score) # Ensure score is float
                reranked_results.append(result_copy)

            # Sort the results based on the new rerank_score (higher is better)
            reranked_results.sort(key=lambda x: x.get('rerank_score', -float('inf')), reverse=True)

            logger.info(f"Reranking complete. Returning top {min(top_n, len(reranked_results))} results.")
            return reranked_results[:top_n]

        except Exception as e:
            logger.error(f"❌ Error during cross-encoder prediction or processing: {e}", exc_info=True)
            # Fallback: return the original results sorted by fusion_score if available
            results.sort(key=lambda x: x.get('fusion_score', 0.0), reverse=True)
            return results[:top_n]


if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO) # Show logs for example

    # Dummy results from a hypothetical merger
    dummy_results = [
        {'id': 'vec1', 'text': 'Paris is the capital of France.', 'source': 'vector', 'fusion_score': 0.85},
        {'id': 'graph1', 'text': 'France is a country in Europe.', 'source': 'graph', 'fusion_score': 0.7},
        {'id': 'vec2', 'text': 'The Eiffel Tower is in Paris.', 'source': 'vector', 'fusion_score': 0.8},
        {'id': 'vec3', 'text': 'What is the primary language spoken in France?', 'source': 'vector', 'fusion_score': 0.6},
        {'id': 'graph2', 'text': 'Paris has many famous landmarks.', 'source': 'graph', 'fusion_score': 0.75},
        {'id': 'invalid', 'text': None, 'source': 'vector', 'fusion_score': 0.5}, # Invalid text
    ]

    query = "Tell me about landmarks in the capital of France."

    try:
        reranker = CrossEncoderReranker()
        # In a real async app, you'd await load_model() first
        # For this sync example, we assume it loads or handle error in rerank
        # Example doesn't run async, so rerank will likely fail if model not preloaded
        # top_results = reranker.rerank(query, dummy_results, top_n=3) # This would need await in async context

        print("\n--- Reranked Results (Example - requires async execution) ---")
        # for i, res in enumerate(top_results):
        #     print(f"{i+1}. ID: {res.get('id', 'N/A')}, Score: {res.get('rerank_score', 'N/A'):.4f}, Text: {res.get('text')}")
        print("Example usage requires async execution to load model and rerank.")

    except RuntimeError as e:
         print(f"Could not run example: {e}")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")