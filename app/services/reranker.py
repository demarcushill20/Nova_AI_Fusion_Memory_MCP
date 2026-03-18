import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple, Optional

# Optional imports — CrossEncoder may not be installed or may fail on low-RAM VPS
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    _CROSSENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None  # type: ignore[misc,assignment]
    _CROSSENCODER_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# Pinecone import for the hosted reranker
try:
    from pinecone import Pinecone
    _PINECONE_AVAILABLE = True
except ImportError:
    Pinecone = None  # type: ignore[misc,assignment]
    _PINECONE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidate documents based on relevance to a query using a Cross-Encoder model.

    Supports lazy loading with retry: the model is loaded on first rerank call,
    with up to 3 attempts before giving up permanently.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', device: Optional[str] = None):
        """
        Initializes the CrossEncoderReranker, deferring model loading.

        Args:
            model_name: The name of the Cross-Encoder model to load from sentence-transformers.
            device: The device to run the model on ('cuda', 'cpu', or None for auto-detection).
        """
        self.model_name = model_name
        if device:
            self.device = device
        elif _TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model: Optional[Any] = None  # CrossEncoder instance or None
        self._loaded = False
        self._load_attempts = 0
        self._load_lock: Optional[asyncio.Lock] = None
        logger.info(f"CrossEncoderReranker initialized for model '{self.model_name}' on device '{self.device}'. Model loading deferred.")

    async def load_model(self) -> bool:
        """Loads the CrossEncoder model asynchronously with structured logging."""
        if self._loaded and self.model:
            logger.info("CrossEncoder model already loaded.")
            return True

        if not _CROSSENCODER_AVAILABLE:
            logger.error("RERANKER_LOAD status=error error_type=ImportError msg='sentence-transformers not installed'")
            return False

        logger.info(f"Attempting to load CrossEncoder model '{self.model_name}' asynchronously...")

        # Capture memory before load
        mem_before_mb = 0
        if _PSUTIL_AVAILABLE:
            mem_before_mb = psutil.Process().memory_info().rss // (1024 * 1024)

        t0 = time.time()
        try:
            self.model = await asyncio.to_thread(
                CrossEncoder, self.model_name, device=self.device
            )
            duration_ms = int((time.time() - t0) * 1000)

            # Capture memory after load
            mem_after_mb = 0
            if _PSUTIL_AVAILABLE:
                mem_after_mb = psutil.Process().memory_info().rss // (1024 * 1024)

            logger.info(
                f"RERANKER_LOAD duration_ms={duration_ms} mem_before_mb={mem_before_mb} "
                f"mem_after_mb={mem_after_mb} status=ok"
            )
            self._loaded = True
            return True
        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            error_type = type(e).__name__

            mem_after_mb = 0
            if _PSUTIL_AVAILABLE:
                mem_after_mb = psutil.Process().memory_info().rss // (1024 * 1024)

            logger.error(
                f"RERANKER_LOAD duration_ms={duration_ms} mem_before_mb={mem_before_mb} "
                f"mem_after_mb={mem_after_mb} status=error error_type={error_type} msg='{e}'"
            )
            self.model = None
            self._loaded = False
            return False

    async def ensure_loaded(self) -> bool:
        """Lazy-load reranker on first query, with retry.

        Returns True if model is ready, False if all retries exhausted.
        Uses an asyncio.Lock to prevent concurrent model loading.
        """
        if self._loaded and self.model:
            return True
        # Lazy-init the lock (safe: only one coroutine runs at a time per event loop)
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            # Double-check after acquiring lock
            if self._loaded and self.model:
                return True
            if self._load_attempts >= 3:
                logger.warning(
                    f"CrossEncoderReranker: giving up after {self._load_attempts} failed load attempts."
                )
                return False
            self._load_attempts += 1
            logger.info(f"CrossEncoderReranker: lazy-load attempt {self._load_attempts}/3")
            return await self.load_model()

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
        valid_results_indices: List[int] = []  # Keep track of indices of results with text

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
             logger.error("Reranker model is not loaded. Cannot perform reranking.")
             # Fallback: return original results sorted by fusion_score if available
             results.sort(key=lambda x: x.get('fusion_score', 0.0), reverse=True)
             return results[:top_n]

        # Predict scores using the cross-encoder model
        try:
            logger.info(f"Predicting rerank scores for {len(sentence_pairs)} valid pairs...")
            scores = await asyncio.to_thread(
                 self.model.predict, sentence_pairs, show_progress_bar=False, batch_size=32
            )
            logger.info(f"Predicted {len(scores)} scores successfully.")

            if len(scores) != len(valid_results_indices):
                 logger.error(f"Mismatch between number of scores ({len(scores)}) and valid results ({len(valid_results_indices)}). Cannot proceed with reranking.")
                 return results[:top_n]

            # Add scores back to the corresponding valid results
            reranked_results = []
            for i, score in enumerate(scores):
                original_index = valid_results_indices[i]
                result_copy = results[original_index].copy()
                result_copy['rerank_score'] = float(score)
                reranked_results.append(result_copy)

            # Sort the results based on the new rerank_score (higher is better)
            reranked_results.sort(key=lambda x: x.get('rerank_score', -float('inf')), reverse=True)

            logger.info(f"Reranking complete. Returning top {min(top_n, len(reranked_results))} results.")
            return reranked_results[:top_n]

        except Exception as e:
            logger.error(f"Error during cross-encoder prediction or processing: {e}", exc_info=True)
            # Fallback: return the original results sorted by fusion_score if available
            results.sort(key=lambda x: x.get('fusion_score', 0.0), reverse=True)
            return results[:top_n]


class PineconeReranker:
    """Uses Pinecone's hosted rerank endpoint -- no local model needed.

    Falls back gracefully if the Pinecone SDK is missing or the API call fails.
    """

    def __init__(self, model: str = "pinecone-rerank-v0"):
        self.model = model
        self.pc: Optional[Any] = None  # Lazy-init on first call
        self._available = _PINECONE_AVAILABLE
        logger.info(f"PineconeReranker initialized with model='{self.model}' available={self._available}")

    def _ensure_client(self) -> bool:
        """Initialize Pinecone client if not yet created."""
        if self.pc is not None:
            return True
        if not self._available:
            logger.error("PineconeReranker: pinecone SDK not installed.")
            return False
        try:
            self.pc = Pinecone()  # uses PINECONE_API_KEY from env
            return True
        except Exception as e:
            logger.error(f"PineconeReranker: failed to init Pinecone client: {e}")
            self._available = False
            return False

    async def rerank(self, query: str, results: List[Dict[str, Any]], top_n: int = 15) -> List[Dict[str, Any]]:
        """Rerank results using Pinecone Inference API.

        Args:
            query: The user's query string.
            results: List of result dicts, each with at least 'text' and 'id'.
            top_n: Max number of results to return.

        Returns:
            Reranked results with 'rerank_score' added, or raises on failure.
        """
        if not results:
            return []
        if not query:
            return results[:top_n]

        if not self._ensure_client():
            raise RuntimeError("PineconeReranker: Pinecone client unavailable")

        # Build document list for the API — only include items with text
        documents = []
        valid_indices = []
        for i, r in enumerate(results):
            text = r.get("text")
            if isinstance(text, str) and text.strip():
                documents.append({"id": r.get("id", str(i)), "text": text})
                valid_indices.append(i)

        if not documents:
            logger.warning("PineconeReranker: no documents with text to rerank.")
            return []

        try:
            response = await asyncio.to_thread(
                self.pc.inference.rerank,
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                return_documents=True,
            )

            reranked = []
            for item in response.data:
                # item.index is relative to the documents list, map back to original results
                original_idx = valid_indices[item.index]
                original_copy = results[original_idx].copy()
                original_copy["rerank_score"] = item.score
                reranked.append(original_copy)

            logger.info(f"PineconeReranker: reranked {len(documents)} docs, returning top {len(reranked)}")
            return reranked

        except Exception as e:
            logger.error(f"PineconeReranker: API call failed: {e}", exc_info=True)
            raise  # Let caller handle fallback


if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)

    dummy_results = [
        {'id': 'vec1', 'text': 'Paris is the capital of France.', 'source': 'vector', 'fusion_score': 0.85},
        {'id': 'graph1', 'text': 'France is a country in Europe.', 'source': 'graph', 'fusion_score': 0.7},
        {'id': 'vec2', 'text': 'The Eiffel Tower is in Paris.', 'source': 'vector', 'fusion_score': 0.8},
        {'id': 'vec3', 'text': 'What is the primary language spoken in France?', 'source': 'vector', 'fusion_score': 0.6},
        {'id': 'graph2', 'text': 'Paris has many famous landmarks.', 'source': 'graph', 'fusion_score': 0.75},
        {'id': 'invalid', 'text': None, 'source': 'vector', 'fusion_score': 0.5},
    ]

    query = "Tell me about landmarks in the capital of France."

    try:
        reranker = CrossEncoderReranker()
        print("\n--- Reranked Results (Example - requires async execution) ---")
        print("Example usage requires async execution to load model and rerank.")
    except RuntimeError as e:
         print(f"Could not run example: {e}")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
