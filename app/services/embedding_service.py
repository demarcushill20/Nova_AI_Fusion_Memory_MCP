import logging
import asyncio
import threading
from typing import List, Optional
from functools import lru_cache
import openai
from cachetools import TTLCache

# Import settings from the config module
try:
    from ..config import settings
except ImportError:
    print("Error: Could not import settings from app.config. Ensure the file exists and is configured.")
    # Fallback or raise error - Raising for clarity during development
    raise

logger = logging.getLogger(__name__)

# --- OpenAI Client Initialization ---
# Ensure the API key is set before making calls
if not settings.OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY not found in settings. Please set it in your environment or .env file.")
    # Optionally raise an error or handle appropriately
    # raise ValueError("OpenAI API Key is not configured.")
else:
    openai.api_key = settings.OPENAI_API_KEY
    logger.info("OpenAI API key configured.")

# --- Embedding Cache ---
# Using TTLCache: items expire after 24 hours (86400 seconds), max size 1000 items
embedding_cache = TTLCache(maxsize=1000, ttl=86400)
embedding_cache_lock = threading.RLock() # Thread-safe access

# --- Embedding Functions ---

# Note: @lru_cache might not be ideal here if we want thread-safety and TTL from TTLCache.
# We will implement the caching logic manually within the function using TTLCache.
# @lru_cache(maxsize=128) # Keep lru_cache for very frequent, short-lived calls if desired, but TTLCache is primary
def get_embedding(text: str, model: str = settings.EMBEDDING_MODEL) -> List[float]:
    """
    Retrieves the embedding for the given text using the configured OpenAI model.
    Uses a thread-safe TTL cache to reduce redundant API calls.

    Args:
        text: The input text string.
        model: The embedding model to use (defaults to settings.EMBEDDING_MODEL).

    Returns:
        The embedding vector as a list of floats. Returns a zero vector on error.
    """
    if not text or not isinstance(text, str):
        logger.warning("get_embedding called with invalid text input. Returning zero vector.")
        return [0.0] * 1536 # Dimension for ada-002

    cache_key = (text, model) # Cache key includes model

    # Thread-safe cache lookup
    with embedding_cache_lock:
        if cache_key in embedding_cache:
            logger.debug(f"Embedding cache hit for text: '{text[:50]}...'")
            return embedding_cache[cache_key]

    logger.debug(f"Embedding cache miss for text: '{text[:50]}...'. Calling OpenAI API.")
    try:
        # Ensure API key is set (might have failed during initial load)
        if not openai.api_key:
             raise ValueError("OpenAI API Key is not configured.")

        response = openai.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding

        # Thread-safe cache update
        with embedding_cache_lock:
            embedding_cache[cache_key] = embedding
        logger.debug(f"Stored embedding in cache for text: '{text[:50]}...'")
        return embedding
    except Exception as e:
        logger.error(f"❌ Error getting embedding for text '{text[:50]}...': {e}", exc_info=True)
        return [0.0] * 1536 # Return zero vector on error

async def batch_get_embeddings(texts: List[str], model: str = settings.EMBEDDING_MODEL) -> List[List[float]]:
    """
    Asynchronously retrieves embeddings for multiple texts using the configured OpenAI model.
    Uses a thread-safe TTL cache and attempts batch API calls for efficiency.

    Args:
        texts: A list of input text strings.
        model: The embedding model to use (defaults to settings.EMBEDDING_MODEL).

    Returns:
        A list of embedding vectors, corresponding to the input texts. Returns zero vectors for failed embeddings.
    """
    if not texts:
        return []

    results: List[Optional[List[float]]] = [None] * len(texts)
    texts_to_fetch: List[str] = []
    indices_to_fetch: List[int] = []

    # Check cache first
    with embedding_cache_lock:
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid text input at index {i} in batch. Will return zero vector.")
                results[i] = [0.0] * 1536 # Dimension for ada-002
                continue

            cache_key = (text, model)
            if cache_key in embedding_cache:
                logger.debug(f"Batch embedding cache hit for index {i}: '{text[:50]}...'")
                results[i] = embedding_cache[cache_key]
            else:
                texts_to_fetch.append(text)
                indices_to_fetch.append(i)

    # Fetch missing embeddings if any
    if texts_to_fetch:
        logger.debug(f"Batch embedding cache miss for {len(texts_to_fetch)} texts. Calling OpenAI API.")
        try:
            # Ensure API key is set
            if not openai.api_key:
                 raise ValueError("OpenAI API Key is not configured.")

            # Run the potentially blocking API call in a separate thread
            response = await asyncio.to_thread(
                lambda: openai.embeddings.create(input=texts_to_fetch, model=model)
            )

            # Process results and update cache
            with embedding_cache_lock:
                for i, data in enumerate(response.data):
                    original_index = indices_to_fetch[i]
                    embedding = data.embedding
                    results[original_index] = embedding
                    # Update cache
                    cache_key = (texts_to_fetch[i], model)
                    embedding_cache[cache_key] = embedding
                    logger.debug(f"Stored batch embedding in cache for index {original_index}: '{texts_to_fetch[i][:50]}...'")

        except Exception as e:
            logger.error(f"❌ Error getting batch embeddings: {e}", exc_info=True)
            # Fill failed fetches with zero vectors
            for i in indices_to_fetch:
                if results[i] is None: # Only fill if not already filled by cache
                    results[i] = [0.0] * 1536

    # Final check for any None results (e.g., if initial input was invalid) and replace with zero vectors
    final_results = [[0.0] * 1536 if res is None else res for res in results]

    return final_results

# Example usage (optional, for testing)
async def _test_embedding_service():
    print("Testing Embedding Service...")
    test_text_1 = "This is a test sentence."
    test_text_2 = "This is another test sentence."
    test_text_3 = "This is a test sentence." # Duplicate

    # Single embedding
    print(f"\nGetting embedding for: '{test_text_1}'")
    emb1 = get_embedding(test_text_1)
    print(f"Embedding 1 length: {len(emb1)}")
    # print(f"Embedding 1 (first 5): {emb1[:5]}")

    # Cached single embedding
    print(f"\nGetting embedding for (cached): '{test_text_3}'")
    emb3 = get_embedding(test_text_3)
    print(f"Embedding 3 length: {len(emb3)}")
    # print(f"Embedding 3 (first 5): {emb3[:5]}")
    # assert emb1 == emb3 # Should be the same due to cache

    # Batch embedding
    print(f"\nGetting batch embeddings for: ['{test_text_1}', '{test_text_2}', '{test_text_3}']")
    batch_texts = [test_text_1, test_text_2, test_text_3]
    batch_embs = await batch_get_embeddings(batch_texts)
    print(f"Batch embeddings received: {len(batch_embs)}")
    for i, emb in enumerate(batch_embs):
        print(f"Batch Emb {i} length: {len(emb)}")
        # print(f"Batch Emb {i} (first 5): {emb[:5]}")
    # assert batch_embs[0] == emb1 # Check cache worked in batch
    # assert batch_embs[2] == emb1

    print("\nEmbedding Service Test Complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Enable debug logging for test
    # Requires OPENAI_API_KEY to be set in environment or .env
    if settings.OPENAI_API_KEY:
         asyncio.run(_test_embedding_service())
    else:
         print("Skipping embedding service test: OPENAI_API_KEY not set.")