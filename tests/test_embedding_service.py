import asyncio
import time
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

# --- Adjust sys.path to import from the 'app' directory ---
# This assumes the test script is run from the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# ---

# Now import the service and config
try:
    from app.services.embedding_service import get_embedding, batch_get_embeddings, embedding_cache, embedding_cache_lock
    from app.config import settings # Needed to check if API key is set
except ImportError as e:
    print(f"Error importing app modules: {e}")
    print("Ensure the script is run from the project root or PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Mock OpenAI Client ---
# Create a mock response structure similar to openai.types.CreateEmbeddingResponse
class MockEmbeddingData:
    def __init__(self, embedding):
        self.embedding = embedding

class MockEmbeddingResponse:
    def __init__(self, data):
        self.data = data

# Mock function to simulate openai.embeddings.create
def mock_openai_embeddings_create(*args, **kwargs):
    input_texts = kwargs.get('input', [])
    model = kwargs.get('model', 'default_mock_model')
    print(f"[Mock OpenAI] Received request for model '{model}' with {len(input_texts)} texts.")

    mock_data = []
    for i, text in enumerate(input_texts):
        # Create a deterministic mock embedding based on text length and index
        mock_embedding = [float(len(text) * (i + 1))] * 1536 # Example: [5.0, 5.0, ...]
        mock_data.append(MockEmbeddingData(embedding=mock_embedding))
        print(f"[Mock OpenAI] Generated mock embedding for: '{text[:20]}...'")

    return MockEmbeddingResponse(data=mock_data)

# --- Test Harness ---
async def run_embedding_tests():
    print("--- Starting Embedding Service Test Harness ---")

    # Clear cache before tests
    with embedding_cache_lock:
        embedding_cache.clear()
    print("Cache cleared.")

    # Test Data
    text1 = "Hello world"
    text2 = "Another sentence"
    text1_dup = "Hello world"
    texts_batch = [text1, text2, text1_dup, "Third unique sentence"]

    # --- Patch OpenAI API call ---
    # We use patch context managers to replace the actual API call within the test scope
    with patch('openai.embeddings.create', side_effect=mock_openai_embeddings_create) as mock_create:

        # --- Test 1: Single Embedding (Cache Miss) ---
        print("\n[Test 1] Single embedding (cache miss)...")
        start_time = time.time()
        emb1 = get_embedding(text1)
        duration = time.time() - start_time
        print(f"Result 1 (len={len(emb1)}, time={duration:.4f}s): {str(emb1[:3])[:-1]}...]") # Show first few elements
        assert len(emb1) == 1536
        assert emb1[0] > 0 # Check if mock value is generated
        mock_create.assert_called_once() # Should have called OpenAI once
        print("Test 1 PASSED ✅")

        # --- Test 2: Single Embedding (Cache Hit) ---
        print("\n[Test 2] Single embedding (cache hit)...")
        mock_create.reset_mock() # Reset call count
        start_time = time.time()
        emb1_cached = get_embedding(text1_dup)
        duration = time.time() - start_time
        print(f"Result 2 (len={len(emb1_cached)}, time={duration:.4f}s): {str(emb1_cached[:3])[:-1]}...]")
        assert emb1 == emb1_cached # Should be identical to the first call result
        mock_create.assert_not_called() # Should NOT have called OpenAI
        print("Test 2 PASSED ✅")

        # --- Test 3: Batch Embeddings (Mixed Cache) ---
        print("\n[Test 3] Batch embeddings (mixed cache hit/miss)...")
        mock_create.reset_mock()
        # text1 is cached, text2 and "Third unique sentence" are not
        start_time = time.time()
        batch_embs = await batch_get_embeddings(texts_batch)
        duration = time.time() - start_time
        print(f"Batch Result (count={len(batch_embs)}, time={duration:.4f}s)")
        assert len(batch_embs) == len(texts_batch)
        # Check that the cached embedding for text1 was used
        assert batch_embs[0] == emb1
        assert batch_embs[2] == emb1
        # Check that the new embeddings were generated
        assert len(batch_embs[1]) == 1536 and batch_embs[1][0] > 0
        assert len(batch_embs[3]) == 1536 and batch_embs[3][0] > 0
        # Check that OpenAI was called only for the non-cached items
        mock_create.assert_called_once()
        call_args, call_kwargs = mock_create.call_args
        assert len(call_kwargs['input']) == 2 # Called for text2 and "Third unique sentence"
        assert call_kwargs['input'][0] == text2
        assert call_kwargs['input'][1] == "Third unique sentence"
        print("Test 3 PASSED ✅")

        # --- Test 4: Batch Embeddings (All Cached) ---
        print("\n[Test 4] Batch embeddings (all cached)...")
        mock_create.reset_mock()
        start_time = time.time()
        batch_embs_cached = await batch_get_embeddings([text2, text1]) # Use items now in cache
        duration = time.time() - start_time
        print(f"Batch Result (count={len(batch_embs_cached)}, time={duration:.4f}s)")
        assert len(batch_embs_cached) == 2
        assert batch_embs_cached[0] == batch_embs[1] # Compare with previous batch result for text2
        assert batch_embs_cached[1] == emb1
        mock_create.assert_not_called() # Should not call OpenAI
        print("Test 4 PASSED ✅")

        # --- Test 5: Invalid Input ---
        print("\n[Test 5] Invalid input handling...")
        mock_create.reset_mock()
        zero_vector = [0.0] * 1536
        # Test single invalid input
        invalid_emb_single = get_embedding("")
        assert invalid_emb_single == zero_vector
        invalid_emb_single_none = get_embedding(None)
        assert invalid_emb_single_none == zero_vector
        # Test batch invalid input
        invalid_batch = await batch_get_embeddings(["valid", None, ""])
        assert len(invalid_batch) == 3
        assert invalid_batch[0][0] > 0 # Valid embedding
        assert invalid_batch[1] == zero_vector
        assert invalid_batch[2] == zero_vector
        mock_create.assert_called_once() # Should only be called for "valid"
        call_args, call_kwargs = mock_create.call_args
        assert call_kwargs['input'] == ["valid"]
        print("Test 5 PASSED ✅")

    print("\n--- Embedding Service Test Harness Finished ---")

# --- Run the Test Harness ---
if __name__ == "__main__":
    # Check if API key is configured (though it's mocked, the service checks it)
    if not settings.OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY is not set, but proceeding as API calls are mocked.")
        # Temporarily set it for the service check if needed, or modify service check
        # settings.OPENAI_API_KEY = "mock_key_for_test" # Example if needed

    asyncio.run(run_embedding_tests())