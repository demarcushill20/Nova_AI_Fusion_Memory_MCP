import asyncio
import sys
import os
import time
from unittest.mock import patch, MagicMock, AsyncMock

# --- Adjust sys.path to import from the 'app' directory ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# ---

# Import the service to be tested and its dependencies (for mocking)
try:
    from app.services.memory_service import MemoryService
    # Import classes that MemoryService depends on, to mock them
    from app.services.pinecone_client import PineconeClient
    from app.services.graph_client import GraphClient
    from app.services.embedding_service import get_embedding # Used directly? Check memory_service.py - yes, via asyncio.to_thread
    from app.services.query_router import QueryRouter, RoutingMode
    from app.services.hybrid_merger import HybridMerger
    from app.services.reranker import CrossEncoderReranker
    from app.config import settings # Needed if service uses settings directly
except ImportError as e:
    print(f"Error importing app modules: {e}")
    print("Ensure the script is run from the project root or PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Mock Dependencies ---

# Mock PineconeClient methods
mock_pinecone_client = MagicMock(spec=PineconeClient)
mock_pinecone_client.initialize = AsyncMock(return_value=True) # Make initialize async
mock_pinecone_client.query_vector = MagicMock(return_value=[ # Simulate sync query_vector called via to_thread
    {'id': 'vec1', 'score': 0.9, 'metadata': {'text': 'Vector result 1 text'}}
])
mock_pinecone_client.upsert_vector = MagicMock(return_value=True) # Simulate sync upsert_vector called via to_thread
mock_pinecone_client.delete_vector = MagicMock(return_value=True) # Simulate sync delete_vector called via to_thread
mock_pinecone_client.check_connection = MagicMock(return_value=True) # Simulate sync check_connection called via to_thread

# Mock GraphClient methods
mock_graph_client = MagicMock(spec=GraphClient)
mock_graph_client.initialize = AsyncMock(return_value=True)
mock_graph_client.query_graph = AsyncMock(return_value=[ # query_graph is async
    {'id': 'graph1', 'text': 'Graph result 1 text', 'source': 'graph', 'score': 0.0, 'metadata': {}}
])
mock_graph_client.upsert_graph_data = AsyncMock(return_value=True) # upsert_graph_data is async
mock_graph_client.delete_graph_data = AsyncMock(return_value=True) # delete_graph_data is async
mock_graph_client.check_connection = AsyncMock(return_value=True) # check_connection is async
mock_graph_client.close = AsyncMock(return_value=None)

# Mock Embedding Service function (get_embedding is called via to_thread)
# We patch the function directly where it's imported in memory_service
mock_get_embedding = MagicMock(return_value=[0.1] * 1536) # Simulate sync get_embedding

# Mock Nova Modules
mock_query_router = MagicMock(spec=QueryRouter)
mock_query_router.route = MagicMock(return_value=RoutingMode.HYBRID)

mock_hybrid_merger = MagicMock(spec=HybridMerger)
# Simulate the structure merge_results returns (list of dicts with 'text')
mock_hybrid_merger.merge_results = MagicMock(return_value=[
    {'id': 'vec1', 'text': 'Vector result 1 text', 'source': 'vector', 'rrf_score': 0.01639},
    {'id': 'graph1', 'text': 'Graph result 1 text', 'source': 'graph', 'rrf_score': 0.01639},
]) # Simulate sync merge_results called via to_thread

mock_reranker = MagicMock(spec=CrossEncoderReranker)
mock_reranker.load_model = AsyncMock(return_value=True)
# Simulate the structure rerank returns
mock_reranker.rerank = AsyncMock(return_value=[
     {'id': 'vec1', 'text': 'Vector result 1 text', 'source': 'vector', 'rrf_score': 0.01639, 'rerank_score': 5.5},
     {'id': 'graph1', 'text': 'Graph result 1 text', 'source': 'graph', 'rrf_score': 0.01639, 'rerank_score': 4.5},
])

# --- Test Harness ---
async def run_memory_service_tests():
    print("--- Starting Memory Service Test Harness ---")

    # Patch the dependencies *before* MemoryService is instantiated if they are created in __init__
    # Or patch the instances *after* instantiation if they are passed in or assigned later.
    # MemoryService creates instances in __init__, so we patch the classes/functions it imports.
    with patch('app.services.memory_service.PineconeClient', return_value=mock_pinecone_client), \
         patch('app.services.memory_service.GraphClient', return_value=mock_graph_client), \
         patch('app.services.memory_service.get_embedding', side_effect=mock_get_embedding) as patched_get_embedding, \
         patch('app.services.memory_service.QueryRouter', return_value=mock_query_router), \
         patch('app.services.memory_service.HybridMerger', return_value=mock_hybrid_merger), \
         patch('app.services.memory_service.CrossEncoderReranker', return_value=mock_reranker) as patched_reranker_class:

        # Instantiate the service *within the patch context*
        service = MemoryService()
        # Manually trigger async initialization
        await service.initialize()
        print("MemoryService initialized with mocked dependencies.")

        # Reset mocks before each test section
        mock_pinecone_client.reset_mock()
        mock_graph_client.reset_mock()
        patched_get_embedding.reset_mock()
        mock_query_router.reset_mock()
        mock_hybrid_merger.reset_mock()
        mock_reranker.reset_mock()
        # Reset the mock for the reranker instance created within MemoryService
        # (Accessing the instance directly - might be fragile if init changes)
        if service.reranker:
             service.reranker.rerank.reset_mock()


        # --- Test 1: Perform Query ---
        print("\n[Test 1] Perform Query...")
        query = "Test query about vectors and graphs"
        start_time = time.time()
        results = await service.perform_query(query)
        duration = time.time() - start_time
        print(f"Query Result (count={len(results)}, time={duration:.4f}s): {results}")

        # Assertions
        mock_query_router.route.assert_called_once_with(query)
        patched_get_embedding.assert_called_once_with(query, service.embedding_model_name)
        mock_pinecone_client.query_vector.assert_called_once()
        mock_graph_client.query_graph.assert_called_once()
        mock_hybrid_merger.merge_results.assert_called_once()
        # Check if reranker was called (it should be, given the mocks)
        if service.reranker and service._reranker_loaded:
             service.reranker.rerank.assert_called_once()
        assert len(results) == 2 # Based on mock reranker return value
        assert results[0]['id'] == 'vec1' # Check order from reranker
        print("Test 1 PASSED ✅")

        # --- Test 2: Perform Upsert (No ID provided) ---
        print("\n[Test 2] Perform Upsert (auto-ID)...")
        mock_pinecone_client.reset_mock()
        mock_graph_client.reset_mock()
        patched_get_embedding.reset_mock()
        content = "New memory content for upsert."
        metadata = {"topic": "testing"}
        start_time = time.time()
        upsert_id = await service.perform_upsert(content=content, metadata=metadata)
        duration = time.time() - start_time
        print(f"Upsert Result (ID={upsert_id}, time={duration:.4f}s)")

        # Assertions
        assert upsert_id is not None # Should return the generated ID
        patched_get_embedding.assert_called_once_with(content, service.embedding_model_name)
        mock_pinecone_client.upsert_vector.assert_called_once()
        mock_graph_client.upsert_graph_data.assert_called_once()
        # Check args passed to upsert calls
        pinecone_args, _ = mock_pinecone_client.upsert_vector.call_args
        assert pinecone_args[0] == upsert_id # Check ID
        assert pinecone_args[2]['text'] == content # Check metadata includes text
        assert pinecone_args[2]['topic'] == "testing"
        graph_args, _ = mock_graph_client.upsert_graph_data.call_args
        assert graph_args[0] == upsert_id # Check ID
        assert graph_args[1] == content
        assert graph_args[2] == metadata
        print("Test 2 PASSED ✅")

        # --- Test 3: Perform Upsert (With ID provided) ---
        print("\n[Test 3] Perform Upsert (provided ID)...")
        mock_pinecone_client.reset_mock()
        mock_graph_client.reset_mock()
        patched_get_embedding.reset_mock()
        provided_id = "custom-id-123"
        content_custom = "Another memory with custom ID."
        start_time = time.time()
        upsert_id_custom = await service.perform_upsert(content=content_custom, memory_id=provided_id)
        duration = time.time() - start_time
        print(f"Upsert Result (ID={upsert_id_custom}, time={duration:.4f}s)")

        # Assertions
        assert upsert_id_custom == provided_id
        patched_get_embedding.assert_called_once_with(content_custom, service.embedding_model_name)
        mock_pinecone_client.upsert_vector.assert_called_once()
        mock_graph_client.upsert_graph_data.assert_called_once()
        pinecone_args_custom, _ = mock_pinecone_client.upsert_vector.call_args
        assert pinecone_args_custom[0] == provided_id
        graph_args_custom, _ = mock_graph_client.upsert_graph_data.call_args
        assert graph_args_custom[0] == provided_id
        print("Test 3 PASSED ✅")

        # --- Test 4: Perform Delete ---
        print("\n[Test 4] Perform Delete...")
        mock_pinecone_client.reset_mock()
        mock_graph_client.reset_mock()
        delete_id = "id-to-delete"
        start_time = time.time()
        delete_success = await service.perform_delete(delete_id)
        duration = time.time() - start_time
        print(f"Delete Result (Success={delete_success}, time={duration:.4f}s)")

        # Assertions
        assert delete_success is True
        mock_pinecone_client.delete_vector.assert_called_once_with(delete_id)
        mock_graph_client.delete_graph_data.assert_called_once_with(delete_id)
        print("Test 4 PASSED ✅")

        # --- Test 5: Health Check ---
        print("\n[Test 5] Health Check...")
        start_time = time.time()
        health = await service.check_health()
        duration = time.time() - start_time
        print(f"Health Result (time={duration:.4f}s): {health}")

        # Assertions
        mock_pinecone_client.check_connection.assert_called_once()
        mock_graph_client.check_connection.assert_called_once()
        assert health['status'] == 'ok'
        assert health['pinecone'] == 'ok'
        assert health['graph'] == 'ok'
        assert health['reranker'] == 'loaded' # Based on mock
        print("Test 5 PASSED ✅")

        # --- Test 6: Query with Reranker Disabled ---
        print("\n[Test 6] Query with Reranker Disabled...")
        # Simulate reranker failing to load
        service.reranker = None
        service._reranker_loaded = False
        mock_pinecone_client.reset_mock()
        mock_graph_client.reset_mock()
        patched_get_embedding.reset_mock()
        mock_query_router.reset_mock()
        mock_hybrid_merger.reset_mock()

        results_no_rerank = await service.perform_query(query)
        print(f"Query Result (No Reranker): {results_no_rerank}")
        # Assertions
        mock_hybrid_merger.merge_results.assert_called_once()
        # Reranker mock should NOT have been called
        # Accessing the mock instance directly might be tricky if it's None
        # Instead, check the log message or the final result structure/length
        assert len(results_no_rerank) == 2 # Should return fused results (mock returns 2)
        assert 'rerank_score' not in results_no_rerank[0] # Rerank score shouldn't be added
        print("Test 6 PASSED ✅")

        # Restore reranker for subsequent tests if needed (though none follow here)
        # service.reranker = mock_reranker # Restore if needed
        # service._reranker_loaded = True

    print("\n--- Memory Service Test Harness Finished ---")

# --- Run the Test Harness ---
if __name__ == "__main__":
    # Set logging level to INFO or DEBUG to see test outputs
    import logging
    logging.basicConfig(level=logging.INFO)
    # Adjust logger levels for specific modules if needed
    logging.getLogger('app.services.memory_service').setLevel(logging.DEBUG)

    # Run the async test function
    asyncio.run(run_memory_service_tests())