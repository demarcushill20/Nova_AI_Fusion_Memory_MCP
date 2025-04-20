import asyncio
import sys
import os
import time
import logging

# --- Adjust sys.path to import from the 'app' directory ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# ---

# Import the client and config
try:
    from app.services.pinecone_client import PineconeClient
    from app.config import settings
except ImportError as e:
    print(f"Error importing app modules: {e}")
    print("Ensure the script is run from the project root or PYTHONPATH is set correctly.")
    sys.exit(1)

# Configure logging for the test harness
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Test Harness ---
async def run_pinecone_integration_test():
    """
    Integration test harness for PineconeClient.
    Requires valid Pinecone credentials in .env and network access.
    WARNING: This performs REAL operations on your Pinecone index.
    """
    logger.info("--- Starting Pinecone Client Integration Test Harness ---")

    # Check for required settings
    if not settings.PINECONE_API_KEY or not settings.PINECONE_ENV:
        logger.error("❌ PINECONE_API_KEY or PINECONE_ENV not found in settings. Skipping integration test.")
        print("Skipping Pinecone integration test: API Key or Environment not set in .env")
        return

    client = PineconeClient()

    # --- Test 1: Initialization ---
    logger.info("\n[Test 1] Initializing Pinecone connection...")
    start_time = time.time()
    initialized = client.initialize() # This is synchronous in the current implementation
    duration = time.time() - start_time
    logger.info(f"Initialization attempt finished in {duration:.4f}s. Success: {initialized}")
    if not initialized:
        logger.error("❌ Initialization failed. Aborting further tests.")
        print("❌ Pinecone initialization failed. Check credentials and index status.")
        return
    print(f"Test 1 PASSED ✅ (Initialized index: {client.index_name})")

    # --- Test 2: Health Check ---
    logger.info("\n[Test 2] Checking connection health...")
    start_time = time.time()
    # Run sync check_connection in thread from async context
    is_healthy = await asyncio.to_thread(client.check_connection)
    duration = time.time() - start_time
    logger.info(f"Health check finished in {duration:.4f}s. Healthy: {is_healthy}")
    if not is_healthy:
        logger.error("❌ Health check failed.")
        print("❌ Pinecone health check failed.")
        # Optionally abort, or continue cautiously
    else:
        print("Test 2 PASSED ✅")

    # --- Test 3: Upsert ---
    test_id = "integration_test_vector_001"
    # Use a fixed, known vector dimension (1536 for ada-002)
    test_vector = [0.01] * 1536
    test_metadata = {"text": "Integration test content for Pinecone.", "category": "integration_test", "timestamp": time.time()}
    logger.info(f"\n[Test 3] Upserting vector with ID: {test_id}...")
    start_time = time.time()
    # Run sync upsert_vector in thread from async context
    upsert_success = await asyncio.to_thread(client.upsert_vector, test_id, test_vector, test_metadata)
    duration = time.time() - start_time
    logger.info(f"Upsert finished in {duration:.4f}s. Success: {upsert_success}")
    if not upsert_success:
        logger.error("❌ Upsert failed.")
        print("❌ Pinecone upsert failed.")
        # Abort if upsert fails, as subsequent tests depend on it
        return
    print("Test 3 PASSED ✅")
    logger.info("Waiting briefly for indexing...")
    await asyncio.sleep(3) # Allow time for Pinecone to index

    # --- Test 4: Query ---
    logger.info(f"\n[Test 4] Querying for vector similar to ID: {test_id}...")
    query_vector = [0.011] * 1536 # Slightly different vector
    start_time = time.time()
    # Run sync query_vector in thread from async context
    query_results = await asyncio.to_thread(client.query_vector, query_vector, top_k=5, filter={"category": "integration_test"})
    duration = time.time() - start_time
    logger.info(f"Query finished in {duration:.4f}s. Found {len(query_results)} results.")

    found_in_query = False
    if query_results:
        logger.info("Top query results:")
        for i, match in enumerate(query_results):
            logger.info(f"  {i+1}. ID: {match.get('id')}, Score: {match.get('score'):.4f}, Metadata: {match.get('metadata')}")
            if match.get('id') == test_id:
                found_in_query = True
    else:
        logger.warning("Query returned no results.")

    if not found_in_query:
        logger.error(f"❌ Failed to find upserted vector {test_id} in query results.")
        print(f"❌ Pinecone query failed to retrieve test vector {test_id}.")
        # Continue to delete attempt anyway
    else:
        print("Test 4 PASSED ✅")

    # --- Test 5: Delete ---
    logger.info(f"\n[Test 5] Deleting vector with ID: {test_id}...")
    start_time = time.time()
    # Run sync delete_vector in thread from async context
    delete_success = await asyncio.to_thread(client.delete_vector, test_id)
    duration = time.time() - start_time
    logger.info(f"Delete finished in {duration:.4f}s. Success: {delete_success}")
    if not delete_success:
        logger.error("❌ Delete operation failed.")
        print("❌ Pinecone delete failed.")
        # Continue anyway to see if query confirms deletion
    else:
        print("Test 5 PASSED ✅")
    logger.info("Waiting briefly for deletion to reflect...")
    await asyncio.sleep(3) # Allow time for deletion

    # --- Test 6: Query After Delete ---
    logger.info(f"\n[Test 6] Querying again for ID: {test_id} after deletion...")
    start_time = time.time()
    # Run sync query_vector in thread from async context
    query_results_after_delete = await asyncio.to_thread(client.query_vector, query_vector, top_k=5, filter={"category": "integration_test"})
    duration = time.time() - start_time
    logger.info(f"Query finished in {duration:.4f}s. Found {len(query_results_after_delete)} results.")

    found_after_delete = False
    if query_results_after_delete:
        for match in query_results_after_delete:
            if match.get('id') == test_id:
                found_after_delete = True
                break

    if found_after_delete:
        logger.error(f"❌ Found vector {test_id} in query results after deletion attempt.")
        print(f"❌ Pinecone query found test vector {test_id} after delete.")
    else:
        logger.info(f"Vector {test_id} correctly not found after deletion.")
        print("Test 6 PASSED ✅")

    logger.info("--- Pinecone Client Integration Test Harness Finished ---")

# --- Run the Test Harness ---
if __name__ == "__main__":
    print("Running Pinecone Integration Test Harness...")
    print("WARNING: This test performs REAL operations on your configured Pinecone index.")
    asyncio.run(run_pinecone_integration_test())