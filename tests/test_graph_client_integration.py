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
    from app.services.graph_client import GraphClient, NEO4J_NODE_LABEL
    from app.config import settings
except ImportError as e:
    print(f"Error importing app modules: {e}")
    print("Ensure the script is run from the project root or PYTHONPATH is set correctly.")
    sys.exit(1)

# Configure logging for the test harness
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Test Harness ---
async def run_graph_integration_test():
    """
    Integration test harness for GraphClient (Neo4j).
    Requires a running Neo4j instance and valid credentials in .env.
    WARNING: This performs REAL operations on your Neo4j database.
    """
    logger.info("--- Starting Graph Client Integration Test Harness ---")

    # Check for required settings
    if not settings.NEO4J_PASSWORD: # Password is the most critical indicator
        logger.error("❌ NEO4J_PASSWORD not found in settings. Skipping integration test.")
        print("Skipping Neo4j integration test: NEO4J_PASSWORD not set in .env")
        return

    client = GraphClient()

    # --- Test 1: Initialization and Constraint Check ---
    logger.info("\n[Test 1] Initializing Neo4j connection and ensuring constraints...")
    start_time = time.time()
    initialized = await client.initialize()
    duration = time.time() - start_time
    logger.info(f"Initialization attempt finished in {duration:.4f}s. Success: {initialized}")
    if not initialized:
        logger.error("❌ Initialization failed. Aborting further tests.")
        print("❌ Neo4j initialization failed. Check connection details and DB status.")
        return
    print(f"Test 1 PASSED ✅ (Initialized DB: {settings.NEO4J_DATABASE}, Ensured constraint on :{NEO4J_NODE_LABEL}(entity_id))")

    # --- Test 2: Health Check ---
    logger.info("\n[Test 2] Checking connection health...")
    start_time = time.time()
    is_healthy = await client.check_connection()
    duration = time.time() - start_time
    logger.info(f"Health check finished in {duration:.4f}s. Healthy: {is_healthy}")
    if not is_healthy:
        logger.error("❌ Health check failed.")
        print("❌ Neo4j health check failed.")
        # Optionally abort, or continue cautiously
    else:
        print("Test 2 PASSED ✅")

    # --- Test 3: Upsert ---
    test_id = "integration_test_graph_node_002"
    test_content = "Neo4j integration test content."
    test_metadata = {"type": "test_data", "runner": "harness", "timestamp": time.time()}
    logger.info(f"\n[Test 3] Upserting graph node with entity_id: {test_id}...")
    start_time = time.time()
    upsert_success = await client.upsert_graph_data(test_id, test_content, test_metadata)
    duration = time.time() - start_time
    logger.info(f"Upsert finished in {duration:.4f}s. Success: {upsert_success}")
    if not upsert_success:
        logger.error("❌ Upsert failed.")
        print("❌ Neo4j upsert failed.")
        # Abort if upsert fails, as subsequent tests depend on it
        await client.close() # Close connection before exiting
        return
    print("Test 3 PASSED ✅")
    # No explicit wait needed for Neo4j usually

    # --- Test 4: Query (Simple Retrieval) ---
    # Note: The current query_graph retrieves recent nodes, not specifically by ID or content.
    # We will check if our inserted node is among the results.
    logger.info(f"\n[Test 4] Querying graph nodes (limit 10)...")
    start_time = time.time()
    query_results = await client.query_graph(query_text="dummy", top_k=10) # Query text unused here
    duration = time.time() - start_time
    logger.info(f"Query finished in {duration:.4f}s. Found {len(query_results)} results.")

    found_in_query = False
    retrieved_node = None
    if query_results:
        logger.info("Sample query results:")
        for i, node in enumerate(query_results):
            logger.info(f"  {i+1}. ID: {node.get('id')}, Text: {node.get('text', '')[:50]}..., Metadata: {node.get('metadata')}")
            if node.get('id') == test_id:
                found_in_query = True
                retrieved_node = node
                break # Found it
    else:
        logger.warning("Query returned no results.")

    if not found_in_query:
        logger.error(f"❌ Failed to find upserted node {test_id} in query results.")
        print(f"❌ Neo4j query failed to retrieve test node {test_id}.")
        # Continue to delete attempt anyway
    else:
        logger.info(f"Found test node {test_id}. Verifying content...")
        assert retrieved_node is not None
        assert retrieved_node.get('text') == test_content
        assert retrieved_node.get('metadata', {}).get('type') == "test_data"
        assert retrieved_node.get('metadata', {}).get('runner') == "harness"
        print("Test 4 PASSED ✅")

    # --- Test 5: Delete ---
    logger.info(f"\n[Test 5] Deleting graph node with entity_id: {test_id}...")
    start_time = time.time()
    delete_success = await client.delete_graph_data(test_id)
    duration = time.time() - start_time
    logger.info(f"Delete finished in {duration:.4f}s. Success: {delete_success}")
    if not delete_success:
        logger.error("❌ Delete operation failed.")
        print("❌ Neo4j delete failed.")
        # Continue anyway to see if query confirms deletion
    else:
        print("Test 5 PASSED ✅")

    # --- Test 6: Query After Delete ---
    logger.info(f"\n[Test 6] Querying again for ID: {test_id} after deletion...")
    start_time = time.time()
    query_results_after_delete = await client.query_graph(query_text="dummy", top_k=10)
    duration = time.time() - start_time
    logger.info(f"Query finished in {duration:.4f}s. Found {len(query_results_after_delete)} results.")

    found_after_delete = False
    if query_results_after_delete:
        for node in query_results_after_delete:
            if node.get('id') == test_id:
                found_after_delete = True
                break

    if found_after_delete:
        logger.error(f"❌ Found node {test_id} in query results after deletion attempt.")
        print(f"❌ Neo4j query found test node {test_id} after delete.")
    else:
        logger.info(f"Node {test_id} correctly not found after deletion.")
        print("Test 6 PASSED ✅")

    # --- Cleanup ---
    await client.close()
    logger.info("--- Graph Client Integration Test Harness Finished ---")

# --- Run the Test Harness ---
if __name__ == "__main__":
    print("Running Neo4j Integration Test Harness...")
    print("WARNING: This test performs REAL operations on your configured Neo4j database.")
    print(f"Targeting Database: {settings.NEO4J_DATABASE} at {settings.NEO4J_URI}")
    asyncio.run(run_graph_integration_test())