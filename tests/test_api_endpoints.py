import asyncio
import sys
import os
import time
import httpx # For making async HTTP requests
from unittest.mock import patch, MagicMock, AsyncMock

# --- Adjust sys.path to import from the 'app' directory ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# ---

# Import the FastAPI app and the service to mock
try:
    # Import the app instance directly to potentially use TestClient later if needed
    # from app.main import app # Using httpx for now, TestClient might be better
    from app.services.memory_service import MemoryService
    from app.api.memory_routes import memory_service_instance # Import the instance used by routes
except ImportError as e:
    print(f"Error importing app modules: {e}")
    print("Ensure the script is run from the project root or PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Mock MemoryService Methods ---
# We need to mock the *instance* used by the routes (memory_service_instance)
mock_perform_query = AsyncMock(return_value=[ # Simulate results from MemoryService.perform_query
    {'id': 'mock_res_1', 'text': 'Mocked query result 1', 'source': 'vector', 'score': 0.9, 'metadata': {}},
    {'id': 'mock_res_2', 'text': 'Mocked query result 2', 'source': 'graph', 'score': 0.8, 'metadata': {}},
])
mock_perform_upsert = AsyncMock(return_value="mock_upsert_id_123") # Simulate returning the ID
mock_perform_delete = AsyncMock(return_value=True) # Simulate successful delete
mock_check_health = AsyncMock(return_value={ # Simulate health check result
    "status": "ok", "pinecone": "ok", "graph": "ok", "reranker": "loaded"
})

# --- Test Harness ---
# Base URL for the locally running server (adjust if needed)
BASE_URL = "http://127.0.0.1:8000" # Assumes server runs on port 8000

async def run_api_tests():
    print("--- Starting API Endpoint Test Harness ---")
    print(f"Targeting server at: {BASE_URL}")
    print("NOTE: Ensure the FastAPI server (uvicorn app.main:app --port 8000) is running separately.")
    print("      This harness mocks the MemoryService, so DB connections are not required for these tests.")

    # Apply mocks to the actual MemoryService instance used by the routes
    # This is crucial for testing the API layer in isolation
    original_methods = {
        'perform_query': memory_service_instance.perform_query,
        'perform_upsert': memory_service_instance.perform_upsert,
        'perform_delete': memory_service_instance.perform_delete,
        'check_health': memory_service_instance.check_health,
    }
    memory_service_instance.perform_query = mock_perform_query
    memory_service_instance.perform_upsert = mock_perform_upsert
    memory_service_instance.perform_delete = mock_perform_delete
    memory_service_instance.check_health = mock_check_health
    # We also need to simulate that the service is initialized for the dependency check
    memory_service_instance._initialized = True

    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # --- Test 1: Health Check Endpoint ---
        print("\n[Test 1] GET /memory/health...")
        try:
            response = await client.get("/memory/health")
            response.raise_for_status() # Check for HTTP errors
            data = response.json()
            print(f"Response Status: {response.status_code}")
            print(f"Response Body: {data}")
            assert response.status_code == 200
            assert data['status'] == 'ok'
            assert data['pinecone'] == 'ok'
            assert data['graph'] == 'ok'
            assert data['reranker'] == 'loaded'
            mock_check_health.assert_awaited_once()
            print("Test 1 PASSED ✅")
        except Exception as e:
            print(f"Test 1 FAILED ❌: {e}")
        mock_check_health.reset_mock()

        # --- Test 2: Query Endpoint ---
        print("\n[Test 2] POST /memory/query...")
        query_payload = {"query": "What is mocking?"}
        try:
            response = await client.post("/memory/query", json=query_payload)
            response.raise_for_status()
            data = response.json()
            print(f"Response Status: {response.status_code}")
            print(f"Response Body: {data}")
            assert response.status_code == 200
            assert "results" in data
            assert len(data["results"]) == 2 # Based on mock_perform_query
            assert data["results"][0]["id"] == "mock_res_1"
            assert data["results"][1]["text"] == "Mocked query result 2"
            mock_perform_query.assert_awaited_once_with(query_payload["query"])
            print("Test 2 PASSED ✅")
        except Exception as e:
            print(f"Test 2 FAILED ❌: {e}")
        mock_perform_query.reset_mock()

        # --- Test 3: Upsert Endpoint ---
        print("\n[Test 3] POST /memory/upsert...")
        upsert_payload = {"content": "Test content", "metadata": {"source": "api_test"}}
        try:
            response = await client.post("/memory/upsert", json=upsert_payload)
            response.raise_for_status()
            data = response.json()
            print(f"Response Status: {response.status_code}")
            print(f"Response Body: {data}")
            assert response.status_code == 200
            assert data['id'] == "mock_upsert_id_123" # From mock_perform_upsert
            assert data['status'] == 'success'
            mock_perform_upsert.assert_awaited_once_with(
                content=upsert_payload["content"],
                memory_id=None, # ID was not provided in payload
                metadata=upsert_payload["metadata"]
            )
            print("Test 3 PASSED ✅")
        except Exception as e:
            print(f"Test 3 FAILED ❌: {e}")
        mock_perform_upsert.reset_mock()

        # --- Test 4: Delete Endpoint ---
        print("\n[Test 4] DELETE /memory/{memory_id}...")
        delete_id = "item_to_delete_abc"
        try:
            response = await client.delete(f"/memory/{delete_id}")
            response.raise_for_status()
            data = response.json()
            print(f"Response Status: {response.status_code}")
            print(f"Response Body: {data}")
            assert response.status_code == 200
            assert data['id'] == delete_id
            assert data['status'] == 'deleted'
            mock_perform_delete.assert_awaited_once_with(delete_id)
            print("Test 4 PASSED ✅")
        except Exception as e:
            print(f"Test 4 FAILED ❌: {e}")
        mock_perform_delete.reset_mock()

        # --- Test 5: Delete Endpoint (Simulate Failure) ---
        print("\n[Test 5] DELETE /memory/{memory_id} (Simulate Service Failure)...")
        mock_perform_delete.return_value = False # Configure mock to return False
        delete_id_fail = "item_failed_delete"
        try:
            response = await client.delete(f"/memory/{delete_id_fail}")
            # Expecting an HTTP error now
            print(f"Response Status: {response.status_code}")
            print(f"Response Body: {response.text}")
            assert response.status_code == 500 # Or 404 depending on desired logic
            mock_perform_delete.assert_awaited_once_with(delete_id_fail)
            print("Test 5 PASSED ✅ (Correctly handled service failure)")
        except httpx.HTTPStatusError as e:
             print(f"Response Status: {e.response.status_code}")
             print(f"Response Body: {e.response.text}")
             assert e.response.status_code == 500 # Check the specific error code
             mock_perform_delete.assert_awaited_once_with(delete_id_fail)
             print("Test 5 PASSED ✅ (Correctly handled service failure with HTTPStatusError)")
        except Exception as e:
            print(f"Test 5 FAILED ❌: Unexpected error {e}")
        mock_perform_delete.reset_mock()
        mock_perform_delete.return_value = True # Reset mock behavior

    # Restore original methods after tests
    memory_service_instance.perform_query = original_methods['perform_query']
    memory_service_instance.perform_upsert = original_methods['perform_upsert']
    memory_service_instance.perform_delete = original_methods['perform_delete']
    memory_service_instance.check_health = original_methods['check_health']
    memory_service_instance._initialized = False # Reset initialized state
    print("\nRestored original MemoryService methods.")

    print("\n--- API Endpoint Test Harness Finished ---")

# --- Run the Test Harness ---
if __name__ == "__main__":
    print("Running API Endpoint Test Harness...")
    print("Ensure the FastAPI server is running separately on port 8000.")
    asyncio.run(run_api_tests())