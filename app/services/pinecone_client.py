import logging
import pinecone
from pinecone import Index, PineconeException # Use specific exception
from typing import List, Dict, Any, Optional

# Import settings from the config module
try:
    from ..config import settings
except ImportError:
    print("Error: Could not import settings from app.config. Ensure the file exists and is configured.")
    # Fallback or raise error - Raising for clarity during development
    raise

logger = logging.getLogger(__name__)

class PineconeClient:
    """
    Manages interactions with the Pinecone vector database.
    Handles initialization, connection, and CRUD operations.
    """
    def __init__(self):
        """Initializes the PineconeClient, deferring connection."""
        self.index_name: str = settings.PINECONE_INDEX
        self.index: Optional[Index] = None
        self.dimension: int = 1536 # Dimension for text-embedding-ada-002
        self.metric: str = "cosine" # Similarity metric
        logger.info(f"PineconeClient initialized for index '{self.index_name}'. Connection deferred.")

    def initialize(self) -> bool:
        """
        Initializes the connection to Pinecone and ensures the index exists.

        Returns:
            True if initialization is successful, False otherwise.
        """
        if self.index:
            logger.info(f"Pinecone index '{self.index_name}' already initialized.")
            return True

        try:
            logger.info(f"Initializing Pinecone connection (API Key: {'*' * (len(settings.PINECONE_API_KEY) - 4) + settings.PINECONE_API_KEY[-4:]}, Env: {settings.PINECONE_ENV})...")
            # Use the new Pinecone client initialization
            pc = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV)

            # Check if index exists
            # Get index names from the IndexList object
            index_list = pc.list_indexes()
            existing_index_names = [index.name for index in index_list.indexes] if index_list and index_list.indexes else []
            if self.index_name not in existing_index_names:
                logger.warning(f"Pinecone index '{self.index_name}' not found. Attempting to create...")
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric
                    # Add other configurations like pod_type if needed, e.g., pod_type="p1.x1"
                )
                logger.info(f"Successfully created Pinecone index '{self.index_name}'.")
            else:
                logger.info(f"Pinecone index '{self.index_name}' already exists.")

            # Get the index object
            self.index = pc.Index(self.index_name)
            logger.info(f"Successfully connected to Pinecone index '{self.index_name}'.")
            return True

        except PineconeException as pe:
            logger.error(f"❌ Pinecone API error during initialization: {pe}", exc_info=True)
            self.index = None
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pinecone client for index '{self.index_name}': {e}", exc_info=True)
            self.index = None
            return False

    def upsert_vector(self, vector_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Upserts (inserts or updates) a single vector into the Pinecone index.

        Args:
            vector_id: The unique ID for the vector.
            vector: The embedding vector (list of floats).
            metadata: A dictionary containing metadata (must include 'text').

        Returns:
            True if upsert is successful, False otherwise.
        """
        if not self.index:
            logger.error("Pinecone index not initialized. Cannot upsert vector.")
            return False
        if 'text' not in metadata:
             logger.warning(f"Upsert metadata for ID '{vector_id}' is missing the required 'text' field. Proceeding, but retrieval might be affected.")
             # Consider raising an error or adding a default text if this is critical

        try:
            upsert_response = self.index.upsert(vectors=[{"id": vector_id, "values": vector, "metadata": metadata}])
            logger.debug(f"Upsert response for ID {vector_id}: {upsert_response}")
            if upsert_response.upserted_count == 1:
                 logger.info(f"Successfully upserted vector ID: {vector_id}")
                 return True
            else:
                 logger.warning(f"Pinecone upsert for ID {vector_id} reported {upsert_response.upserted_count} upserted vectors (expected 1).")
                 # Treat as success for now, but might indicate an issue.
                 return True
        except PineconeException as pe:
            logger.error(f"❌ Pinecone API error during upsert for ID {vector_id}: {pe}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"❌ Failed to upsert vector ID {vector_id}: {e}", exc_info=True)
            return False

    def query_vector(self, query_vector: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Queries the Pinecone index for similar vectors.

        Args:
            query_vector: The embedding vector of the query.
            top_k: The number of top results to retrieve.
            filter: Optional dictionary for metadata filtering (e.g., {"category": "conversation"}).

        Returns:
            A list of matching results (dictionaries with id, score, metadata), or an empty list on error.
        """
        if not self.index:
            logger.error("Pinecone index not initialized. Cannot query vectors.")
            return []

        try:
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False # Usually not needed for results
            }
            if filter:
                query_params["filter"] = filter

            logger.debug(f"Querying Pinecone with top_k={top_k}, filter={filter}")
            results = self.index.query(**query_params)
            logger.info(f"Pinecone query returned {len(results.get('matches', []))} matches.")
            return results.get("matches", [])

        except PineconeException as pe:
            logger.error(f"❌ Pinecone API error during query: {pe}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"❌ Failed to query vectors: {e}", exc_info=True)
            return []

    def delete_vector(self, vector_id: str) -> bool:
        """
        Deletes a vector from the Pinecone index by its ID.

        Args:
            vector_id: The unique ID of the vector to delete.

        Returns:
            True if deletion is successful or if the ID didn't exist, False on error.
        """
        if not self.index:
            logger.error("Pinecone index not initialized. Cannot delete vector.")
            return False

        try:
            delete_response = self.index.delete(ids=[vector_id])
            logger.info(f"Attempted deletion for vector ID {vector_id}. Response: {delete_response}")
            # Pinecone delete returns {} on success, even if ID didn't exist.
            # We consider it successful unless an exception occurs.
            return True
        except PineconeException as pe:
            logger.error(f"❌ Pinecone API error during delete for ID {vector_id}: {pe}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"❌ Failed to delete vector ID {vector_id}: {e}", exc_info=True)
            return False

    def check_connection(self) -> bool:
        """
        Checks the connection to the Pinecone index by fetching stats.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        if not self.index:
            logger.warning("Cannot check connection: Pinecone index not initialized.")
            return False
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Pinecone connection check successful. Stats: {stats}")
            return True
        except PineconeException as pe:
            logger.error(f"❌ Pinecone API error during connection check: {pe}")
            return False
        except Exception as e:
            logger.error(f"❌ Pinecone connection check failed: {e}")
            return False

# Example usage (optional, for testing)
async def _test_pinecone_client():
    print("Testing Pinecone Client...")
    # Requires PINECONE_API_KEY, PINECONE_ENV to be set
    if not settings.PINECONE_API_KEY or not settings.PINECONE_ENV:
        print("Skipping Pinecone client test: API Key or Environment not set.")
        return

    client = PineconeClient()
    initialized = client.initialize()

    if not initialized:
        print("❌ Pinecone initialization failed. Aborting test.")
        return

    print(f"Connected to index: {client.index_name}")

    # Test Upsert
    test_id = "test_vector_123"
    test_vector = [0.1] * 1536 # Dummy vector
    test_metadata = {"text": "This is a test vector.", "category": "test"}
    print(f"\nAttempting upsert for ID: {test_id}")
    upsert_ok = client.upsert_vector(test_id, test_vector, test_metadata)
    print(f"Upsert successful: {upsert_ok}")

    # Test Query (allow time for upsert to index)
    import time
    time.sleep(2) # Pinecone indexing can take a moment
    print(f"\nAttempting query similar to test vector...")
    query_vec = [0.11] * 1536 # Slightly different vector
    matches = client.query_vector(query_vec, top_k=1, filter={"category": "test"})
    print(f"Query returned {len(matches)} matches.")
    if matches:
        print(f"Top match: {matches[0]}")
        assert matches[0]['id'] == test_id

    # Test Delete
    print(f"\nAttempting delete for ID: {test_id}")
    delete_ok = client.delete_vector(test_id)
    print(f"Delete successful: {delete_ok}")

    # Test Query After Delete (allow time for delete)
    time.sleep(2)
    print(f"\nAttempting query after delete...")
    matches_after_delete = client.query_vector(query_vec, top_k=1, filter={"category": "test"})
    print(f"Query after delete returned {len(matches_after_delete)} matches.")
    assert len(matches_after_delete) == 0

    # Test Health Check
    print(f"\nAttempting health check...")
    health_ok = client.check_connection()
    print(f"Health check successful: {health_ok}")

    print("\nPinecone Client Test Complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Requires environment variables to be set
    import asyncio
    asyncio.run(_test_pinecone_client())