import logging
import asyncio
from neo4j import AsyncGraphDatabase, exceptions as neo4j_exceptions, AsyncDriver, AsyncSession, Result # Import async components
from typing import List, Dict, Any, Optional

# Import settings from the config module
try:
    from ..config import settings
except ImportError:
    print("Error: Could not import settings from app.config. Ensure the file exists and is configured.")
    # Fallback or raise error - Raising for clarity during development
    raise

# Import embedding service (needed if we decide to query by embedding later)
# from .embedding_service import get_embedding, batch_get_embeddings

logger = logging.getLogger(__name__)

# Define the node label consistent with Nova_AI.py usage
NEO4J_NODE_LABEL = "base" # Using lowercase as seen in Nova_AI.py graph_task

class GraphClient:
    """
    Manages interactions with the Neo4j graph database.
    Handles initialization, connection, and CRUD operations for memory nodes.
    Uses the neo4j async driver.
    """
    def __init__(self):
        """Initializes the GraphClient, deferring connection."""
        self.driver: Optional[AsyncDriver] = None
        self._DATABASE = settings.NEO4J_DATABASE
        logger.info("GraphClient initialized. Connection deferred.")

    async def initialize(self) -> bool:
        """
        Initializes the asynchronous connection to the Neo4j database.

        Returns:
            True if initialization is successful, False otherwise.
        """
        if self.driver:
            logger.info("Neo4j driver already initialized.")
            return True

        uri = settings.NEO4J_URI
        user = settings.NEO4J_USER
        password = settings.NEO4J_PASSWORD

        if not password:
             logger.error("❌ NEO4J_PASSWORD is not set. Cannot connect to Neo4j.")
             return False

        try:
            logger.info(f"Initializing Neo4j async driver for URI: {uri}, User: {user}...")
            # Use AsyncGraphDatabase for the async driver
            self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
            # Verify connectivity during initialization
            await self.check_connection()
            logger.info(f"Successfully connected to Neo4j database: {self._DATABASE} at {uri}")
            # Ensure constraints/indexes if needed (run once)
            await self._ensure_constraints()
            return True
        except neo4j_exceptions.AuthError as auth_err:
             logger.error(f"❌ Neo4j authentication failed for user '{user}': {auth_err}", exc_info=True)
             self.driver = None
             return False
        except neo4j_exceptions.ServiceUnavailable as su_err:
             logger.error(f"❌ Neo4j service unavailable at {uri}: {su_err}", exc_info=True)
             self.driver = None
             return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j async driver: {e}", exc_info=True)
            self.driver = None
            return False

    async def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            logger.info("Closing Neo4j driver connection...")
            await self.driver.close()
            self.driver = None
            logger.info("Neo4j driver connection closed.")

    async def _ensure_constraints(self):
        """Ensure necessary constraints/indexes exist (e.g., unique ID)."""
        if not self.driver:
            logger.error("Cannot ensure constraints: Neo4j driver not initialized.")
            return
        # Using 'entity_id' as the property name for uniqueness, matching Nova_AI.py's graph ingestion logic
        constraint_query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{NEO4J_NODE_LABEL}) REQUIRE n.entity_id IS UNIQUE"
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                logger.info(f"Ensuring unique constraint on :{NEO4J_NODE_LABEL}(entity_id)...")
                await session.run(constraint_query)
                logger.info(f"Constraint on :{NEO4J_NODE_LABEL}(entity_id) ensured.")
        except Exception as e:
            logger.error(f"❌ Failed to ensure Neo4j constraint: {e}", exc_info=True)
            # Continue execution, but log the error

    async def check_connection(self) -> bool:
        """
        Checks the connection to the Neo4j database by running a simple query.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        if not self.driver:
            logger.warning("Cannot check connection: Neo4j driver not initialized.")
            return False
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result: Result = await session.run("RETURN 1")
                summary = await result.consume() # Consume the result to check for errors
                logger.info(f"Neo4j connection check successful. Query counters: {summary.counters}")
                return True
        except Exception as e:
            logger.error(f"❌ Neo4j connection check failed: {e}", exc_info=True)
            return False

    async def upsert_graph_data(self, node_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Upserts a single node representing a memory item into Neo4j.
        Creates a node with the label ':base' (matching Nova_AI.py) and sets properties.

        Args:
            node_id: The unique ID for the node (should match Pinecone ID, e.g., MD5 hash).
            content: The text content of the memory item.
            metadata: Optional dictionary of metadata to store as properties.

        Returns:
            True if upsert is successful, False otherwise.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot upsert graph data.")
            return False

        # Prepare properties, ensuring metadata is handled safely
        properties = {
            "text": content,
            # Store metadata fields directly if possible, avoid nesting complex objects if not needed
            **(metadata or {}) # Unpack metadata dict into properties
        }
        # Ensure required 'entity_id' property is set for the constraint
        properties['entity_id'] = node_id

        # Cypher query to MERGE the node based on 'entity_id' and set/update properties
        # Using SET n += $properties handles both creation and update cleanly.
        # Adding the specific label :base dynamically.
        cypher = f"""
        MERGE (n:{NEO4J_NODE_LABEL} {{entity_id: $node_id}})
        SET n += $props
        RETURN n.entity_id AS id
        """
        params = {"node_id": node_id, "props": properties}

        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result = await session.run(cypher, params)
                record = await result.single()
                summary = await result.consume() # Important to consume results

                if record and record["id"] == node_id:
                     # Check summary counters to see if properties were set or node was created
                     if summary.counters.properties_set > 0 or summary.counters.nodes_created > 0:
                          logger.info(f"Successfully upserted graph node ID: {node_id} (Nodes created: {summary.counters.nodes_created}, Properties set: {summary.counters.properties_set})")
                          return True
                     else:
                          # Node existed but no properties were updated (unlikely with SET n += $props unless props were identical)
                          logger.warning(f"Graph node ID {node_id} merged but no properties were updated.")
                          return True # Still considered success as the node exists with the ID
                else:
                     # This case should ideally not happen with MERGE if the query ran successfully
                     logger.error(f"Failed to verify upsert for graph node ID: {node_id}. Result record: {record}")
                     return False

        except neo4j_exceptions.ConstraintError as ce:
             logger.error(f"❌ Constraint error during graph upsert for ID {node_id} (potential duplicate?): {ce}", exc_info=True)
             return False # Indicates an issue with the unique ID constraint logic
        except Exception as e:
            logger.error(f"❌ Failed to upsert graph data for ID {node_id}: {e}", exc_info=True)
            return False

    async def query_graph(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries the Neo4j graph for relevant memory nodes.
        Currently retrieves recent ':base' nodes as a simple strategy,
        mirroring the broad retrieval approach potentially used in Nova_AI.py
        before RRF/reranking.

        Args:
            query_text: The user's query text (currently unused in this simple strategy).
            top_k: The maximum number of nodes to return.

        Returns:
            A list of dictionaries representing graph nodes, or an empty list on error.
            Each dict should ideally contain 'id', 'text', 'metadata', and a placeholder 'score'.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot query graph.")
            return []

        # Simple Strategy: Retrieve top_k most recently added/updated nodes?
        # Requires a timestamp property. Let's assume we retrieve nodes and rely on RRF/reranker.
        # We retrieve nodes with the label ':base'.
        # TODO: Implement a more sophisticated query (e.g., full-text search, vector similarity) if needed.
        cypher = f"""
        MATCH (n:{NEO4J_NODE_LABEL})
        RETURN n.entity_id AS id, n.text AS text, n AS node_properties
        LIMIT $limit
        """
        params = {"limit": top_k}

        results = []
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result_cursor: Result = await session.run(cypher, params)
                records = await result_cursor.data() # Fetch all records

                for record in records:
                    node_props = dict(record.get("node_properties", {})) # Get all properties
                    # Construct the result dictionary expected by the merger
                    results.append({
                        "id": record.get("id"),
                        "text": record.get("text"),
                        "source": "graph",
                        "score": 0.0, # Placeholder score - RRF uses rank, not score here
                        "metadata": {k: v for k, v in node_props.items() if k not in ['entity_id', 'text']} # Store other props in metadata
                    })
            logger.info(f"Neo4j graph query returned {len(results)} nodes (limit {top_k}).")
            return results
        except Exception as e:
            logger.error(f"❌ Failed to query graph: {e}", exc_info=True)
            return []

    async def delete_graph_data(self, node_id: str) -> bool:
        """
        Deletes a node from the Neo4j graph by its unique ID ('entity_id').

        Args:
            node_id: The unique ID of the node to delete.

        Returns:
            True if deletion is successful or if the node didn't exist, False on error.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot delete graph data.")
            return False

        # Cypher query to match the node by 'entity_id' and detach delete it
        cypher = f"""
        MATCH (n:{NEO4J_NODE_LABEL} {{entity_id: $node_id}})
        DETACH DELETE n
        """
        params = {"node_id": node_id}

        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result = await session.run(cypher, params)
                summary = await result.consume() # Consume to get summary
                nodes_deleted = summary.counters.nodes_deleted
                logger.info(f"Attempted deletion for graph node ID {node_id}. Nodes deleted: {nodes_deleted}")
                # Consider successful if no error, regardless of whether node existed
                return True
        except Exception as e:
            logger.error(f"❌ Failed to delete graph data for ID {node_id}: {e}", exc_info=True)
            return False

# Example usage (optional, for testing)
async def _test_graph_client():
    print("Testing Graph Client...")
    # Requires Neo4j connection details in settings
    if not settings.NEO4J_PASSWORD:
        print("Skipping Graph client test: NEO4J_PASSWORD not set.")
        return

    client = GraphClient()
    initialized = await client.initialize()

    if not initialized:
        print("❌ Neo4j initialization failed. Aborting test.")
        return

    # Test Upsert
    test_id = "test_graph_node_456"
    test_content = "Neo4j is a graph database."
    test_metadata = {"category": "database", "type": "graph"}
    print(f"\nAttempting upsert for ID: {test_id}")
    upsert_ok = await client.upsert_graph_data(test_id, test_content, test_metadata)
    print(f"Upsert successful: {upsert_ok}")
    assert upsert_ok

    # Test Query
    print(f"\nAttempting query (simple retrieve)...")
    matches = await client.query_graph(query_text="dummy", top_k=5) # Query text unused in simple strategy
    print(f"Query returned {len(matches)} matches.")
    found = False
    if matches:
        print("Sample matches:")
        for match in matches[:2]:
             print(match)
             if match.get("id") == test_id:
                  found = True
                  assert match.get("text") == test_content
                  assert match.get("metadata", {}).get("category") == "database"
    assert found # Check if the inserted node was retrieved

    # Test Delete
    print(f"\nAttempting delete for ID: {test_id}")
    delete_ok = await client.delete_graph_data(test_id)
    print(f"Delete successful: {delete_ok}")
    assert delete_ok

    # Test Query After Delete
    print(f"\nAttempting query after delete...")
    matches_after_delete = await client.query_graph(query_text="dummy", top_k=5)
    print(f"Query after delete returned {len(matches_after_delete)} matches.")
    found_after_delete = any(match.get("id") == test_id for match in matches_after_delete)
    assert not found_after_delete

    # Test Health Check
    print(f"\nAttempting health check...")
    health_ok = await client.check_connection()
    print(f"Health check successful: {health_ok}")
    assert health_ok

    # Close connection
    await client.close()
    print("\nGraph Client Test Complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Requires environment variables for Neo4j connection
    asyncio.run(_test_graph_client())