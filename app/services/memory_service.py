import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional

# Import dependent services and modules
try:
    from ..config import settings
    from .embedding_service import get_embedding, batch_get_embeddings
    from .pinecone_client import PineconeClient
    from .graph_client import GraphClient
    # Import reused Nova modules (assuming they are in the same directory)
    from .query_router import QueryRouter, RoutingMode
    from .hybrid_merger import HybridMerger
    from .reranker import CrossEncoderReranker
except ImportError as e:
    print(f"Error importing modules in memory_service.py: {e}. Ensure all service files and Nova modules exist.")
    # Depending on severity, might raise error or proceed with caution
    raise

logger = logging.getLogger(__name__)

class MemoryService:
    """
    Orchestrates memory operations by integrating embedding, vector store (Pinecone),
    graph store (Neo4j), query routing, merging, and reranking components.
    """
    def __init__(self):
        """Initializes the MemoryService and its components."""
        logger.info("Initializing MemoryService...")
        # Initialize clients and Nova modules
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.pinecone_client = PineconeClient()
        self.graph_client = GraphClient()
        self.query_router = QueryRouter()
        self.hybrid_merger = HybridMerger() # Uses default RRF k=60
        self.reranker: Optional[CrossEncoderReranker] = None # Initialize as None, load in async init

        # Flag to track initialization status
        self._initialized = False
        self._reranker_loaded = False
        logger.info("MemoryService components instantiated.")

    async def initialize(self):
        """
        Asynchronously initializes backend clients (Pinecone, Neo4j) and loads
        the reranker model. Should be called before using the service.
        """
        if self._initialized:
            logger.info("MemoryService already initialized.")
            return

        logger.info("Starting asynchronous initialization of MemoryService...")
        init_tasks = {
            "pinecone": self.pinecone_client.initialize(),
            "graph": self.graph_client.initialize(),
            "reranker": self._load_reranker_model() # Separate method for reranker loading
        }

        results = await asyncio.gather(*init_tasks.values(), return_exceptions=True)

        # Check results
        pinecone_ok = isinstance(results[0], bool) and results[0]
        graph_ok = isinstance(results[1], bool) and results[1]
        reranker_ok = isinstance(results[2], bool) and results[2]

        if not pinecone_ok:
            logger.error("❌ Pinecone client failed to initialize.")
        if not graph_ok:
            logger.error("❌ Graph client failed to initialize.")
        if not reranker_ok:
            logger.warning("⚠️ Reranker model failed to load. Reranking will be disabled.")
            self.reranker = None # Ensure reranker is None if loading failed
        else:
             self._reranker_loaded = True

        # Service is considered initialized even if some components failed,
        # but operations requiring failed components will not work.
        self._initialized = True
        logger.info(f"MemoryService initialization complete. Status - Pinecone: {'OK' if pinecone_ok else 'Failed'}, Graph: {'OK' if graph_ok else 'Failed'}, Reranker: {'Loaded' if self._reranker_loaded else 'Failed/Disabled'}")

    async def _load_reranker_model(self) -> bool:
        """Loads the reranker model asynchronously."""
        try:
            model_name = settings.RERANKER_MODEL_NAME
            if model_name:
                self.reranker = CrossEncoderReranker(model_name=model_name)
                loaded = await self.reranker.load_model()
                return loaded
            else:
                logger.warning("No RERANKER_MODEL_NAME configured. Reranker disabled.")
                return False # Consider False as "not loaded"
        except Exception as e:
            logger.error(f"❌ Exception during reranker initialization/loading: {e}", exc_info=True)
            self.reranker = None
            return False

    async def close(self):
        """Closes connections (e.g., Neo4j driver)."""
        logger.info("Closing MemoryService resources...")
        await self.graph_client.close()
        # Pinecone client might not need explicit closing depending on version
        logger.info("MemoryService resources closed.")

    async def perform_query(self, query_text: str, top_k_vector: int = 50, top_k_final: int = 15) -> List[Dict[str, Any]]:
        """
        Performs a fused memory query using the full pipeline.

        Args:
            query_text: The user's query.
            top_k_vector: Number of initial results to fetch from Pinecone.
            top_k_final: Number of final results to return after reranking.

        Returns:
            A list of relevant memory items, sorted by relevance.
        """
        if not self._initialized:
            logger.error("MemoryService not initialized. Cannot perform query.")
            # raise ServiceUnavailableError("MemoryService is not ready.") # Or return empty list
            return []
        if not query_text:
            logger.warning("Received empty query text.")
            return []

        logger.info(f"Performing fused query for: '{query_text[:100]}...'")

        # 1. Query Routing (for logging/potential future use)
        routing_mode = self.query_router.route(query_text)
        logger.info(f"Query classified as: {routing_mode.name}")

        # 2. Get Query Embedding (Run sync function in thread)
        try:
            query_embedding = await asyncio.to_thread(get_embedding, query_text, self.embedding_model_name)
            if not any(query_embedding): # Check if it's a zero vector (error indicator)
                 logger.error("Failed to get valid query embedding. Aborting query.")
                 return []
        except Exception as e:
            logger.error(f"❌ Error getting query embedding: {e}", exc_info=True)
            return []

        # 3. Parallel Retrieval (Vector + Graph)
        vector_results = []
        graph_results = []
        try:
            # We always retrieve from both as per plan, regardless of routing_mode
            logger.debug(f"Initiating parallel retrieval (Vector k={top_k_vector}, Graph k={top_k_vector})...") # Graph k is indicative
            results = await asyncio.gather(
                self.pinecone_client.query_vector(query_embedding, top_k=top_k_vector),
                self.graph_client.query_graph(query_text, top_k=top_k_vector), # Pass query_text for potential graph search strategies
                return_exceptions=True
            )

            # Process results, handling potential errors
            if isinstance(results[0], list):
                vector_results = results[0]
                logger.info(f"Vector retrieval returned {len(vector_results)} results.")
            elif isinstance(results[0], Exception):
                logger.error(f"❌ Vector retrieval failed: {results[0]}", exc_info=results[0])

            if isinstance(results[1], list):
                graph_results = results[1]
                logger.info(f"Graph retrieval returned {len(graph_results)} results.")
            elif isinstance(results[1], Exception):
                logger.error(f"❌ Graph retrieval failed: {results[1]}", exc_info=results[1])

        except Exception as e:
            logger.error(f"❌ Error during parallel retrieval: {e}", exc_info=True)
            # Continue with potentially partial results if possible

        # 4. Hybrid Merging (RRF)
        if not vector_results and not graph_results:
             logger.warning("No results from either vector or graph store.")
             return []

        try:
            logger.debug(f"Merging {len(vector_results)} vector and {len(graph_results)} graph results...")
            # Run sync merge function in thread
            fused_results = await asyncio.to_thread(
                self.hybrid_merger.merge_results, vector_results, graph_results
            )
            logger.info(f"Hybrid merging complete. {len(fused_results)} unique results after RRF.")
        except Exception as e:
            logger.error(f"❌ Error during hybrid merging: {e}", exc_info=True)
            # Fallback: maybe just return vector results? Or empty? Returning empty for now.
            return []

        # 5. Reranking
        if self.reranker and self._reranker_loaded and fused_results:
            try:
                logger.debug(f"Reranking {len(fused_results)} fused results...")
                # Reranker method is already async
                final_results = await self.reranker.rerank(query_text, fused_results, top_n=top_k_final)
                logger.info(f"Reranking complete. Returning top {len(final_results)} results.")
            except Exception as e:
                logger.error(f"❌ Error during reranking: {e}. Returning fused results without reranking.", exc_info=True)
                # Fallback to fused results if reranking fails
                final_results = fused_results[:top_k_final]
        else:
            logger.info("Reranker disabled or no results to rerank. Returning top fused results.")
            final_results = fused_results[:top_k_final] # Return top N fused results

        return final_results

    async def perform_upsert(self, content: str, memory_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Upserts memory content into both Pinecone and Neo4j.

        Args:
            content: The text content to store.
            memory_id: Optional specific ID. If None, an MD5 hash of content is used.
            metadata: Optional dictionary of metadata.

        Returns:
            The ID of the upserted item, or None if upsert failed.
        """
        if not self._initialized:
            logger.error("MemoryService not initialized. Cannot perform upsert.")
            return None
        if not content:
            logger.warning("Received empty content for upsert.")
            return None

        # 1. Determine ID
        item_id = memory_id or hashlib.md5(content.encode()).hexdigest()
        logger.info(f"Performing upsert for ID: {item_id}, Content: '{content[:100]}...'")

        # 2. Get Embedding
        try:
            # Run sync function in thread
            embedding = await asyncio.to_thread(get_embedding, content, self.embedding_model_name)
            if not any(embedding): # Check for zero vector
                 logger.error(f"Failed to get valid embedding for upsert ID {item_id}. Aborting.")
                 return None
        except Exception as e:
            logger.error(f"❌ Error getting embedding for upsert ID {item_id}: {e}", exc_info=True)
            return None

        # 3. Prepare Metadata for Pinecone (ensure 'text' field exists)
        pinecone_meta = metadata.copy() if metadata else {}
        pinecone_meta['text'] = content # Crucial for retrieval pipeline

        # 4. Perform Upserts (Vector + Graph) - Attempt both, handle errors
        pinecone_success = False
        graph_success = False

        try:
            # Run sync function in thread
            pinecone_success = await asyncio.to_thread(
                self.pinecone_client.upsert_vector, item_id, embedding, pinecone_meta
            )
        except Exception as e:
            logger.error(f"❌ Error during Pinecone upsert thread execution for ID {item_id}: {e}", exc_info=True)
            pinecone_success = False # Ensure flag is False on exception

        try:
            # Graph upsert is already async
            graph_success = await self.graph_client.upsert_graph_data(item_id, content, metadata)
        except Exception as e:
            logger.error(f"❌ Error during Graph upsert execution for ID {item_id}: {e}", exc_info=True)
            graph_success = False # Ensure flag is False on exception

        # 5. Handle Results and Potential Rollback/Logging
        if pinecone_success and graph_success:
            logger.info(f"Successfully upserted ID {item_id} to both Pinecone and Graph.")
            return item_id
        elif pinecone_success and not graph_success:
            logger.error(f"❌ Upsert failed for ID {item_id}: Succeeded in Pinecone but failed in Graph.")
            # Optional: Attempt rollback from Pinecone
            logger.warning(f"Attempting rollback: Deleting ID {item_id} from Pinecone due to graph failure.")
            # Run sync function in thread
            rollback_ok = await asyncio.to_thread(self.pinecone_client.delete_vector, item_id)
            logger.warning(f"Pinecone rollback successful: {rollback_ok}")
            return None
        elif not pinecone_success and graph_success:
            logger.error(f"❌ Upsert failed for ID {item_id}: Succeeded in Graph but failed in Pinecone.")
            # Optional: Attempt rollback from Graph
            logger.warning(f"Attempting rollback: Deleting ID {item_id} from Graph due to Pinecone failure.")
            rollback_ok = await self.graph_client.delete_graph_data(item_id) # Already async
            logger.warning(f"Graph rollback successful: {rollback_ok}")
            return None
        else:
            logger.error(f"❌ Upsert failed for ID {item_id}: Failed in both Pinecone and Graph.")
            return None

    async def perform_delete(self, memory_id: str) -> bool:
        """
        Deletes a memory item from both Pinecone and Neo4j.

        Args:
            memory_id: The ID of the item to delete.

        Returns:
            True if deletion was successful in at least one store (or item didn't exist), False otherwise.
        """
        if not self._initialized:
            logger.error("MemoryService not initialized. Cannot perform delete.")
            return False
        if not memory_id:
            logger.warning("Received empty memory_id for delete.")
            return False

        logger.info(f"Performing delete for ID: {memory_id}")

        # Attempt deletions in parallel
        try:
            results = await asyncio.gather(
                asyncio.to_thread(self.pinecone_client.delete_vector, memory_id), # Run sync in thread
                self.graph_client.delete_graph_data(memory_id), # Already async
                return_exceptions=True
            )

            pinecone_success = isinstance(results[0], bool) and results[0]
            graph_success = isinstance(results[1], bool) and results[1]

            if isinstance(results[0], Exception):
                 logger.error(f"❌ Error during Pinecone delete thread execution for ID {memory_id}: {results[0]}", exc_info=results[0])
            if isinstance(results[1], Exception):
                 logger.error(f"❌ Error during Graph delete execution for ID {memory_id}: {results[1]}", exc_info=results[1])

            if pinecone_success or graph_success:
                 logger.info(f"Deletion attempt for ID {memory_id} complete. Pinecone success: {pinecone_success}, Graph success: {graph_success}")
                 # Consider successful if at least one deletion worked or if ID didn't exist in one/both
                 return True
            else:
                 logger.error(f"Deletion failed for ID {memory_id} in both stores.")
                 return False # Failed in both

        except Exception as e:
            logger.error(f"❌ Unexpected error during parallel delete for ID {memory_id}: {e}", exc_info=True)
            return False

    async def check_health(self) -> Dict[str, str]:
        """
        Checks the health of the service and its dependencies.

        Returns:
            A dictionary indicating the status of each component.
        """
        if not self._initialized:
            return {"status": "error", "detail": "MemoryService not initialized"}

        statuses = {"status": "ok"} # Assume ok initially

        # Check dependencies in parallel
        try:
            results = await asyncio.gather(
                asyncio.to_thread(self.pinecone_client.check_connection), # Run sync in thread
                self.graph_client.check_connection(), # Already async
                return_exceptions=True
            )

            # Pinecone status
            if isinstance(results[0], bool) and results[0]:
                statuses["pinecone"] = "ok"
            else:
                statuses["pinecone"] = f"error: {results[0]}" if isinstance(results[0], Exception) else "error: Failed check"
                statuses["status"] = "error" # Overall status degraded

            # Graph status
            if isinstance(results[1], bool) and results[1]:
                statuses["graph"] = "ok"
            else:
                statuses["graph"] = f"error: {results[1]}" if isinstance(results[1], Exception) else "error: Failed check"
                statuses["status"] = "error" # Overall status degraded

            # Reranker status (based on loading)
            statuses["reranker"] = "loaded" if self._reranker_loaded else "disabled/failed"
            if not self._reranker_loaded:
                 # Don't mark overall status as error just for reranker, but indicate issue
                 pass

        except Exception as e:
            logger.error(f"❌ Unexpected error during health check: {e}", exc_info=True)
            statuses["status"] = "error"
            statuses["detail"] = f"Unexpected error: {e}"

        return statuses