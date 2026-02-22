import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass # Import dataclass
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP, Context
# ToolParam is not needed; FastMCP uses type hints directly

# Assuming your MemoryService and config are structured appropriately
# Adjust imports based on your actual project structure
from app.services.memory_service import MemoryService
from app.config import settings # To load necessary env vars for MemoryService init

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("nova-memory-mcp-server")

# Global instance (or manage via lifespan context)
# Using a global instance might be simpler if lifespan context proves tricky
# memory_service_instance: Optional[MemoryService] = None

# Define a context structure for type hinting if needed
@dataclass # Add dataclass decorator
class NovaMemoryContext:
    memory_service: MemoryService

@asynccontextmanager
async def service_lifespan(server: FastMCP) -> AsyncIterator[NovaMemoryContext]:
    """Manage MemoryService lifecycle."""
    logger.info("Initializing MemoryService...")
    # Ensure settings are loaded (Pydantic settings usually load on import or first use)
    # If MemoryService takes settings directly, pass them:
    # memory_service = MemoryService(settings=settings) 
    memory_service = MemoryService() 
    
    initialized = await memory_service.initialize()
    if not initialized:
        logger.error("MemoryService failed to initialize!")
        # How to handle fatal init error in lifespan? 
        # Maybe raise an exception to stop the server?
        raise RuntimeError("MemoryService initialization failed")
    else:
        logger.info("MemoryService initialized successfully.")
        try:
            yield NovaMemoryContext(memory_service=memory_service)
        finally:
            logger.info("Shutting down MemoryService (if applicable)...")
            # Add cleanup logic here if MemoryService needs it
            # await memory_service.shutdown() 
            logger.info("MemoryService shutdown complete.")

# Create the FastMCP server instance
# Pass dependencies needed for installation via `mcp install`
# Ensure all packages from requirements.txt needed at runtime are listed
# (mcp itself is implicitly included)
mcp = FastMCP(
    "nova-memory", 
    lifespan=service_lifespan,
    # Add key runtime dependencies here if needed for `mcp install` packaging
    # dependencies=["fastapi", "uvicorn", "neo4j", "openai", "pinecone", ...] 
    # Alternatively, rely on requirements.txt being installed in the environment
)

# --- Tool Definitions ---

@mcp.tool()
async def query_memory(ctx: Context, query: str) -> Dict[str, Any]:
    """
    Retrieves relevant memory items based on a query text.
    Uses vector search (Pinecone) and graph search (Neo4j), fuses results, and reranks.
    """
    logger.info(f"Tool 'query_memory' called with query: '{query}'")
    memory_service = ctx.request_context.lifespan_context.memory_service
    try:
        # Assuming perform_query returns a list of dicts suitable for the API
        results = await memory_service.perform_query(query)
        logger.info(f"Query returned {len(results)} results.")
        # FastMCP automatically serializes the return value (dict, list, primitives) to JSON
        return {"results": results}
    except Exception as e:
        logger.error(f"Error during query_memory: {e}", exc_info=True)
        # Return an error structure that MCP client can understand
        return {"error": f"Failed to execute query: {str(e)}"}

@mcp.tool()
async def upsert_memory(ctx: Context, content: str, id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Adds or updates a memory item. Generates embeddings and stores in Pinecone and Neo4j.
    If 'id' is not provided, MemoryService generates a deterministic ID.
    """
    logger.info(f"Tool 'upsert_memory' called. ID: {id}, Content: '{content[:50]}...'")
    memory_service = ctx.request_context.lifespan_context.memory_service
    if metadata is None:
        metadata = {} # Ensure metadata is a dict

    try:
        item_id = await memory_service.perform_upsert(
            content=content,
            memory_id=id,
            metadata=metadata
        )
        if item_id:
            logger.info(f"Upsert successful for ID: {item_id}")
            return {"id": item_id, "status": "success"}
        logger.warning("Upsert failed: MemoryService returned no item ID.")
        return {"error": "Failed to upsert memory item."}
    except Exception as e:
        logger.error(f"Error during upsert_memory: {e}", exc_info=True)
        return {"error": f"Failed to upsert memory: {str(e)}"}

@mcp.tool()
async def delete_memory(ctx: Context, memory_id: str) -> Dict[str, Any]:
    """
    Deletes a memory item by its ID from both Pinecone and Neo4j.
    """
    logger.info(f"Tool 'delete_memory' called for ID: {memory_id}")
    memory_service = ctx.request_context.lifespan_context.memory_service
    try:
        success = await memory_service.perform_delete(memory_id)
        if success:
             logger.info(f"Delete successful for ID: {memory_id}")
             return {"id": memory_id, "status": "deleted"}
        else:
             logger.warning(f"Delete operation returned false for ID: {memory_id}")
             # Adjust based on expected MemoryService behavior for not found etc.
             return {"error": f"Memory item with ID '{memory_id}' not found or delete failed."}
    except Exception as e:
        logger.error(f"Error during delete_memory: {e}", exc_info=True)
        return {"error": f"Failed to delete memory ID '{memory_id}': {str(e)}"}

@mcp.tool()
async def check_health(ctx: Context) -> Dict[str, Any]:
    """
    Checks the health of the memory service and its dependencies (Pinecone, Neo4j).
    """
    logger.info("Tool 'check_health' called.")
    memory_service = ctx.request_context.lifespan_context.memory_service
    try:
        # Assuming check_health returns a dict with status details
        health_status = await memory_service.check_health()
        logger.info(f"Health check status: {health_status}")
        return health_status
    except Exception as e:
        logger.error(f"Error during check_health: {e}", exc_info=True)
        return {"status": "error", "details": f"Health check failed: {str(e)}"}


if __name__ == "__main__":
    # This allows running the server directly using `python mcp_server.py`
    # It will use the stdio transport by default.
    logger.info("Starting Nova Memory MCP Server directly...")
    mcp.run()
