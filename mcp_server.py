import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass  # Import dataclass
from typing import Any, Dict, List, Optional

# Configure logging BEFORE any app imports to prevent stdout pollution
# Libraries like sentence_transformers/tqdm can interfere with MCP stdio transport
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
# App-level loggers at INFO, noisy libraries stay at WARNING
for _name in ("nova-memory-mcp-server", "app.services"):
    logging.getLogger(_name).setLevel(logging.INFO)

from mcp.server.fastmcp import FastMCP, Context
from app.services.memory_service import MemoryService

logger = logging.getLogger("nova-memory-mcp-server")

# Global instance (or manage via lifespan context)
# Using a global instance might be simpler if lifespan context proves tricky
# memory_service_instance: Optional[MemoryService] = None


# Define a context structure for type hinting if needed
@dataclass  # Add dataclass decorator
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

# --- Internal Helpers ---


def _require_memory_service(ctx: Context) -> MemoryService:
    """
    Returns the initialized MemoryService from FastMCP lifespan context.

    Raises:
        RuntimeError: If request/lifespan context is unavailable or uninitialized.
    """
    request_context = getattr(ctx, "request_context", None)
    if request_context is None:
        raise RuntimeError(
            "MCP request context is unavailable. Ensure the server is running with lifespan enabled."
        )

    lifespan_context = getattr(request_context, "lifespan_context", None)
    if lifespan_context is None:
        raise RuntimeError(
            "MCP lifespan context is unavailable. MemoryService has not been initialized."
        )

    memory_service = getattr(lifespan_context, "memory_service", None)
    if memory_service is None:
        raise RuntimeError(
            "MemoryService is unavailable in lifespan context. Check server startup logs."
        )

    return memory_service


def _resolve_result_score(result: Dict[str, Any]) -> Optional[float]:
    """Extracts the best available score field from a memory result."""
    for key in ("rerank_score", "rrf_score", "fusion_score", "score"):
        raw = result.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _normalize_tags(value: Any) -> set[str]:
    """Normalizes metadata tags into a lowercase set for matching."""
    if value is None:
        return set()
    if isinstance(value, str):
        stripped = value.strip().lower()
        return {stripped} if stripped else set()
    if isinstance(value, list):
        tags = set()
        for item in value:
            if isinstance(item, str) and item.strip():
                tags.add(item.strip().lower())
        return tags
    return set()


def _filter_query_results(
    results: List[Dict[str, Any]],
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    min_score: Optional[float] = None,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Applies metadata and score filtering to fused memory query results."""
    normalized_category = (
        category.strip().lower()
        if isinstance(category, str) and category.strip()
        else None
    )
    normalized_run_id = (
        run_id.strip() if isinstance(run_id, str) and run_id.strip() else None
    )
    requested_tags = _normalize_tags(tags) if tags else set()

    filtered: List[Dict[str, Any]] = []
    for result in results:
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}

        if normalized_category:
            item_category = str(metadata.get("category", "")).strip().lower()
            if item_category != normalized_category:
                continue

        if normalized_run_id:
            item_run_id = str(metadata.get("run_id", "")).strip()
            if item_run_id != normalized_run_id:
                continue

        if requested_tags:
            item_tags = _normalize_tags(metadata.get("tags"))
            if not requested_tags.issubset(item_tags):
                continue

        if min_score is not None:
            score = _resolve_result_score(result)
            if score is None or score < min_score:
                continue

        filtered.append(result)

    return filtered


# --- Tool Definitions ---


@mcp.tool()
async def query_memory(
    ctx: Context,
    query: str,
    top_k_vector: int = 50,
    top_k_final: int = 15,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    min_score: Optional[float] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves relevant memory items based on a query text.
    Uses vector search (Pinecone) and graph search (Neo4j), fuses results, and reranks.
    Supports optional retrieval controls and metadata filtering.
    """
    logger.info(
        f"Tool 'query_memory' called with query='{query}', top_k_vector={top_k_vector}, "
        f"top_k_final={top_k_final}, category={category}, tags={tags}, min_score={min_score}, run_id={run_id}"
    )
    try:
        memory_service = _require_memory_service(ctx)
        if top_k_vector < 1 or top_k_final < 1:
            return {"error": "top_k_vector and top_k_final must be >= 1"}

        results = await memory_service.perform_query(
            query_text=query, top_k_vector=top_k_vector, top_k_final=top_k_final
        )
        filtered_results = _filter_query_results(
            results=results,
            category=category,
            tags=tags,
            min_score=min_score,
            run_id=run_id,
        )
        logger.info(
            f"Query returned {len(results)} results before filtering, {len(filtered_results)} after filtering."
        )
        # FastMCP automatically serializes the return value (dict, list, primitives) to JSON
        return {"results": filtered_results}
    except Exception as e:
        logger.error(f"Error during query_memory: {e}", exc_info=True)
        # Return an error structure that MCP client can understand
        return {"error": f"Failed to execute query: {str(e)}"}


@mcp.tool()
async def upsert_memory(
    ctx: Context,
    content: str,
    id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Adds or updates a memory item. Generates embeddings and stores in Pinecone and Neo4j.
    If 'id' is not provided, MemoryService generates a deterministic ID.
    """
    logger.info(f"Tool 'upsert_memory' called. ID: {id}, Content: '{content[:50]}...'")
    if metadata is None:
        metadata = {}  # Ensure metadata is a dict

    try:
        memory_service = _require_memory_service(ctx)
        item_id = await memory_service.perform_upsert(
            content=content, memory_id=id, metadata=metadata
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
async def bulk_upsert_memory(
    ctx: Context,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Adds or updates multiple memory items in one call.

    Expected item shape:
      {"content": str, "id": Optional[str], "metadata": Optional[dict]}
    """
    item_count = len(items) if isinstance(items, list) else 0
    logger.info(f"Tool 'bulk_upsert_memory' called with {item_count} items.")

    if not isinstance(items, list) or not items:
        return {"error": "items must be a non-empty list"}
    if len(items) > 500:
        return {"error": "items length must be <= 500"}

    try:
        memory_service = _require_memory_service(ctx)
        summary = await memory_service.perform_bulk_upsert(items=items)
        logger.info(
            "Bulk upsert summary: "
            f"status={summary.get('status')}, total={summary.get('total')}, "
            f"succeeded={summary.get('succeeded')}, failed={summary.get('failed')}"
        )
        return summary
    except Exception as e:
        logger.error(f"Error during bulk_upsert_memory: {e}", exc_info=True)
        return {"error": f"Failed to bulk upsert memories: {str(e)}"}


@mcp.tool()
async def delete_memory(ctx: Context, memory_id: str) -> Dict[str, Any]:
    """
    Deletes a memory item by its ID from both Pinecone and Neo4j.
    """
    logger.info(f"Tool 'delete_memory' called for ID: {memory_id}")
    try:
        memory_service = _require_memory_service(ctx)
        success = await memory_service.perform_delete(memory_id)
        if success:
            logger.info(f"Delete successful for ID: {memory_id}")
            return {"id": memory_id, "status": "deleted"}
        else:
            logger.warning(f"Delete operation returned false for ID: {memory_id}")
            # Adjust based on expected MemoryService behavior for not found etc.
            return {
                "error": f"Memory item with ID '{memory_id}' not found or delete failed."
            }
    except Exception as e:
        logger.error(f"Error during delete_memory: {e}", exc_info=True)
        return {"error": f"Failed to delete memory ID '{memory_id}': {str(e)}"}


@mcp.tool()
async def create_checkpoint(
    ctx: Context,
    session_id: str,
    session_summary: str,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    open_threads: Optional[List[str]] = None,
    next_actions: Optional[List[str]] = None,
    project: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a session checkpoint — a structured summary of a completed session.
    System auto-injects event_time, event_seq, and snapshots last_event_seq.

    Use this at the end of a work session to record what was accomplished,
    what threads are still open, and what should happen next.
    """
    logger.info(f"Tool 'create_checkpoint' called for session_id='{session_id}'")
    try:
        memory_service = _require_memory_service(ctx)
        checkpoint_id = await memory_service.create_checkpoint(
            session_id=session_id,
            session_summary=session_summary,
            started_at=started_at,
            ended_at=ended_at,
            open_threads=open_threads,
            next_actions=next_actions,
            project=project,
            thread_id=thread_id,
        )
        if checkpoint_id:
            logger.info(f"Checkpoint created successfully: {checkpoint_id}")
            return {"id": checkpoint_id, "status": "success", "session_id": session_id}
        logger.warning("Checkpoint creation failed: no ID returned.")
        return {"error": "Failed to create checkpoint."}
    except Exception as e:
        logger.error(f"Error during create_checkpoint: {e}", exc_info=True)
        return {"error": f"Failed to create checkpoint: {str(e)}"}


@mcp.tool()
async def get_last_checkpoint(
    ctx: Context,
    project: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves the most recent session checkpoint.
    Deterministic — does NOT use semantic search.
    Returns the checkpoint with the highest event_seq where memory_type == "checkpoint".

    Use this to answer "what did we do last session?" or to resume work from where you left off.
    """
    logger.info(
        f"Tool 'get_last_checkpoint' called with project={project}, thread_id={thread_id}"
    )
    try:
        memory_service = _require_memory_service(ctx)
        checkpoint = await memory_service.get_last_checkpoint(
            project=project,
            thread_id=thread_id,
        )
        if checkpoint:
            logger.info(f"Last checkpoint found: {checkpoint.get('id', 'unknown')}")
            return {"checkpoint": checkpoint, "status": "found"}
        logger.info("No matching checkpoint found.")
        return {"checkpoint": None, "status": "not_found"}
    except Exception as e:
        logger.error(f"Error during get_last_checkpoint: {e}", exc_info=True)
        return {"error": f"Failed to get last checkpoint: {str(e)}"}


@mcp.tool()
async def get_recent_events(
    ctx: Context,
    n: int = 20,
    project: Optional[str] = None,
    thread_id: Optional[str] = None,
    memory_type: Optional[str] = None,
    since_seq: Optional[int] = None,
    since_time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves the N most recent memory events, ordered by event_seq (descending).
    Does NOT use semantic similarity — purely metadata-driven.

    Use this when you need chronologically ordered results, such as:
    - "Show me the last 10 things we worked on"
    - "What happened after event_seq 42?"
    - "List recent decisions for project NovaTrade"

    Optional filters narrow the result set before sorting.
    """
    logger.info(
        f"Tool 'get_recent_events' called with n={n}, project={project}, "
        f"thread_id={thread_id}, memory_type={memory_type}, "
        f"since_seq={since_seq}, since_time={since_time}"
    )
    try:
        memory_service = _require_memory_service(ctx)
        if n < 1:
            return {"error": "n must be >= 1"}

        events = await memory_service.get_recent_events(
            n=n,
            project=project,
            thread_id=thread_id,
            memory_type=memory_type,
            since_seq=since_seq,
            since_time=since_time,
        )
        logger.info(f"get_recent_events returning {len(events)} events.")
        return {"events": events, "count": len(events)}
    except Exception as e:
        logger.error(f"Error during get_recent_events: {e}", exc_info=True)
        return {"error": f"Failed to get recent events: {str(e)}"}


@mcp.tool()
async def get_session_events(
    ctx: Context,
    session_id: str,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Retrieves all events belonging to a specific session via graph traversal.
    Returns events ordered by event_seq descending. Uses Neo4j INCLUDES edges.

    Use this to inspect exactly what happened during a specific session.
    """
    logger.info(f"Tool 'get_session_events' called for session_id='{session_id}', limit={limit}")
    try:
        memory_service = _require_memory_service(ctx)
        events = await memory_service.graph_client.get_session_events(
            session_id=session_id, limit=limit
        )
        logger.info(f"get_session_events returning {len(events)} events.")
        return {"events": events, "count": len(events), "session_id": session_id}
    except Exception as e:
        logger.error(f"Error during get_session_events: {e}", exc_info=True)
        return {"error": f"Failed to get session events: {str(e)}"}


@mcp.tool()
async def check_health(ctx: Context) -> Dict[str, Any]:
    """
    Checks the health of the memory service and its dependencies (Pinecone, Neo4j).
    """
    logger.info("Tool 'check_health' called.")
    try:
        memory_service = _require_memory_service(ctx)
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
