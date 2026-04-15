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
from app.services.associations.associative_recall import INTENT_EDGE_FILTER, DIRECTED_EDGE_DIRECTION
from app.services.associations.memory_edges import VALID_EDGE_TYPES

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

# --- Module-level constants ---

VALID_INTENTS = frozenset(INTENT_EDGE_FILTER.keys())

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
    expand_graph: bool = False,
    intent: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves relevant memory items based on a query text.
    Uses vector search (Pinecone) and graph search (Neo4j), fuses results, and reranks.
    Supports optional retrieval controls and metadata filtering.

    When expand_graph is True and ASSOC_GRAPH_RECALL_ENABLED is set, results are
    expanded via associative graph traversal. The intent parameter controls which
    edge types are prioritized during expansion (e.g. 'temporal_recall',
    'decision_recall', 'entity_recall', 'general').
    """
    logger.info(
        "Tool 'query_memory' called with query=%s, top_k_vector=%s, "
        "top_k_final=%s, category=%s, tags=%s, min_score=%s, "
        "run_id=%s, expand_graph=%s, intent=%s",
        query[:200], top_k_vector, top_k_final, category, tags,
        min_score, run_id, expand_graph, intent,
    )
    try:
        memory_service = _require_memory_service(ctx)
        if intent is not None and intent not in VALID_INTENTS:
            return {"error": f"Invalid intent: {intent!r}. Valid values: {sorted(VALID_INTENTS)}"}
        if isinstance(top_k_vector, bool) or isinstance(top_k_final, bool):
            return {"error": "top_k_vector and top_k_final must be integers, not booleans"}
        if top_k_vector < 1 or top_k_final < 1:
            return {"error": "top_k_vector and top_k_final must be >= 1"}
        top_k_vector = min(top_k_vector, 500)
        top_k_final = min(top_k_final, 200)

        results = await memory_service.perform_query(
            query_text=query,
            top_k_vector=top_k_vector,
            top_k_final=top_k_final,
            expand_graph=expand_graph,
            intent=intent,
        )
        filtered_results = _filter_query_results(
            results=results,
            category=category,
            tags=tags,
            min_score=min_score,
            run_id=run_id,
        )
        logger.info(
            "Query returned %d results before filtering, %d after filtering.",
            len(results), len(filtered_results),
        )
        # FastMCP automatically serializes the return value (dict, list, primitives) to JSON
        return {"results": filtered_results}
    except Exception as e:
        logger.error("Error during query_memory: %s", e, exc_info=True)
        # Return an error structure that MCP client can understand
        return {"error": f"Failed to execute query: {type(e).__name__}"}


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


# --- Associative Linking Tools (Phase 8, PLAN-0759) ---
# SLO: all read-only association tools target p99 < 500ms for typical graphs.
# These tools are always available (no feature flag gate) because they are
# read-only. The only exception is get_related_memories with expand via
# AssociativeRecall, which is gated on ASSOC_GRAPH_RECALL_ENABLED.


def _get_edge_service(memory_service: "MemoryService"):
    """Lazily construct a MemoryEdgeService from the MemoryService's graph driver.

    Returns a cached instance so repeated tool calls within the same
    server lifetime reuse the same executor.
    """
    if not hasattr(memory_service, "_mcp_edge_service") or memory_service._mcp_edge_service is None:
        from app.services.associations.edge_service import MemoryEdgeService

        memory_service._mcp_edge_service = MemoryEdgeService(memory_service.graph_client.driver)
    return memory_service._mcp_edge_service


def _get_entity_linker(memory_service: "MemoryService"):
    """Lazily construct an EntityLinker from the MemoryService's graph driver.

    Returns a cached instance so repeated tool calls within the same
    server lifetime reuse the same executor.
    """
    if not hasattr(memory_service, "_mcp_entity_linker") or memory_service._mcp_entity_linker is None:
        from app.services.associations.entity_linker import EntityLinker

        memory_service._mcp_entity_linker = EntityLinker(memory_service.graph_client.driver)
    return memory_service._mcp_entity_linker


@mcp.tool()
async def get_related_memories(
    ctx: Context,
    memory_id: str,
    edge_types: Optional[List[str]] = None,
    max_hops: int = 2,
    limit: int = 20,
    intent: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get memories related to a given memory via associative graph edges.
    Traverses the memory graph up to max_hops hops from the seed memory.

    SLO: p99 < 500ms for max_hops <= 2.

    Args:
        memory_id: The entity_id of the seed memory (required).
        edge_types: Optional list of edge types to traverse (e.g. ['SIMILAR_TO', 'MENTIONS']).
        max_hops: Maximum traversal depth (default 2, max 3).
        limit: Maximum number of related memories to return (default 20, max 50).
        intent: Recall intent for edge prioritization (e.g. 'temporal_recall', 'entity_recall', 'general').
    """
    logger.info(
        "Tool 'get_related_memories' called: memory_id=%s, edge_types=%s, "
        "max_hops=%s, limit=%s, intent=%s",
        memory_id, edge_types, max_hops, limit, intent,
    )
    try:
        memory_service = _require_memory_service(ctx)

        # --- Input validation ---
        if not isinstance(memory_id, str) or not memory_id.strip():
            return {"error": "memory_id must be a non-empty string"}
        memory_id = memory_id.strip()

        if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
            max_hops = 1
        if max_hops > 3:
            max_hops = 3

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            limit = 1
        if limit > 50:
            limit = 50

        if intent is not None and intent not in VALID_INTENTS:
            return {"error": f"Invalid intent: {intent!r}. Valid values: {sorted(VALID_INTENTS)}"}

        if edge_types is not None:
            if not isinstance(edge_types, list) or not edge_types:
                return {"error": "edge_types must be a non-empty list of strings"}
            invalid = [et for et in edge_types if not isinstance(et, str) or et not in VALID_EDGE_TYPES]
            if invalid:
                return {"error": f"Invalid edge_types: {invalid!r}. Valid: {sorted(VALID_EDGE_TYPES)}"}

        if edge_types is None and intent is not None:
            edge_types = list(INTENT_EDGE_FILTER.get(intent, INTENT_EDGE_FILTER["general"]))

        edge_service = _get_edge_service(memory_service)

        # Multi-hop BFS: collect neighbors iteratively
        visited: set = {memory_id}
        related: List[Dict[str, Any]] = []
        frontier = [memory_id]

        for hop in range(1, max_hops + 1):
            next_frontier: List[str] = []
            for node_id in frontier:
                active_types = edge_types
                neighbors: list = []
                if active_types:
                    symmetric = [et for et in active_types if et not in DIRECTED_EDGE_DIRECTION]
                    directed = [et for et in active_types if et in DIRECTED_EDGE_DIRECTION]
                    if symmetric:
                        neighbors.extend(await edge_service.get_neighbors(
                            node_id=node_id, edge_types=symmetric,
                            direction="both", limit=25,
                        ))
                    if directed:
                        neighbors.extend(await edge_service.get_neighbors(
                            node_id=node_id, edge_types=directed,
                            direction="out", limit=25,
                        ))
                else:
                    neighbors = await edge_service.get_neighbors(
                        node_id=node_id, limit=50,
                    )
                for n in neighbors:
                    nid = n["node_id"]
                    if nid not in visited:
                        visited.add(nid)
                        related.append({
                            "memory_id": nid,
                            "edge_type": n["edge_type"],
                            "weight": n["weight"],
                            "hop_distance": hop,
                            "metadata": {
                                "created_at": n.get("created_at"),
                                "last_seen_at": n.get("last_seen_at"),
                            },
                        })
                        next_frontier.append(nid)
            frontier = next_frontier[:20]
            if not frontier or len(related) >= limit:
                break

        # Sort by relevance (weight desc, then nearest hop) before truncation
        related.sort(key=lambda x: (-x.get("weight", 0), x["hop_distance"]))
        related = related[:limit]

        logger.info(
            "get_related_memories returning %d related memories for %s",
            len(related), memory_id,
        )
        return {"memory_id": memory_id, "related": related, "count": len(related)}

    except Exception as e:
        logger.error("Error during get_related_memories: %s", e, exc_info=True)
        return {"error": f"Failed to get related memories: {type(e).__name__}"}


@mcp.tool()
async def get_entity_memories(
    ctx: Context,
    entity_name: str,
    project: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Get memories that mention a specific entity.
    Uses the Entity graph to find all memories linked via MENTIONS edges.

    SLO: p99 < 300ms for typical entity lookups.

    Args:
        entity_name: The entity name to search for (required). Will be canonicalized.
        project: Optional project namespace to scope the search.
        limit: Maximum number of memories to return (default 20, max 100).
    """
    logger.info(
        "Tool 'get_entity_memories' called: entity_name=%s, project=%s, limit=%s",
        entity_name, project, limit,
    )
    try:
        memory_service = _require_memory_service(ctx)

        # --- Input validation ---
        if not isinstance(entity_name, str) or not entity_name.strip():
            return {"error": "entity_name must be a non-empty string"}
        entity_name = entity_name.strip()

        if project is not None and (not isinstance(project, str) or not project.strip()):
            return {"error": "project must be a non-empty string or null"}
        if project is not None:
            project = project.strip()

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            limit = 1
        if limit > 100:
            limit = 100

        # EntityLinker requires a project; if not provided, return informative error
        if project is None:
            return {
                "error": "project is required for entity lookups (entity nodes are scoped by project)"
            }

        entity_linker = _get_entity_linker(memory_service)
        memories = await entity_linker.get_memories_for_entity(
            project=project,
            entity_name=entity_name,
            limit=limit,
        )

        # Shape the response
        result_list = []
        for mem in memories:
            result_list.append({
                "memory_id": mem["memory_id"],
                "created_at": mem.get("created_at"),
                "last_seen_at": mem.get("last_seen_at"),
            })

        logger.info(
            "get_entity_memories returning %d memories for entity=%s project=%s",
            len(result_list), entity_name, project,
        )
        return {
            "entity_name": entity_name,
            "project": project,
            "memories": result_list,
            "count": len(result_list),
        }

    except Exception as e:
        logger.error("Error during get_entity_memories: %s", e, exc_info=True)
        return {"error": f"Failed to get entity memories: {type(e).__name__}"}


@mcp.tool()
async def get_memory_graph(
    ctx: Context,
    memory_id: str,
    max_hops: int = 2,
    edge_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get the subgraph around a memory node: nodes and edges within max_hops.
    Returns a graph structure suitable for visualization or analysis.

    SLO: p99 < 800ms for max_hops <= 2.
    Response cap: 200 nodes, 400 edges.

    Args:
        memory_id: The entity_id of the center memory (required).
        max_hops: Maximum traversal depth (default 2, max 3).
        edge_types: Optional list of edge types to include.
    """
    logger.info(
        "Tool 'get_memory_graph' called: memory_id=%s, max_hops=%s, edge_types=%s",
        memory_id, max_hops, edge_types,
    )
    try:
        memory_service = _require_memory_service(ctx)

        # --- Input validation ---
        if not isinstance(memory_id, str) or not memory_id.strip():
            return {"error": "memory_id must be a non-empty string"}
        memory_id = memory_id.strip()

        if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
            max_hops = 1
        if max_hops > 3:
            max_hops = 3

        if edge_types is not None:
            if not isinstance(edge_types, list) or not edge_types:
                return {"error": "edge_types must be a non-empty list of strings"}
            invalid = [et for et in edge_types if not isinstance(et, str) or et not in VALID_EDGE_TYPES]
            if invalid:
                return {"error": f"Invalid edge_types: {invalid!r}. Valid: {sorted(VALID_EDGE_TYPES)}"}

        MAX_NODES = 200
        MAX_EDGES = 400

        edge_service = _get_edge_service(memory_service)

        # Build subgraph via iterative BFS
        nodes: Dict[str, Dict[str, Any]] = {
            memory_id: {"id": memory_id, "type": "memory", "hop": 0}
        }
        edges: List[Dict[str, Any]] = []
        seen_edges: set = set()
        frontier = [memory_id]

        for hop in range(1, max_hops + 1):
            if len(nodes) >= MAX_NODES:
                break
            next_frontier: List[str] = []
            for node_id in frontier:
                active_types = edge_types
                neighbors: list = []
                if active_types:
                    symmetric = [et for et in active_types if et not in DIRECTED_EDGE_DIRECTION]
                    directed = [et for et in active_types if et in DIRECTED_EDGE_DIRECTION]
                    if symmetric:
                        neighbors.extend(await edge_service.get_neighbors(
                            node_id=node_id, edge_types=symmetric,
                            direction="both", limit=25,
                        ))
                    if directed:
                        neighbors.extend(await edge_service.get_neighbors(
                            node_id=node_id, edge_types=directed,
                            direction="out", limit=25,
                        ))
                else:
                    neighbors = await edge_service.get_neighbors(
                        node_id=node_id, limit=50,
                    )
                for n in neighbors:
                    nid = n["node_id"]
                    # Add edge (regardless of whether node was already seen)
                    edge_key = (node_id, nid, n["edge_type"])
                    if len(edges) < MAX_EDGES and edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append({
                            "source": node_id,
                            "target": nid,
                            "type": n["edge_type"],
                            "weight": n["weight"],
                            "created_at": n.get("created_at"),
                            "last_seen_at": n.get("last_seen_at"),
                        })
                    # Add node if new
                    if nid not in nodes and len(nodes) < MAX_NODES:
                        nodes[nid] = {"id": nid, "type": "memory", "hop": hop}
                        next_frontier.append(nid)
            frontier = next_frontier[:50]
            if not frontier or len(nodes) >= MAX_NODES:
                break

        node_list = list(nodes.values())
        logger.info(
            "get_memory_graph returning %d nodes, %d edges for %s",
            len(node_list), len(edges), memory_id,
        )
        return {
            "memory_id": memory_id,
            "nodes": node_list,
            "edges": edges,
            "node_count": len(node_list),
            "edge_count": len(edges),
            "truncated": len(node_list) >= MAX_NODES or len(edges) >= MAX_EDGES,
        }

    except Exception as e:
        logger.error("Error during get_memory_graph: %s", e, exc_info=True)
        return {"error": f"Failed to get memory graph: {type(e).__name__}"}


@mcp.tool()
async def get_provenance(
    ctx: Context,
    memory_id: str,
    max_depth: int = 10,
) -> Dict[str, Any]:
    """
    Get the provenance chain of a memory — traces SUPERSEDES, PROMOTED_FROM,
    and COMPACTED_FROM edges to find original episodic sources.

    SLO: p99 < 400ms for typical provenance chains (depth <= 10).
    Response cap: 30 nodes in the provenance chain.

    Args:
        memory_id: The entity_id of the memory to trace (required).
        max_depth: Maximum depth to traverse (default 10, clamped to [1, 10]).
    """
    logger.info(
        "Tool 'get_provenance' called: memory_id=%s, max_depth=%s",
        memory_id, max_depth,
    )
    try:
        memory_service = _require_memory_service(ctx)

        # --- Input validation ---
        if not isinstance(memory_id, str) or not memory_id.strip():
            return {"error": "memory_id must be a non-empty string"}
        memory_id = memory_id.strip()

        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
            max_depth = 1
        if max_depth > 10:
            max_depth = 10

        MAX_CHAIN_NODES = 30

        edge_service = _get_edge_service(memory_service)
        prov = await edge_service.get_provenance(
            memory_id=memory_id,
            max_depth=max_depth,
        )

        # Enforce response-size cap on provenance chain
        full_chain = prov.get("provenance_chain", [])
        chain = full_chain[:MAX_CHAIN_NODES]
        result = {
            "memory_id": prov["memory_id"],
            "provenance_chain": chain,
            "original_sources": prov.get("original_sources", [])[:MAX_CHAIN_NODES],
            "depth": prov.get("depth", 0),
            "depth_limited": prov.get("depth_limited", False),
            "chain_count": len(chain),
            "full_chain_count": len(full_chain),
            "truncated": len(full_chain) > MAX_CHAIN_NODES,
        }

        logger.info(
            "get_provenance returning chain of %d for %s (depth=%d)",
            len(chain), memory_id, result["depth"],
        )
        return result

    except Exception as e:
        logger.error("Error during get_provenance: %s", e, exc_info=True)
        return {"error": f"Failed to get provenance: {type(e).__name__}"}


@mcp.tool()
async def get_session_timeline(
    ctx: Context,
    session_id: str,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Get the ordered timeline for a session via INCLUDES edges.
    Returns memories ordered by event_seq, showing the causal flow within a session.

    SLO: p99 < 400ms for typical sessions (< 100 memories).
    Response cap: 100 memories.

    Args:
        session_id: The session identifier (required).
        limit: Maximum number of memories to return (default 50, max 100).
    """
    logger.info(
        "Tool 'get_session_timeline' called: session_id=%s, limit=%s",
        session_id, limit,
    )
    try:
        memory_service = _require_memory_service(ctx)

        # --- Input validation ---
        if not isinstance(session_id, str) or not session_id.strip():
            return {"error": "session_id must be a non-empty string"}
        session_id = session_id.strip()

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            limit = 1
        if limit > 100:
            limit = 100

        # Query Neo4j for memories included in this session. Find all
        # :base nodes linked via INCLUDES edges from the :Session node,
        # and order by event_seq.
        driver = memory_service.graph_client.driver
        query = (
            "MATCH (s:Session {entity_id: $session_id})-[:INCLUDES]->(m:base)\n"
            "RETURN m.entity_id AS memory_id, "
            "m.event_seq AS event_seq, "
            "m.event_time AS created_at\n"
            "ORDER BY m.event_seq ASC\n"
            "LIMIT $limit"
        )

        memories: List[Dict[str, Any]] = []
        db_name = getattr(memory_service.graph_client, "_database", "neo4j")
        async with driver.session(database=db_name) as session:
            result = await session.run(
                query, {"session_id": session_id, "limit": limit}
            )
            async for record in result:
                memories.append({
                    "memory_id": record["memory_id"],
                    "event_seq": record["event_seq"],
                    "created_at": record["created_at"],
                })
            await result.consume()

        logger.info(
            "get_session_timeline returning %d memories for session=%s",
            len(memories), session_id,
        )
        return {
            "session_id": session_id,
            "timeline": memories,
            "count": len(memories),
        }

    except Exception as e:
        logger.error("Error during get_session_timeline: %s", e, exc_info=True)
        return {"error": f"Failed to get session timeline: {type(e).__name__}"}


@mcp.tool()
async def get_edge_stats(ctx: Context) -> Dict[str, Any]:
    """
    Get global edge statistics for the associative memory graph.
    Returns aggregate counts and weight distributions for all edge types.

    SLO: p99 < 1s (scans all edges; acceptable for an admin/observability tool).
    """
    logger.info("Tool 'get_edge_stats' called.")
    try:
        memory_service = _require_memory_service(ctx)
        edge_service = _get_edge_service(memory_service)

        stats = await edge_service.get_edge_stats()

        # Compute a total edge count for convenience
        total_edges = sum(v.get("count", 0) for v in stats.values())

        logger.info("get_edge_stats returning stats for %d edge types, %d total edges",
                     len(stats), total_edges)
        return {
            "edge_stats": stats,
            "total_edges": total_edges,
            "edge_type_count": len(stats),
        }

    except Exception as e:
        logger.error("Error during get_edge_stats: %s", e, exc_info=True)
        return {"error": f"Failed to get edge stats: {type(e).__name__}"}


if __name__ == "__main__":
    # This allows running the server directly using `python mcp_server.py`
    # It will use the stdio transport by default.
    logger.info("Starting Nova Memory MCP Server directly...")
    mcp.run()
