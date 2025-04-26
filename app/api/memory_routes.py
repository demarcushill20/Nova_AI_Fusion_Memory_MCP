import logging
from fastapi import APIRouter, HTTPException, Depends, status, Path
from typing import List, Dict, Any

# Import Pydantic models and the core MemoryService
try:
    from ..models import (
        QueryRequest, QueryResponse, MemoryItem,
        UpsertRequest, UpsertResponse,
        DeleteResponse, HealthResponse
    )
    from ..services.memory_service import MemoryService
except ImportError as e:
    print(f"Error importing modules in memory_routes.py: {e}. Ensure models.py and memory_service.py exist.")
    raise

logger = logging.getLogger(__name__)

# --- Router Setup ---
# Using a prefix makes it easy to version or group API endpoints
# Example: prefix="/api/v1/memory"
router = APIRouter(
    prefix="/memory", # Base path for all memory-related routes
    tags=["Memory Operations"] # Tag for grouping in API docs
)

# --- Service Instantiation ---
# Instantiate the MemoryService.
# In a real application, consider using FastAPI's dependency injection system
# tied to the application lifespan for better resource management.
# For now, a module-level instance is created. It needs async initialization.
# We'll rely on the lifespan manager in main.py to call initialize.
# TODO: Refactor to use proper dependency injection later if needed.
memory_service_instance = MemoryService()

# --- Dependency for Initialized Service ---
# This dependency ensures the service is initialized before routes use it.
# It doesn't re-initialize but checks the flag set during app startup.
async def get_memory_service() -> MemoryService:
    """
    Dependency function to get the initialized MemoryService instance.
    Relies on the service being initialized during application startup via lifespan.
    """
    # In a real app, the lifespan manager in main.py should call memory_service_instance.initialize()
    # Here, we assume it has been called. If not initialized, methods should handle it.
    if not memory_service_instance._initialized:
         # This state should ideally not be reached if lifespan management is correct.
         logger.error("MemoryService accessed before initialization!")
         raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
             detail="Memory service is not ready. Initialization may have failed."
         )
    return memory_service_instance

# --- API Endpoints ---

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Memory",
    description="Retrieves relevant memory items based on a query string using the fused retrieval pipeline."
)
async def query_memory(
    request: QueryRequest,
    service: MemoryService = Depends(get_memory_service)
):
    """
    Handles POST requests to query the memory system.
    """
    try:
        results = await service.perform_query(request.query)
        # Convert results to MemoryItem models if needed (assuming perform_query returns dicts matching the model)
        # If perform_query already returns Pydantic models, this conversion isn't needed.
        # Assuming perform_query returns list of dicts for now:
        # Map the result dictionary fields to the MemoryItem model fields
        response_results = []
        for res in results:
            # Prioritize 'rerank_score', fall back to 'fusion_score' or 0.0 if missing
            score_value = res.get('rerank_score', res.get('fusion_score', 0.0))
            try:
                item = MemoryItem(
                    id=res.get('id', 'unknown_id'),
                    text=res.get('text', ''),
                    source=res.get('source', 'unknown'),
                    score=float(score_value), # Use the determined score
                    metadata=res.get('metadata')
                )
                response_results.append(item)
            except Exception as item_exc:
                 # Log if a specific item fails validation, but continue processing others
                 logger.error(f"Failed to validate MemoryItem for result ID {res.get('id', 'N/A')}: {item_exc}", exc_info=True)

        return QueryResponse(results=response_results)
    except Exception as e:
        logger.error(f"❌ Error during /query endpoint execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while querying memory: {e}"
        )

@router.post(
    "/upsert",
    response_model=UpsertResponse,
    summary="Upsert Memory Item",
    description="Adds a new memory item or updates an existing one."
)
async def upsert_memory(
    request: UpsertRequest,
    service: MemoryService = Depends(get_memory_service)
):
    """
    Handles POST requests to add or update a memory item.
    """
    try:
        upserted_id = await service.perform_upsert(
            content=request.content,
            memory_id=request.id,
            metadata=request.metadata
        )
        if upserted_id:
            return UpsertResponse(id=upserted_id, status="success")
        else:
            # If perform_upsert returns None, it indicates failure
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upsert memory item. Check server logs for details."
            )
    except Exception as e:
        logger.error(f"❌ Error during /upsert endpoint execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during memory upsert: {e}"
        )

@router.delete(
    "/{memory_id}",
    response_model=DeleteResponse,
    summary="Delete Memory Item",
    description="Deletes a memory item by its unique ID."
)
async def delete_memory(
    memory_id: str = Path(..., description="The unique ID of the memory item to delete."),
    service: MemoryService = Depends(get_memory_service)
):
    """
    Handles DELETE requests to remove a memory item.
    """
    try:
        success = await service.perform_delete(memory_id)
        if success:
            return DeleteResponse(id=memory_id, status="deleted")
        else:
            # Deletion might fail if underlying stores fail, even if ID existed.
            # Or it might return True even if ID didn't exist.
            # We return success based on the service's boolean response.
            # If specific "not found" is needed, service logic should change.
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or 404 if service indicated not found specifically
                detail=f"Failed to delete memory item with ID {memory_id}. Check server logs."
            )
    except Exception as e:
        logger.error(f"❌ Error during /memory/{memory_id} DELETE endpoint execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during memory deletion: {e}"
        )

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Checks the operational status of the server and its dependencies (Pinecone, Neo4j, Reranker)."
)
async def health_check(
    service: MemoryService = Depends(get_memory_service)
):
    """
    Handles GET requests to check the health of the service.
    """
    try:
        health_status = await service.check_health()
        if health_status.get("status") == "error":
            # Raise 503 if any critical component reported an error
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_status
            )
        # Map the dictionary from check_health to the Pydantic model
        return HealthResponse(**health_status)
    except HTTPException as http_exc:
         # Re-raise HTTPException if already raised (e.g., 503 from check_health)
         raise http_exc
    except Exception as e:
        logger.error(f"❌ Error during /health endpoint execution: {e}", exc_info=True)
        # Return a generic 500 error if the health check itself fails unexpectedly
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during health check: {e}"
        )