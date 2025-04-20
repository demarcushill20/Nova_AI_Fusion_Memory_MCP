from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- API Request Models ---

class QueryRequest(BaseModel):
    """
    Request model for the /query endpoint.
    """
    query: str = Field(..., description="The query text to search for in memory.")
    # Optional: Add parameters like top_k if needed for the API layer
    # top_k_vector: Optional[int] = Field(50, description="Number of initial results from vector store.")
    # top_k_final: Optional[int] = Field(15, description="Number of final results after reranking.")

class UpsertRequest(BaseModel):
    """
    Request model for the /upsert endpoint.
    """
    id: Optional[str] = Field(None, description="Optional unique ID for the memory item. If None, an ID will be generated (MD5 hash of content).")
    content: str = Field(..., description="The text content of the memory item to be stored.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary of metadata associated with the content.")

# Note: Delete request uses path parameter, so no specific request body model needed here.

# --- API Response Models ---

class MemoryItem(BaseModel):
    """
    Represents a single memory item returned in query results.
    """
    id: str = Field(..., description="Unique identifier of the memory item.")
    text: str = Field(..., description="The text content of the memory item.")
    source: str = Field(..., description="The origin of the result ('vector' or 'graph').")
    score: float = Field(..., description="The relevance score (e.g., RRF score or rerank score).")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Associated metadata.")
    # Optional: Include raw_score or normalized_score if needed for debugging/analysis
    # raw_score: Optional[float] = None
    # normalized_score: Optional[float] = None
    # rrf_score: Optional[float] = None # RRF score specifically
    # rerank_score: Optional[float] = None # Rerank score specifically

class QueryResponse(BaseModel):
    """
    Response model for the /query endpoint.
    """
    results: List[MemoryItem] = Field(..., description="List of relevant memory items, sorted by relevance.")

class UpsertResponse(BaseModel):
    """
    Response model for the /upsert endpoint.
    """
    id: str = Field(..., description="The unique ID of the upserted memory item.")
    status: str = Field(..., description="Status of the upsert operation (e.g., 'success', 'updated').")

class DeleteResponse(BaseModel):
    """
    Response model for the /memory/{id} DELETE endpoint.
    """
    id: str = Field(..., description="The unique ID of the memory item targeted for deletion.")
    status: str = Field(..., description="Status of the delete operation (e.g., 'deleted').")

class HealthResponse(BaseModel):
    """
    Response model for the /health endpoint.
    """
    status: str = Field(..., description="Overall status ('ok' or 'error').")
    pinecone: Optional[str] = Field(None, description="Status of the Pinecone connection ('ok' or 'error: <details>').")
    graph: Optional[str] = Field(None, description="Status of the Neo4j connection ('ok' or 'error: <details>').") # Changed from 'neo4j' to 'graph' for consistency
    reranker: Optional[str] = Field(None, description="Status of the reranker model ('loaded' or 'disabled/failed').")
    detail: Optional[str] = Field(None, description="Additional details in case of overall error status.")