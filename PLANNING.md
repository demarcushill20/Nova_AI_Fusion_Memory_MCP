Great — I’ll prepare a full implementation plan for turning Nova AI’s existing memory system into a modular, local-first MCP (Model Context Protocol) server with REST API endpoints. This version will:

- Use the exact memory logic from Nova AI (including Pinecone, Neo4j, reranking, hybrid merging, and query routing)
- Support only fused memory queries (vector + graph)
- Be accessible via a clean API (`/query`, `/upsert`, `/delete`)
- Be designed for easy future cloud deployment

I’ll break it down step-by-step, including architecture, file structure, FastAPI routes, integration of existing modules, and examples for testing. I’ll let you know as soon as it’s ready!

# Nova AI Memory MCP Server – Implementation Plan

## Project Structure and Organization

Organize the MCP server codebase as a standalone FastAPI application with a clear, modular structure. A suggested folder/file layout is:

```
nova_memory_mcp/
├── app/
│   ├── main.py              # FastAPI app initialization and server startup
│   ├── api/
│   │   └── memory_routes.py # Defines FastAPI routes: /query, /upsert, /memory/{id}, /health
│   ├── services/
│   │   ├── memory_service.py  # Core logic wrapping Nova AI’s memory components
│   │   ├── pinecone_client.py # Helper to initialize/connect to Pinecone index
│   │   └── graph_client.py    # Helper to initialize/connect to Neo4j (LightRAG) graph
│   ├── models.py           # (Optional) Pydantic models for request/response schemas
│   └── config.py           # Configuration management (Pinecone keys, Neo4j creds, model paths)
├── requirements.txt         # Dependencies (FastAPI, Uvicorn, pinecone-client, neo4j-driver, lightrag, etc.)
├── Dockerfile               # Container image setup for the MCP server
├── docker-compose.yml       # (Optional) for local dev: services for Neo4j, etc.
├── .env                     # (Optional) Local environment variables (NEO4J_URI, Pinecone keys, etc.)
└── README.md                # Developer guide for running and deploying the server
```

**Key points:**

- The **`app/main.py`** will create the FastAPI app and include the API router. It might also handle any startup events (e.g. connecting to Pinecone/Neo4j).
- The **`app/api/memory_routes.py`** will contain the FastAPI route functions (`/query`, `/upsert`, `/memory/{id}`, `/health`), separated for clarity.
- The **`app/services/memory_service.py`** encapsulates the memory retrieval pipeline. This is where Nova AI’s memory logic (query routing, hybrid retrieval, merging, reranking) is orchestrated. By separating this from the FastAPI routes, we keep a clean architecture – the service can be used independently of the web API (helpful for testing or future reuse).
- Helper modules like **`pinecone_client.py`** and **`graph_client.py`** can initialize and hold references to external systems (Pinecone index, Neo4j connection via LightRAG). This avoids cluttering the main code with connection details and makes it easy to swap out or configure these components.
- If needed, **`app/models.py`** can define Pydantic models for request bodies and response formats (for example, a `QueryRequest` model with a `query: str` field, an `UpsertRequest` with fields for `id`, `content`, and optional `metadata`, etc., and perhaps a `MemoryItem` model for results).
- We use a **`config.py`** (or a `settings.py`) to load configuration (from environment variables or .env) such as `PINECONE_API_KEY`, `PINECONE_ENV`, `PINECONE_INDEX_NAME`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and any model names/paths. This centralizes config and makes switching to cloud deployment easier (just change env vars).
- The repository includes **`Dockerfile`** and optionally a **`docker-compose.yml`** for local setup, which we will detail later. Keeping these config files in the project ensures consistency across environments.

This structure ensures a **clean separation of concerns**: API layer vs. memory logic vs. external system clients vs. configuration. Such modularity will facilitate local development and later cloud deployment (you can swap implementations or adjust configurations without touching the core logic).

## Reusing Nova AI Memory Modules

We will **reuse Nova AI’s existing memory components** directly, rather than rewriting their logic. This means importing and integrating the modules like `query_router.py`, `hybrid_merger.py`, and `reranker.py` into the MCP server. There are a couple of ways to achieve this cleanly:

- **Include Nova’s modules in the project**: If Nova AI’s memory code is accessible (as part of the same codebase or installable), we can add those Python files to our `services` or a `nova_memory` package folder. For example, copy or symlink `hybrid_merger.py`, `query_router.py`, `reranker.py` into `app/services/` (or include the whole Nova memory package if one exists). This way, we can do `from app.services.query_router import QueryRouter` etc. **without modifying their content**. We want one source of truth for these algorithms.
- **Install as a dependency**: If Nova AI’s memory system is packaged or can be installed (e.g. via pip or as a git submodule), we could add it to `requirements.txt`. For instance, if there’s a package `nova_memory`, we might do `pip install nova_memory` and then just import the needed classes. This is ideal to avoid code duplication. If that’s not available, the first approach (including the files) is fine for now.
- **Module import examples**: In our FastAPI app, we will import and initialize these components. For example: 

  ```python
  # memory_service.py
  from query_router import QueryRouter, RoutingMode
  from hybrid_merger import HybridMerger
  from reranker import CrossEncoderReranker

  # Initialize router, merger, reranker
  router = QueryRouter()  # QueryRouter.route() will classify queries
  merger = HybridMerger()  # Handles RRF merging and deduplication
  reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
  ```
  
  We’ll reuse their interfaces as-is. For example, Nova’s `QueryRouter.route()` returns a `RoutingMode` (VECTOR, GRAPH, or HYBRID). Nova’s `HybridMerger.merge_results(vector_results, graph_results)` already implements Reciprocal Rank Fusion and deduplication. The `CrossEncoderReranker.rerank(query, results)` method reorders results by relevance. By using these directly, we preserve the “sophisticated logic” that Nova AI had.

- **Avoid altering Nova’s logic**: We will **not rewrite or simplify** the internal algorithms. The MCP server will call these modules just like Nova AI did. For instance, if Nova’s code normalizes vector vs. graph scores or uses certain parameters (like the `rrf_k` constant in RRF), those will remain unchanged to ensure we get identical behavior. This means our server’s retrieval results will match what Nova’s in-app memory system would produce.

- **Unified (fused) mode only**: Note that Nova’s QueryRouter could route queries to only vector or only graph. Since the MCP server is intended to **support only hybrid fused retrieval**, we will generally default to using both stores for each query. We can still utilize `QueryRouter.route(query)` to gauge the query’s nature (for logging or potential future use), but our implementation will **not expose separate modes**. In practice, this means even if `QueryRouter` classifies a query as `VECTOR` or `GRAPH`, the service will still retrieve from both and merge. (Alternatively, we could decide to honor the classification internally to optimize – e.g. if `route==VECTOR` skip graph retrieval – but since the prompt says fused only, we will *always* perform both retrievals for completeness. We may log the routing decision, but we won’t require the client to choose a mode.)

- **Neo4j via LightRAG**: Nova’s code uses `Neo4JStorage` from the LightRAG library (`lightrag.kg.neo4j_impl`). We should reuse this for graph connectivity. For example, we can initialize the graph store in our service startup:
  
  ```python
  from lightrag.kg.neo4j_impl import Neo4JStorage
  # ...
  embedding_func = ...  # embedding function (same model as Pinecone uses)
  try:
      graph_store = Neo4JStorage(namespace="nova_memory", config=neo4j_config, embedding_func=embedding_func)
  except Exception as e:
      graph_store = None
      print(f"Neo4j connection failed: {e}")
  ```
  
  This will give us an object similar to `nova_ai_instance.memory.graph_store` in Nova. We can then call methods on `graph_store` to retrieve data. Nova’s implementation calls `graph_store.get_knowledge_graph(node_label="base")` to fetch graph memory. We will use the analogous call (or any provided search method) to get relevant graph results for a query. If LightRAG provides a more direct query interface (like `graph_store.query(query_text)`), that would be ideal – otherwise, we may use the same approach as Nova (fetch all base nodes or a subgraph and rely on RRF/reranker to pick what’s relevant).

By reusing these modules, we ensure the **hybrid retrieval pipeline (vector + graph), RRF merging, deduplication, and cross-encoder reranking** behave exactly as in Nova AI. This not only saves development time but also guarantees continuity in how memory results are produced. Any improvements made to those modules in Nova’s codebase can be pulled into the MCP server easily if we maintain them as shared code.

## FastAPI API Endpoints and Logic

We will create four REST endpoints as specified, each implementing the needed logic by calling into Nova’s memory components through our service layer. Below is a breakdown of each endpoint with its purpose and pseudocode outline:

### **POST** `/query` – Retrieve from Memory

This endpoint accepts a query (question or search string) and returns relevant memory snippets from the combined memory (vector + graph). It uses the full hybrid retrieval pipeline:

- **Request**: JSON body containing the query string (e.g. `{"query": "How does X relate to Y?"}`).
- **Processing**:
  1. **Query Classification** – Use `QueryRouter` to classify the query (VECTOR/GRAPH/HYBRID). As noted, we won’t branch the API behavior by this, but we might log or potentially use it to inform how we query (for example, we could choose to adjust the number of results from each source based on the type).
  2. **Vector Retrieval** – Embed the query using the chosen embedding model (same one used for Pinecone indexing). Query the Pinecone index for top matches. For example, use `pinecone_index.query(vector=query_embedding, top_k=N, include_metadata=True)`. This returns a list of matches (with `id`, `score`, and `metadata` which includes stored text).
  3. **Graph Retrieval** – Query the Neo4j graph for relevant information. If using LightRAG’s storage, this might involve calling a method on `graph_store`. For instance, Nova used `graph_store.get_knowledge_graph(node_label="base")` which returns graph nodes (and maybe relationships) of type “base”. If possible, we should query the graph in a targeted way – e.g. if `graph_store` has a method to search by embedding or by a text key. We may need to embed the query and use Neo4j’s vector index, or run a Cypher query that finds nodes whose content matches the query (depending on how LightRAG is implemented). For planning, we’ll assume we can get a list of graph results (each with an `id`, maybe a `text` or properties, and a score or rank) from `graph_store`.
  4. **Fusion (RRF Merge)** – Use `HybridMerger.merge_results(vector_results, graph_results)` to combine the two result sets. This function will perform **Reciprocal Rank Fusion (RRF)**: each result is scored based on its rank in the individual lists, and the scores are summed ([Better RAG results with Reciprocal Rank Fusion and Hybrid Search](https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search#:~:text=1,the%20rankings%20from%20all%20individual)). Results that appear in both lists are deduplicated (the merger uses an MD5 hash of the text to identify duplicates) and their scores combined. The output is a single list of results sorted by fused relevance.
  5. **Rerank** – Pass the merged results and the original query to the `CrossEncoderReranker`. The reranker (if its model is loaded) will score each result’s text against the query using a cross-encoder model (such as MS MARCO MiniLM). It then returns the top N results sorted by this rerank score. If the reranker is not available (e.g. model not loaded), we fall back to the fused ranking. In our implementation, we will likely load the cross-encoder model at startup (this can be done in a background task to avoid slow start). We’ll default to returning, say, the top 10-15 results after reranking.
  6. **Response** – Return a JSON with the final results list. Each result can include the memory `id`, the `text` content (or snippet), maybe a `source` field indicating it came from vector or graph, and relevance scores if useful (we could include the `score` or `rrf_score` or `rerank_score` for transparency). The exact schema can be defined in `models.py` – for example:
     ```python
     class MemoryItem(BaseModel):
         id: str
         text: str
         source: str  # "vector" or "graph"
         score: float  # maybe the fused or rerank score
         metadata: Optional[dict] = None
     ```
     We might return `{"results": [MemoryItem, ...]}`. For simplicity, a list of dicts is fine.

**Pseudocode for `/query`:**

```python
# app/api/memory_routes.py
from fastapi import APIRouter
from app.services.memory_service import MemoryService, MemoryItem

router = APIRouter()
memory_service = MemoryService()  # This would initialize Pinecone, Neo4j, etc.

@router.post("/query")
async def query_memory(request: QueryRequest):
    query_text = request.query
    # 1. Classify query (optional, for logging/tuning)
    routing = memory_service.router.route(query_text)
    logging.info(f"Query classified as {routing.name}")
    # 2. Retrieve from Pinecone (vector store)
    vector_results = memory_service.search_vector_store(query_text)
    # 3. Retrieve from Neo4j (graph store)
    graph_results = memory_service.search_graph_store(query_text)
    # 4. Merge results using RRF
    fused_results = memory_service.merger.merge_results(vector_results, graph_results)
    # 5. Rerank the fused results (if reranker is enabled)
    final_results = memory_service.rerank_results(query_text, fused_results)
    return {"results": final_results}
```

In the above pseudo-implementation:
- `MemoryService.search_vector_store` would handle embedding the query and calling `pinecone_index.query()`, then formatting the results as needed (e.g. put the match metadata into a `text` field, etc.).
- `MemoryService.search_graph_store` would query Neo4j via LightRAG. If LightRAG’s `Neo4JStorage` returns a list of node data, we’d format each into a dict with `'text'` (maybe the node content or name) and `'score'` if available. If no numeric score is provided by the graph query, we might rely on ordering or assign a default ranking.
- `MemoryService.rerank_results` would call `CrossEncoderReranker.rerank(query, results)` if the reranker model is loaded, otherwise just return the input list (or top N). Note: since `CrossEncoderReranker.rerank` is an async method in Nova’s code, we might need to `await` it. We can make `query_memory` an `async def` and await the reranker (as shown) to avoid blocking.

The logic above ensures we’re using **Nova’s exact pipeline**: classification (but default to hybrid), parallel retrieval from both sources, RRF fusion, deduplication, and cross-encoder reranking. This gives the “fused memory retrieval” result.

### **POST** `/upsert` – Insert/Update Memory

This endpoint allows adding new information to the memory system (or updating existing memory entries). The idea is that clients can store new knowledge or conversation snippets via the MCP server.

- **Request**: JSON body containing the content to upsert. We might accept an optional `id` (if the client wants to specify a key or update an existing entry), a required `content` (the text data to remember), and optional `metadata` (for example, tags, category, source info). For example: 
  ```json
  {
    "id": "memory_123", 
    "content": "Neural networks are inspired by the human brain.", 
    "metadata": {"source": "wikipedia", "topic": "AI"}
  }
  ```
  If `id` is not provided, the server can generate one (e.g. an MD5 or UUID of the content).

- **Processing**:
  1. **ID determination** – Determine a unique ID for this memory. Nova’s code used an MD5 hash of the content as an ID (`vector_id = md5(full_memory.encode()).hexdigest()`) ([Nova_AI.py](file://file-D56GZQX5sXJqwrA1xivL2U#:~:text=,hexdigest)). We can adopt the same, or simply use Pinecone’s ability to upsert with a provided or auto-generated ID. Using a hash ensures duplicates produce the same ID.
  2. **Embedding** – Compute the embedding vector for the content (using the same embedding model as for queries). This is needed for Pinecone (and possibly for the graph if LightRAG stores embeddings).
  3. **Upsert to Pinecone** – Use the Pinecone client to upsert the new vector. For example:
     ```python
     pinecone_index.upsert([{"id": vector_id, "values": embedding, "metadata": metadata_with_text}])
     ```
     Here we include the full text in the metadata (`metadata_with_text` should contain at least a `"text": content` field, since our retrieval logic expects each vector match to have the text in metadata). We also include any user-provided metadata like source or category.
  4. **Upsert to Neo4j** – Add the content to the graph store. There are a couple of strategies:
     - Use LightRAG’s `Neo4JStorage` methods if available. If `Neo4JStorage` has an `add_node` or similar, we could create a node with label (say `"Base"` or `"Memory"`), properties including the text and metadata, and possibly connect it to existing nodes if appropriate. In Nova, graph ingestion was done via `async_extract_and_ingest_graph_data` which uses an LLM to extract entities/relations from a conversation ([Nova_AI.py](file://file-D56GZQX5sXJqwrA1xivL2U#:~:text=async%20def%20async_extract_and_ingest_graph_data,into%20the%20Neo4j%20graph%20database)). For a simpler upsert of a knowledge snippet, we might not do an elaborate extraction. Instead, we can create a single node that holds this content.
     - If LightRAG doesn’t expose an easy insert API, we can use the Neo4j Python driver directly. For example, run a Cypher `MERGE` or `CREATE` query: 
       ```cypher
       MERGE (n:Base {id: $id})
       SET n.text = $content, n.metadata = $metadata
       ```
       This would insert the node if not present, or update it if it exists. (We’d need to ensure `$metadata` is handled properly — possibly store as a JSON string or separate properties).
     - We should also consider storing the embedding in the graph (some graph setups store a vector as a property so they can do vector similarity in-database). If LightRAG is configured with `embedding_func`, it might automatically store embeddings. In any case, adding the raw content to the graph at least allows graph traversal or linking later.
  5. **Response**: Return a confirmation, e.g. the `id` of the upserted memory and a status. A simple response could be `{"id": "<id>", "status": "upserted"}` or we could return the full item. For example, returning the stored item’s ID and perhaps the content or metadata echoed back.

- **Synchronization considerations**: Since we’re storing data in two places (vector DB and graph DB), we must ensure they remain consistent. Ideally, both succeed or both fail. In practice, we should handle errors gracefully:
  - If Pinecone upsert succeeds but Neo4j insert fails, we might choose to roll back the Pinecone insert (e.g. delete that vector) to avoid a memory piece that’s only half-stored. Conversely, if Neo4j succeeded and Pinecone failed, consider removing the graph node (or mark it). Implementing full two-phase commit is overkill, but we can at least log any inconsistency and possibly retry. Given this is a plan, we’ll note the importance of this error handling.
  - We could also perform these upserts sequentially and return an error if one fails, so the client knows to retry. For now, assume the operations usually succeed and focus on the happy path.

**Pseudocode for `/upsert`:**

```python
@router.post("/upsert")
async def upsert_memory(item: UpsertRequest):
    # 1. Determine ID
    content = item.content
    memory_id = item.id or hashlib.md5(content.encode()).hexdigest()
    # 2. Compute embedding
    embedding = memory_service.embed_text(content)
    # 3. Pinecone upsert
    pine_metadata = item.metadata or {}
    pine_metadata["text"] = content
    try:
        memory_service.pinecone_index.upsert([{"id": memory_id, "values": embedding, "metadata": pine_metadata}])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to upsert to vector store")
    # 4. Neo4j upsert
    try:
        memory_service.graph_store.add_node(memory_id, content, metadata=item.metadata)
    except Exception as e:
        # If graph fails, optionally remove from Pinecone to avoid inconsistency
        memory_service.pinecone_index.delete(ids=[memory_id])
        raise HTTPException(status_code=500, detail="Failed to upsert to graph store")
    # 5. Return success
    return {"id": memory_id, "status": "success"}
```

In practice, `graph_store.add_node` is a placeholder for however we add to Neo4j:
- If using LightRAG’s `Neo4JStorage`, we might not have a simple method, so we’d use the neo4j driver to run a Cypher query as described. That could be implemented inside `graph_client.py` for cleanliness.
- We include the content text in the vector metadata to ensure the retrieval pipeline can access it. (Nova’s hybrid merger expects a `'text'` in metadata of vector results ([hybrid_merger.py](file://file-TTgKasC3y2e4xkeCxpJYWu#:~:text=,Using%20ID%20as%20placeholder)).)

The `/upsert` endpoint lets us build the memory knowledge base over time. It’s designed for local development (you can run curl commands to feed data) and in production it could be used by an ingestion script or even by the AI system itself to store new info.

### **DELETE** `/memory/{id}` – Delete Memory Item

This endpoint allows removing a memory entry by its ID. This ensures we can manage the memory store (e.g., remove outdated or incorrect info).

- **Request**: The `{id}` path parameter identifies the memory item to delete. For example, a call to `DELETE /memory/memory_123` would remove the item with ID "memory_123".
- **Processing**:
  1. **Delete from Pinecone** – Call the Pinecone index’s delete method for that ID: `pinecone_index.delete(ids=[id])`. This will remove the vector and its metadata.
  2. **Delete from Neo4j** – Remove the corresponding node from the graph. For example, run a Cypher query like:
     ```cypher
     MATCH (n:Base {id: $id}) DETACH DELETE n;
     ```
     (Using DETACH DELETE to remove any relationships as well.) If using LightRAG’s storage class, see if it provides a deletion utility; otherwise use the neo4j driver.
  3. Both operations should be attempted. If one fails, log the error. It might be acceptable to return success if at least the primary store (Pinecone) succeeded, but ideally both succeed.
  4. **Response**: Return a simple status, e.g. `{"id": "<id>", "status": "deleted"}` or just 204 No Content. We’ll return a JSON for consistency.

- **Example**: `DELETE /memory/memory_123` -> returns `{"id": "memory_123", "status": "deleted"}` if found (or a 404 if the ID was not in memory).

**Pseudocode for `/memory/{id}`:**

```python
@router.delete("/memory/{id}")
async def delete_memory(id: str):
    # Delete from Pinecone
    try:
        memory_service.pinecone_index.delete(ids=[id])
    except Exception as e:
        logging.error(f"Error deleting {id} from Pinecone: {e}")
        # (We may still attempt graph deletion even if Pinecone fails)
    # Delete from Neo4j
    try:
        memory_service.graph_client.delete_node(id)
    except Exception as e:
        logging.error(f"Error deleting {id} from Neo4j: {e}")
    return {"id": id, "status": "deleted"}
```

We should handle the case where the ID doesn’t exist gracefully (Pinecone might simply no-op if ID not found; Neo4j match will delete nothing). Returning a 200 OK for deleting a non-existent item is usually fine (idempotent), or we could return 404 if we want the client to know.

By having this delete capability, the memory server can be maintained over time (especially important if this runs long-lived and memory might contain stale data).

### **GET** `/health` – Health Check

This is a lightweight endpoint to check if the server and its backing resources are operational. It will be used for monitoring or for a quick check (e.g., by Render or AWS load balancers to ensure the container is healthy).

- **Request**: No parameters (a simple GET).
- **Processing**: We will check the connectivity/status of the memory components:
  - Ping the Pinecone index – e.g., call something like `pinecone_index.describe_index_stats()` or a trivial query with top_k=0, just to see if the index responds.
  - Ping the Neo4j graph – e.g., run a cheap Cypher query like `RETURN 1` or use Neo4j driver’s session to ensure we can connect. If using `Neo4JStorage`, perhaps verify `graph_store._driver` is active or call a method if available (Nova’s code had a health check implicitly by trying an operation).
  - These checks should be done quickly. We might execute them in parallel (not strictly necessary, as this endpoint won’t be called often).
- **Response**: Return status info. For simplicity: `{"status": "ok"}` if all good. We could also detail each dependency, e.g. `{"pinecone": "ok", "neo4j": "ok"}` or include version info. If any check fails, return an error status (HTTP 500 or 503) with details.

**Pseudocode for `/health`:**

```python
@router.get("/health")
def health_check():
    ok = True
    details = {}
    # Check Pinecone
    try:
        memory_service.pinecone_index.describe_index_stats()
        details["pinecone"] = "ok"
    except Exception as e:
        ok = False
        details["pinecone"] = f"error: {str(e)}"
    # Check Neo4j
    try:
        memory_service.graph_client.ping()  # a method we'd implement to run a test query
        details["neo4j"] = "ok"
    except Exception as e:
        ok = False
        details["neo4j"] = f"error: {str(e)}"
    # Overall
    if ok:
        return {"status": "ok", **details}
    else:
        raise HTTPException(status_code=503, detail=details)
```

This health check can be expanded or integrated with FastAPI’s event system (e.g., perhaps do a more thorough check on startup and cache the result). But the above is enough to ensure we know if connections are alive.

In summary, these routes cover all required operations. The **MemoryService** class that underpins them will encapsulate the Nova AI memory logic, making the route functions themselves simple (just parsing input and returning output). This separation also eases unit testing (you can call MemoryService methods directly without making HTTP calls).

## Configuration for Pinecone, Neo4j, and Embeddings

Proper configuration is crucial for the server to locate resources and credentials. We’ll use a flexible approach so that local development and cloud deployment can supply config in environment variables or a `.env` file:

- **Pinecone Configuration**:
  - **API Key**: Expect `PINECONE_API_KEY` in env or config.
  - **Environment**: Expect `PINECONE_ENV` (like "us-west1-gcp").
  - **Index Name**: We can define an index name like `PINECONE_INDEX` (Nova’s code used `"nova-ai-memory"` as index name ([Nova_AI.py](file://file-D56GZQX5sXJqwrA1xivL2U#:~:text=,%29%20index%20%3D%20pc.Index%28index_name))). We should use the same if we want to attach to existing data; otherwise, we can configure it.
  - On startup, initialize Pinecone:
    ```python
    import pinecone
    pinecone.init(api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV)
    index = pinecone.Index(config.PINECONE_INDEX)
    ```
    We should ensure the index exists. Nova’s code created it if not present (dimension 1536 for OpenAI embeddings, metric cosine) ([Nova_AI.py](file://file-D56GZQX5sXJqwrA1xivL2U#:~:text=,%29%20index%20%3D%20pc.Index%28index_name)). We can do similar, or assume it’s created out-of-band. For safety, our service could attempt to create if not exists (especially in local dev scenario).
  - The Pinecone index object will be stored in `MemoryService.pinecone_index` for use in queries and upserts.

- **Neo4j (LightRAG) Configuration**:
  - **URI**: e.g. `NEO4J_URI` (bolt URL, e.g. `bolt://localhost:7687` if local).
  - **User/Pass**: `NEO4J_USER` and `NEO4J_PASSWORD` for authentication. These can be in .env for local (and set as secrets in cloud).
  - **Database**: If using Aura or multi-database, `NEO4J_DATABASE` (or we ensure using the default).
  - **LightRAG Setup**: We instantiate `Neo4JStorage` with a config. Possibly we need a `global_config` dict or object. If LightRAG expects a certain format, we’ll construct it. For example:
    ```python
    neo4j_config = {
        "uri": config.NEO4J_URI,
        "username": config.NEO4J_USER,
        "password": config.NEO4J_PASSWORD,
        "database": config.NEO4J_DATABASE or "neo4j"
    }
    graph_store = Neo4JStorage(namespace="nova_memory", global_config=neo4j_config, embedding_func=embedding_function)
    ```
    The `namespace` might be used to isolate knowledge; Nova possibly used something like a specific namespace to separate different types of data.
  - Ensure the Neo4j server is running and accessible. For local dev, this might be a Neo4j Docker container (our `docker-compose.yml` sets one up on port 7687 with credentials). For cloud, it could be Neo4j AuraDB or an EC2-hosted instance.
  - **Schema**: We should clarify if any Neo4j constraints or indexes need to be set (e.g., an index on node `id` or on vector property if doing vector similarity). In local dev, we can connect via Neo4j Browser to set those up if needed.
  - The LightRAG storage will handle graph queries. We may configure it with the same embedding model, so it knows how to vectorize node content for similarity. That’s why we pass `embedding_func`. If our embedding model is an API call (like OpenAI), LightRAG might use it too, so be mindful of latency. (Alternatively, LightRAG might not do real-time embedding for queries if it stored all node vectors and can do approximate search. We’ll find out in practice.)

- **Embedding Model for Vectors**:
  - Nova AI likely uses OpenAI’s text-embedding-ada-002 (1536-dim) for Pinecone (given the 1536 dimension). We should continue with the same model for consistency. That means we need an OpenAI API key configured (`OPENAI_API_KEY`).
  - Alternatively, if we want to avoid external calls for embedding (especially for local dev), we could use a local embedding model via SentenceTransformers. But since Nova integrated OpenAI, we’ll plan to use that by default. The `.env` can contain OPENAI_API_KEY which `openai` library will pick up.
  - **Embedding function**: We will implement an `embed_text(text: str) -> List[float]` function in our service. This can call `openai.Embedding.create(...)` or use a cached approach. (Nova had an `get_embedding` function and an LRU cache for embeddings to avoid duplicate calls).
  - For upserting, we use this embedding function. For querying, we use it on the query string as well (unless Pinecone supports querying by raw text via their own internal embedding, but here we want control to ensure same model usage).
  - We should document the model choice and ensure the environment has the needed API access. In local dev, if no OpenAI key, an alternative is needed (maybe fallback to a local model). However, given the target user likely has it configured, we can assume it’s available.
  
- **Other Nova Modules Configuration**:
  - The cross-encoder reranker will download a model (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default) on first load. We might allow configuring `RERANKER_MODEL_NAME` in config if we want to switch models or disable reranker. For now, using the default is fine. The class will automatically use GPU if available or CPU otherwise.
  - If running on a system without GPU, ensure it’s not too slow – the MiniLM cross-encoder is reasonably small, so CPU should be okay for moderate usage.
  - LightRAG/Neo4j might have some config like what node label or relationship types to use. Possibly we ensure that `async_extract_and_ingest_graph_data` in Nova inserts nodes with label "Base". We can decide to use the same label in our upsert. It might be wise to define in config something like `NEO4J_NODE_LABEL = "Base"` (or "Memory") so both the ingestion and retrieval use the same label. In Nova’s retrieval, they used node_label="base" (lowercase) ([Nova_AI.py](file://file-D56GZQX5sXJqwrA1xivL2U#:~:text=,used%20in%20async_extract_and_ingest_graph_data%20graph_task)). We will stick to "base" unless decided otherwise.

- **Loading Config**: In `app/config.py`, we can use Pydantic’s `BaseSettings` for convenience:
  ```python
  from pydantic import BaseSettings
  class Settings(BaseSettings):
      PINECONE_API_KEY: str
      PINECONE_ENV: str
      PINECONE_INDEX: str = "nova-ai-memory"
      NEO4J_URI: str
      NEO4J_USER: str = "neo4j"
      NEO4J_PASSWORD: str
      OPENAI_API_KEY: str
      # etc.
      class Config:
          env_file = ".env"
  settings = Settings()
  ```
  This way, environment variables are automatically read, and for local dev we can have a `.env` file with all keys (which is loaded by BaseSettings). Nova’s code already uses `python-dotenv` to load .env (they call `load_dotenv()` ([Nova_AI.py](file://file-D56GZQX5sXJqwrA1xivL2U#:~:text=,PINECONE_ENV))), so we mirror that behavior. 

- After loading settings, we pass them to our service initialization (e.g., `MemoryService(config=settings)`). The service can then do all the setup for Pinecone and Neo4j as described.

By externalizing these configurations, our code does not hardcode any credential or environment info, making it secure and adaptable. For instance, in a cloud deployment on Render, we’d set these env vars in the dashboard. On AWS ECS, we’d use Secrets Manager or task definitions to provide them. Locally, we edit the .env file.

**Summary of config values**:
- `OPENAI_API_KEY` – for embeddings (and possibly other model calls).
- `PINECONE_API_KEY` and `PINECONE_ENV` – for vector DB access.
- `PINECONE_INDEX` – index name (create if not exists).
- `NEO4J_URI` – likely `bolt://neo4j:7687` in Docker, or a Aura bolt+ssl URL, etc.
- `NEO4J_USER`, `NEO4J_PASSWORD` – credentials.
- (Optional) `NEO4J_DATABASE` – if not default.
- (Optional) `RERANKER_MODEL_NAME` – if we want to allow changing the cross-encoder.
- (Optional) `EMBEDDING_MODEL` – if in future we switch from OpenAI to something else.

All these can be documented in README and sample .env.

## Example Usage of the REST API

Once the server is running (e.g., via Uvicorn on `http://localhost:8000`), users or client applications can interact with it. Below are some example calls using **curl** and **httpx** (a Python HTTP client) to demonstrate each endpoint:

- **Query Memory Example** (retrieve relevant info for a question):

  Using curl:
  ```bash
  curl -X POST "http://localhost:8000/query" \
       -H "Content-Type: application/json" \
       -d '{"query": "What is the relationship between AI and machine learning?"}'
  ```
  This will return a JSON object containing a list of results. For example:
  ```json
  {
    "results": [
      {
        "id": "d1eab8f...1c", 
        "text": "Machine learning is a subset of AI that focuses on algorithms learning from data.", 
        "source": "vector", 
        "score": 0.95
      },
      {
        "id": "node_42", 
        "text": "AI (Artificial Intelligence) encompasses machine learning as one of its approaches.", 
        "source": "graph", 
        "score": 0.93
      },
      ...
    ]
  }
  ```
  (The exact content and scores depend on what’s in the memory stores. The example shows a vector-sourced snippet and a graph-sourced fact, merged by relevance.)

  Using Python httpx:
  ```python
  import httpx, json
  resp = httpx.post("http://localhost:8000/query", json={"query": "What is the relationship between AI and machine learning?"})
  data = resp.json()
  print(json.dumps(data, indent=2))
  ```
  This would output the same JSON structure as above.

- **Upsert Memory Example** (add a new memory record):

  Suppose we want to add a piece of info about “Graph databases”.
  ```bash
  curl -X POST "http://localhost:8000/upsert" \
       -H "Content-Type: application/json" \
       -d '{
             "id": "graphdb_info",
             "content": "Graph databases store data in nodes and relationships, which is useful for representing networks.",
             "metadata": {"source": "tech_blog", "topic": "databases"}
           }'
  ```
  This requests the server to store the content with a specified ID. A successful response might be:
  ```json
  { "id": "graphdb_info", "status": "success" }
  ```
  Now, if we query something related, e.g. “How is data stored in graph databases?”, the memory we just upserted should be among the results.

  *Note:* If `id` was not provided, the response would return the generated ID (likely a hash). The client should capture that if it needs to delete or reference it later.

- **Delete Memory Example**:

  To remove the memory we just added:
  ```bash
  curl -X DELETE "http://localhost:8000/memory/graphdb_info"
  ```
  Response:
  ```json
  { "id": "graphdb_info", "status": "deleted" }
  ```
  After this, a query that would have matched that content should no longer retrieve it from Pinecone or Neo4j.

- **Health Check Example**:

  Simply:
  ```bash
  curl "http://localhost:8000/health"
  ```
  Expected response if healthy:
  ```json
  { "status": "ok", "pinecone": "ok", "neo4j": "ok" }
  ```
  If, say, Neo4j is down, you might get an HTTP 503 with a payload:
  ```json
  { "detail": { "pinecone": "ok", "neo4j": "error: ... connection refused ..." } }
  ```
  This signals an issue with one component.

These examples assume the server is running locally. Adjust the host/port if deployed remotely. Also, ensure the content type is application/json for POST requests. 

During local testing, you can use tools like **HTTPie** or **Postman** similarly to send requests.

## Development and Deployment Considerations

**Local Development:** The initial design is for local use, so make it easy to run the MCP server on a developer’s machine or a local Docker setup.

- **Running locally (without Docker):** After writing the code, one can simply do:
  ```bash
  uvicorn app.main:app --reload --port 8000
  ```
  This will start the FastAPI server with auto-reload on code changes. Ensure you have a Neo4j database running and the environment variables set. For instance, using the provided `docker-compose.yml`, you can start a Neo4j container:
  ```bash
  docker-compose up -d neo4j
  ```
  This will start Neo4j at bolt://localhost:7687 (with credentials from .env). Pinecone is cloud-based, so no container needed; just ensure your API key is in .env. With Neo4j up and the .env loaded, the server can be started and should connect to both Pinecone and Neo4j.

- **Docker for local dev:** We supply a `Dockerfile` to containerize the application. This Dockerfile might look like:
  ```Dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY app ./app
  COPY .env .env  # if we want to bake in local config (or better, supply at runtime)
  EXPOSE 8000
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
  We can then use `docker-compose.yml` to run both the app and Neo4j together for convenience. The provided compose file maps Neo4j’s ports and waits for it to be healthy before starting the app service ([docker-compose.yml](file://file-G8zDsxe4p3Y41KVD1JWRu5#:~:text=env_file%3A%20,Mount%20logs%20directory)). This is helpful because the app might try connecting to Neo4j on startup.

  To build and run via compose:
  ```bash
  docker-compose up --build
  ```
  This will start `nova-ai` service (our MCP server) and `neo4j` service. The FastAPI app will be reachable at `localhost:8000` and connected to the Neo4j container.

- **Hot-reload in Docker:** For development, you might mount the code directory and use `--reload` in Uvicorn so that changes reflect without rebuilding the container. That can be set up in docker-compose by mounting the code and adjusting the command, but it’s optional (many prefer running natively for dev and using Docker mainly for integration testing or deployment).

**Future Cloud Deployment:** The clean architecture and config will facilitate moving this to cloud. Some suggestions for deployment:

- **Render.com or Heroku:** These platforms can deploy a web service directly from the GitHub repo. Render, for example, will detect a FastAPI service (or you can specify using the Dockerfile). You would add the necessary environment variables (Pinecone keys, Neo4j URI, etc.) in the Render dashboard. One consideration: for Neo4j, you might not run it on Render (since it’s a database). Instead, you could use Neo4j Aura (a hosted Neo4j) or run Neo4j on a separate service. The MCP server will just connect to whatever `NEO4J_URI` you provide. So in cloud, you’d point it to a cloud Neo4j instance or perhaps a Neo4j Docker on the same host if using something like a single VM.
- **AWS (ECS/Fargate or EC2):** You can build a Docker image of the server and push to ECR. Then run it in ECS or as a simple container on an EC2. Ensure security groups allow it to reach Pinecone (which is SaaS over HTTPS) and your Neo4j (which might be an EC2 or a managed DB). Storing secrets: on AWS, use ECS secrets or environment variables configured in the task definition for keys.
- **Kubernetes:** If going that route, the Docker image can be deployed in a pod. Use ConfigMaps/Secrets for env vars. Possibly also deploy a Neo4j pod or use a separate managed service. The architecture being stateless (the server has no local state, all memory is in Pinecone/Neo4j) means you can scale the MCP server horizontally if needed – multiple replicas can all connect to the same Pinecone index and Neo4j database. Just ensure only one instance of Neo4j if using embedded graph; but if using an external Neo4j, that’s already handled.

- **Resource considerations:** 
  - The Pinecone usage is network-bound (API calls), so the server needs internet access and the Pinecone environment should correspond to a region near the server for latency.
  - The Neo4j part might require low latency between the app and the DB; deploying them in the same cloud region or VPC is wise for production.
  - The reranker model will consume some memory and CPU (or GPU). On a cloud instance, ensure it has enough RAM to load the cross-encoder model (~100MB) and that the CPU is sufficient for inference speed. If high query throughput is expected and reranking becomes a bottleneck, one could disable it or use a smaller model or scale out the service.

- **Logging and Monitoring:** Since this will be a service, we should implement proper logging (FastAPI’s logging plus any custom logs for each query pipeline step). In local dev, console logs are fine; in production, route them to a file or monitoring system. The health endpoint can be used by uptime monitors. We might also add an endpoint or log for performance metrics (like how long retrieval takes, etc.) if needed.

- **Docker image for Neo4j:** For production, you might not use Docker Compose. Instead, you may use a managed Neo4j or run a separate container. The compose file we provided is mostly for local convenience. In cloud, you’ll likely handle the database separately.

- **Security:** If deploying beyond local, consider adding auth to the MCP server (e.g., an API key or authentication mechanism) since it exposes potentially sensitive memory data. FastAPI could integrate with an auth dependency if needed. For now, as an internal tool in dev, it’s open.

By following this plan, we create a robust MCP server that **mirrors Nova AI’s memory system**. It will be easy to maintain (since it reuses existing, tested components) and easy to evolve (clean structure ready for improvements). In local development, a developer can run the server, test queries/upserts, and iterate. When it’s time to deploy to the cloud, minimal changes are needed – mostly just providing the correct config and ensuring external services (Pinecone, Neo4j) are accessible. The result is a modular “memory microservice” adhering to the Model Context Protocol, which can serve as a backend for Nova AI or any AI assistant needing contextual memory retrieval.