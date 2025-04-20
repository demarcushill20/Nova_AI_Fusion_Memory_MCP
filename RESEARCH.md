Awesome. I’ll put together a detailed technical plan for building an MCP (Model Context Protocol) server that wraps around Nova AI’s fusion memory system, specifically designed for Python agents to interact with it.

This will cover everything from architecture, endpoints, data flow, to authentication and how to integrate Pinecone and LightRAG. I’ll let you know once the full plan is ready!

# MCP Memory Server Design

## High-Level Architecture Overview

The **Model Context Protocol (MCP) Memory Server** acts as a centralized memory API that wraps around the Nova AI fusion memory system. In essence, it provides a unified interface for AI agents to **store and retrieve shared context**. The server is built with FastAPI and integrates two core backend components: **Pinecone** for vector-based semantic memory and **LightRAG** (with a Neo4j graph) for knowledge graph memory.

- **Pinecone Vector Store**: Pinecone is a cloud-native vector database optimized for similarity search on high-dimensional embeddings ([Using Pinecone Vector Database: A beginner guide | by M K Pavan Kumar |  . | Medium](https://medium.com/aimonks/using-pinecone-vector-database-a-beginner-guide-6f81dc827874#:~:text=Pinecone%20is%20not%20a%20traditional,dimensional%20vector%20data)). It stores vector representations of text (e.g. documents, facts, conversation history) and supports fast nearest-neighbor queries. The MCP server will use Pinecone’s Python client to upsert and query embedding vectors.
- **LightRAG Knowledge Graph**: LightRAG is an open-source retrieval-augmented generation system that builds a lightweight knowledge graph from text. Under the hood, it extracts entities and relationships from documents to create a graph of knowledge ([GraphRAG and LightRAG](https://www.linkedin.com/pulse/lightrag-graphrag-new-area-rag-applications-narges-rezaei-skmhe#:~:text=How%20LightRAG%20Works)). In our architecture, LightRAG is configured to use **Neo4j** as the graph database (via its `Neo4JStorage` backend ([GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG#:~:text=match%20at%20L1065%20graph_storage%3D%22Neo4JStorage%22%2C%20%23%3C,KG%20default))). This graph stores nodes for key entities/concepts and edges for relationships, enabling structured queries and retrieval of connected information.
- **MCP Server (FastAPI)**: Sits between AI agents and the memory backends. It exposes HTTP endpoints (e.g. `/memory/query`, `/memory/upsert`, `/memory/delete`) that agents call to read or write memories. The server orchestrates calls to Pinecone and the LightRAG/Neo4j modules, applying **memory fusion logic** to combine results. The FastAPI app can run locally (e.g. on `localhost:8000`) and serve multiple local agents concurrently.
- **Embedding Model**: To interface with Pinecone, the server needs to generate embedding vectors for text inputs. This could be a pre-configured embedding model (e.g. OpenAI text-embedding API or a local model) that the server uses whenever a new text needs indexing or a query vector. The embedding step happens within the server before calling Pinecone since Pinecone stores vectors but does not generate them.

In operation, an AI agent (MCP client) will connect to the MCP server over HTTP. When the agent sends a memory query or upsert request, the server interacts with Pinecone and Neo4j as needed and returns a unified result. This design follows the spirit of Anthropic’s MCP concept – **exposing data through an MCP server so that AI tools can access it via a standard interface** ([Introducing the Model Context Protocol \ Anthropic](https://www.anthropic.com/news/model-context-protocol#:~:text=The%20Model%20Context%20Protocol%20is,that%20connect%20to%20these%20servers)). The high-level data flow is:

1. **Agent -> MCP Server**: Agent makes an HTTP request to a memory endpoint.
2. **MCP Server -> Pinecone/Neo4j**: Server translates the request into operations on Pinecone (vector search or update) and on the LightRAG/Neo4j graph (graph queries or updates).
3. **Fusion and Response**: The server combines vector results and graph results into a coherent response and sends it back to the agent.

This architecture centralizes memory management. All agents share the same Pinecone index and Neo4j knowledge graph via the MCP server, ensuring **consistent context** across agents (one agent’s learning can immediately be accessed by others). It also decouples agents from direct database calls – agents only need to know how to call the MCP API, not the details of Pinecone or Neo4j connections. 

## FastAPI Endpoints for Memory Operations

We will design a set of HTTP endpoints (using FastAPI) to handle core memory interactions: inserting/updating knowledge, querying for relevant context, and deleting outdated memory. Each endpoint will accept/return JSON and be documented via the automatic FastAPI docs (Swagger UI). Below are the primary endpoints:

- **`POST /memory/upsert`** – Add a new memory entry or update an existing one. This will handle inserting text (and associated data) into Pinecone with an embedding, and updating the knowledge graph with any new entities/relations extracted.
- **`POST /memory/query`** – Query the memory for relevant information. The request contains a query (text or other identifier), and the server will return a combination of vector-similar documents and graph-based context that match the query.
- **`DELETE /memory/{id}`** – Remove a memory entry by its ID. This will delete the vector from Pinecone and also remove or mark obsolete the corresponding nodes/edges in the graph.
- **`GET /memory/{id}`** – *(Optional)* Fetch a specific memory entry by ID (including its stored content or metadata). This could be useful for verifying what's stored or retrieving a full memory record.
- **`GET /health`** – *(Optional but recommended)* A simple health check endpoint to verify the server is running (returns "OK" or version info). Useful for monitoring.

These endpoints use a consistent base path (`/memory`) to clearly namespace memory operations. The API can be versioned (e.g., `/api/v1/memory/...`) if needed for future changes. Each operation corresponds to a method in the underlying memory service logic:

- **Upsert**: calls a service function to embed the input text, upsert the vector in Pinecone, and update the graph via LightRAG (e.g., create nodes for new entities).
- **Query**: calls a service function to perform the hybrid search (vector similarity + graph traversal) and fuse results.
- **Delete**: calls a service function to delete by ID in both Pinecone index and the graph store.
- **Get**: calls a service to fetch the item (possibly from a cache or directly from Pinecone/graph).

Using FastAPI, these endpoints will be asynchronous (`async def`) to handle I/O-bound operations (network calls to Pinecone and Neo4j). We will define **Pydantic models** for request and response payloads to enforce schema and for automatic documentation. For example, a `MemoryQueryRequest` model might define fields like `query: str` and `top_k: int`, and a `MemoryQueryResponse` model could define a list of result items.

## Memory Fusion Logic (Combining Vector and Graph Data)

One of the key design considerations is **how to combine vector-based and graph-based data to answer a query**. The goal of fusion is to leverage the strengths of both systems: Pinecone’s semantic similarity can find relevant pieces of unstructured data, while the knowledge graph can capture relationships and context between entities that may span multiple documents ([GraphRAG and LightRAG](https://www.linkedin.com/pulse/lightrag-graphrag-new-area-rag-applications-narges-rezaei-skmhe#:~:text=This%20system%20comes%20with%20its,resulting%20in%20poor%20retrieval%20quality)). We outline the fusion approach for query processing:

1. **Query Vector Search**: When a query comes in (e.g. a user question or a key phrase), the server first uses the embedding model to encode the query into a vector. It then performs a similarity search in Pinecone for the top *k* most relevant vector embeddings (e.g. top 5 or 10 results). This yields a set of memory entries (documents, notes, etc.) that are semantically similar to the query. Each result might include an `id`, a relevance `score`, and stored `metadata` (like original text snippet, source, etc.).
2. **Knowledge Graph Search**: In parallel (or subsequently), the server utilizes the knowledge graph for structured insight:
   - It may perform an **entity lookup**: If the query contains a name or term that matches a graph node (for example, an entity like “Einstein” or a concept like “quantum physics”), the server queries Neo4j (via LightRAG) to retrieve that node and its immediate relationships. For instance, it can fetch all nodes directly connected to the entity node (e.g. attributes of Einstein, related entities like “Theory of Relativity”, etc.).
   - It may also do a **graph traversal query**: LightRAG can use dual-level retrieval, meaning it might extract keywords from the query (both local and global context) ([GraphRAG and LightRAG](https://www.linkedin.com/pulse/lightrag-graphrag-new-area-rag-applications-narges-rezaei-skmhe#:~:text=LightRAG%20extracts%20both%20local%20and,keywords%20using%20a%20vector%20similarity)). Based on those keywords, the server can run a Cypher query in Neo4j to find relevant subgraphs. For example, if the query is a complex question, the server could ask Neo4j for a path or a set of connections between two entities mentioned in the query.
   - The graph query results provide **structured knowledge** – e.g. a set of related facts or identifiers that might not be explicitly mentioned in any single document but are stored as relationships.
3. **Fusion of Results**: The server then combines the results from Pinecone and Neo4j:
   - It can **merge results by reference**: Many vector hits might correspond to documents that contain certain entities. If an entity from the graph appears in a top Pinecone result, we can enrich that result with additional info from the graph (like linking the document result with related entities or a summary of their relationships).
   - It can also **unify the answer set**: The response could include distinct sections for vector-based matches and graph-based findings. For example, vector search might return a document that directly answers a question, while the graph might return a fact triple that also answers it. Both are useful. The server might rank or filter these to avoid duplication.
   - In cases where one modality clearly addresses the query (e.g. a direct fact is in the graph), the fusion logic might prioritize that, but generally the system will return a **comprehensive bundle of context**.
   
A concrete example of this fusion: suppose the query is *"What is the capital of the country where Albert Einstein was born?"*. The server would:
   - Embed the query and find top vector matches (maybe a biography of Einstein, a document about Germany or Ulm, etc.).
   - From the graph, identify the node for "Albert Einstein", find a relationship like (*Albert Einstein* –[born_in]→ *Ulm*, and *Ulm* –[located_in]→ *Germany* –[has_capital]→ *Berlin*).
   - The vector results might give a passage mentioning Einstein’s birthplace and Germany. The graph results can explicitly provide the chain that Germany’s capital is Berlin.
   - The fusion could return both the relevant text snippet and the explicit fact from the graph, which together give a full answer.

To implement this logic, the **MemoryService** component in the server will coordinate between the Pinecone client and a Neo4j query interface (possibly using the official Neo4j Python driver or LightRAG’s API). The retrieval might be done sequentially or concurrently (FastAPI can use `asyncio.gather` to perform Pinecone query and Neo4j query in parallel for efficiency). The combination step may involve simple merging or more complex ranking. For the initial design, a **simple union** of results with an indication of source is effective, letting the agent decide how to use them. 

It’s worth noting that this fusion approach aligns with emerging patterns in advanced RAG systems. By using semantic search to narrow down relevant information and then leveraging graph queries for precise facts or connections, we get the best of both worlds ([Vectors and Graphs: Better Together | Pinecone](https://www.pinecone.io/learn/vectors-and-graphs-better-together/#:~:text=focused%20on%20retrieval%2C%20we%20wanted,done%20either%20manually%20or%20agentically)). The system can reason over both unstructured and structured data: Pinecone provides similarity-based recall, and the knowledge graph ensures no important relationship is missed due to vector space limitations ([GraphRAG and LightRAG](https://www.linkedin.com/pulse/lightrag-graphrag-new-area-rag-applications-narges-rezaei-skmhe#:~:text=Query,a%20broader%20view%20of%20information)). 

*Upsert Fusion:* A brief note on how upserts are handled with both systems – when new data is inserted, the server will not only add the embedding to Pinecone but also update the graph. For example, if a new document is added via `/memory/upsert`, the server could run LightRAG’s parsing on it to extract entities/relations, and merge those into Neo4j (creating new nodes or edges as needed). LightRAG is designed to allow **incremental updates** to the graph without full rebuilds ([GraphRAG and LightRAG](https://www.linkedin.com/pulse/lightrag-graphrag-new-area-rag-applications-narges-rezaei-skmhe#:~:text=match%20at%20L159%20The%20other,need%20to%20rebuilding%20it%20entirely)), which fits our use case of continuously growing memory. This ensures that subsequent queries can immediately leverage the new information in both vector and graph forms.

## API Input/Output Formats

Each endpoint will consume and produce JSON data. We define the format for requests and responses to ensure consistency for all client agents. Below are the input/output schemas for the main endpoints:

- **`POST /memory/upsert`**  
  **Request:** JSON object containing the memory data to add/update. For example:  
  ```json
  {
    "id": "string (optional)", 
    "text": "the content to store", 
    "metadata": { "tags": ["example"], "source": "agent1" }
  }
  ```  
  Fields:
  - `id` (optional string): A unique identifier for the memory. If provided and existing, the entry is updated; if not provided, the server generates an ID (e.g. a UUID).
  - `text` (string): The content or knowledge to store. This could be a chunk of text, a summary, conversation snippet, etc. The server will embed this text for vector storage.
  - `metadata` (object, optional): Any additional metadata to store alongside, such as tags, source info, timestamps. This will be stored in Pinecone (as metadata) and possibly partly in the graph node.
  
  **Response:** JSON confirming the upsert. For example:  
  ```json
  {
    "id": "generated-id-12345",
    "status": "upserted",
    "vector_entries": 1,
    "graph_entries": 3
  }
  ```  
  Fields:
  - `id`: The unique ID of the memory entry that was stored (same as provided or newly generated).
  - `status`: Confirmation message (`"upserted"` or `"updated"`).
  - `vector_entries`: Number of vector records affected (usually 1).
  - `graph_entries`: Number of graph nodes/relationships created or updated as a result (for informational purposes).
  
- **`POST /memory/query`**  
  **Request:** JSON with query parameters. For example:  
  ```json
  {
    "query": "text of the query or question",
    "top_k": 5
  }
  ```  
  Fields:
  - `query` (string): The query text that the agent is searching for in memory. This can be a natural language question or keywords.
  - `top_k` (integer, optional): The number of top results to return from vector search (defaults to, say, 5). The graph search is not limited by `top_k` in the same way (it will return relevant nodes/relations).
  
  **Response:** JSON with combined results. For example:  
  ```json
  {
    "query": "original query text",
    "vector_results": [
      {
        "id": "entry-id-1",
        "score": 0.87,
        "text": "excerpt of stored text matching the query...",
        "metadata": { "source": "agent1", "tags": ["example"] }
      },
      ...
    ],
    "graph_results": [
      {
        "entity": "EntityName",
        "properties": { "type": "Person", "description": "..." },
        "neighbors": [
          { "relation": "born_in", "entity": "LocationNode" },
          { "relation": "related_to", "entity": "OtherEntity" }
        ]
      },
      ...
    ]
  }
  ```  
  Here, we separate `vector_results` and `graph_results` for clarity:
  - `vector_results` is a list of up to `top_k` items from Pinecone, sorted by similarity score. Each item includes the memory `id`, the similarity `score`, the stored `text` (or a snippet/summary of it), and the `metadata` it was stored with.
  - `graph_results` is a list of relevant graph nodes or facts. This could include key entities related to the query. In this format, each result might include an `entity` name (or ID), its `properties` (node attributes like type, description), and its `neighbors` (a list of connected nodes with the relationship type). Essentially, this provides a subgraph centered around relevant entities.
  
  The exact structure of `graph_results` can be adjusted depending on needs; for example, it could also be a list of relationship triples or a serialized subgraph. The above structure is one human-readable way (entity with its immediate relations).
  
  By returning both lists, the agent has the flexibility to use the information as needed – it can read the text from `vector_results` for detailed context, and use `graph_results` for quick fact lookup or logical reasoning. In some cases, there may be overlap (e.g. a vector result’s text might mention an entity that is also in `graph_results`), which is expected.
  
- **`DELETE /memory/{id}`**  
  **Request:** This is a DELETE request on a specific memory ID. The ID can be included in the URL path. No request body is needed (the ID in the path identifies what to delete).
  
  Example: `DELETE /memory/entry-id-12345`
  
  **Response:** JSON confirmation of deletion. For example:  
  ```json
  {
    "id": "entry-id-12345",
    "status": "deleted",
    "details": {
      "vector_deleted": true,
      "graph_deleted": 3
    }
  }
  ```  
  Fields:
  - `id`: The ID of the memory that was deleted.
  - `status`: `"deleted"` if successful (or an error message if not found).
  - `details`: Additional info – e.g. `vector_deleted: true/false` to indicate if a vector entry was removed, and `graph_deleted` indicating how many graph nodes or relationships were removed (if the knowledge graph entry for that memory was cleaned up). In some designs, we might not fully delete nodes that could be connected to other data; we might just remove a link or mark it. This field can reflect what was done.
  
- **`GET /memory/{id}`** (optional retrieval endpoint)  
  **Response:** If implemented, this would return the stored content and metadata for the given ID, possibly like:  
  ```json
  {
    "id": "entry-id-1",
    "text": "full text content of the memory entry",
    "metadata": { ... },
    "graph_context": { ... }
  }
  ```  
  Where `graph_context` might include any graph info directly linked to this entry (for instance, entities that were extracted from this text).

All responses should use standard HTTP status codes (200 for success, 404 if an ID is not found on GET/DELETE, etc.). Errors or exceptions (like Pinecone timeouts or Neo4j errors) will be caught and returned as JSON error messages with appropriate status codes (FastAPI makes it easy to raise HTTPException for this). 

The input/output formats are designed to be **JSON serializable** (using Pydantic ensures that). This way, any Python agent can easily decode the JSON into Python dictionaries or data classes. The format favors clarity and completeness: an agent could log the entire response for debugging or analysis of what the memory server returned.

## Example: Python Agent Using the MCP Server

To illustrate how a Python-based AI agent might interact with the MCP memory server, let's walk through a basic usage scenario. In this example, the agent will store a piece of information and then query it.

**Setting up the agent client:** The agent can use Python’s `requests` library (or `httpx`, etc.) to make HTTP calls to the MCP server endpoints.

```python
import requests
import json

# Base URL of the MCP server
MCP_BASE = "http://localhost:8000/memory"

# 1. Upsert a new memory entry (e.g., the agent learned a new fact or read a document)
memory_data = {
    "id": "einstein_note",  # optional custom ID
    "text": "Albert Einstein was born in Ulm, Germany in 1879.",
    "metadata": {"source": "wiki", "tags": ["person", "birthplace"]}
}
response = requests.post(f"{MCP_BASE}/upsert", json=memory_data)
print(response.status_code, response.json())
# Expected output: 200 {"id": "einstein_note", "status": "upserted", "vector_entries": 1, "graph_entries": 2}
```

In the above, the agent adds a fact about Albert Einstein. The MCP server will embed this sentence and store it in Pinecone, and also update the graph (nodes for "Albert Einstein" and "Ulm, Germany" with a relationship "born_in"). The response indicates success and how many graph entries were added (e.g. two nodes and their relation).

```python
# 2. Query the memory for a related question
query_payload = {"query": "Where was Albert Einstein born?", "top_k": 3}
response = requests.post(f"{MCP_BASE}/query", json=query_payload)
result = response.json()

# The agent can now use the returned information:
print(result["vector_results"][0]["text"])
# e.g., "Albert Einstein was born in Ulm, Germany in 1879."

# If the agent wants to use graph results:
for entity in result.get("graph_results", []):
    print(entity["entity"], "-> connected to ->",
          [nbr["entity"] for nbr in entity.get("neighbors", [])])
# e.g., "Albert Einstein -> connected to -> ['Ulm', 'Germany']"
```

The query results provide the agent with both the text snippet (which directly answers the question) and structured data: the graph might show the connection between Einstein and the location Ulm/Germany. The agent can decide how to incorporate this into its reasoning or responses. For instance, an LLM-based agent might include the text snippet in a prompt to answer the user, and use the graph info to verify facts or enhance the answer (ensuring it knows that Ulm is in Germany, etc.).

```python
# 3. Delete memory if not needed (cleanup example)
del_response = requests.delete(f"{MCP_BASE}/einstein_note")
print(del_response.status_code, del_response.json())
# Expected output: 200 {"id": "einstein_note", "status": "deleted", "details": {"vector_deleted": true, "graph_deleted": 3}}
```

This would remove the Einstein note from the shared memory, so future queries about Einstein would not return this particular entry (unless there were other entries about him).

In a real deployment, agents might wrap these calls in convenience functions or a client library. For example, a small Python class `MCPClient` could provide methods like `upsert_memory(text, id=None)` and handle the HTTP calls internally. This would make the agent code cleaner (`mcp_client.upsert_memory("Einstein...", id="einstein_note")`). Agents could also use asynchronous calls if using an async HTTP client and if the FastAPI server is asynchronous (which it is), to maximize throughput.

The above example demonstrates basic CRUD operations on the memory. In practice, agents will likely query the memory before formulating answers (to retrieve context) and upsert new knowledge after processing new information or concluding a task. The centralized MCP server makes these steps uniform for all agents in the system.

## Optional Enhancements and Considerations

Beyond the core functionality, several enhancements and best-practice features are recommended to make the MCP memory server more robust and secure:

- **Authentication & Authorization**: In a multi-agent or multi-client setting, it may be important to secure the API. We could implement API key authentication (e.g., requiring a header like `Authorization: Bearer <token>` for each request). FastAPI can integrate with dependency-based auth or OAuth2 if needed. Simple shared-secret token auth might suffice for local usage. This ensures only authorized processes can read/write the global memory.
- **Logging and Monitoring**: Incorporate logging for each request and important internal events. For example, log every query and upsert (possibly with truncated text for privacy) and whether the operation succeeded. Logging can help debug agent behaviors and measure usage (e.g., which queries are frequent). Additionally, metrics like number of queries, average latency, Pinecone/Neo4j errors, etc., can be collected. Using an APM or even simple log files will help maintain the server. 
- **Asynchronous and Concurrency**: FastAPI already supports async endpoints. We should ensure to use asynchronous IO for network calls if the Pinecone and Neo4j clients offer async functions. If not, those calls can be offloaded to a thread pool via `asyncio.to_thread`. This allows the server to handle many simultaneous agent requests without blocking. If the memory operations (like heavy graph traversal) might be slow, consider using background tasks or optimizing queries (e.g., proper indexing in Neo4j).
- **Caching**: To improve performance, a caching layer could be added. For instance, recent query results can be cached in memory (for a short TTL) so if agents repeatedly ask the same question, the server can return results faster without hitting Pinecone/Neo4j every time. Another cache opportunity is embedding generation – if the same text is embedded often (like recurring queries), cache the vector. We might use an LRU cache for embeddings or query->result mapping. 
- **Batch Operations**: The design can be extended to support batch upserts or queries. For example, an endpoint could allow uploading a batch of documents for indexing in one call (reducing overhead if an agent ingests a large knowledge base at once). Similarly, batch query might not be common, but could be useful for analyzing multiple queries offline.
- **Error Handling & Retries**: The server should gracefully handle backend unavailability. If Pinecone or Neo4j is down or a query fails, the server can catch exceptions and return a clear error message to the agent. Implementing retries for transient failures (with exponential backoff) will increase reliability. For example, if a Pinecone upsert times out, retry once or twice before giving up.
- **Scalability**: Although this is a local server for local agents, it could become a bottleneck if many agents are hitting it heavily. We can design the server to be stateless such that it can be horizontally scaled (multiple instances behind a load balancer). Since all state is in Pinecone/Neo4j, multiple MCP server instances can serve requests concurrently. In a local environment, scaling might mean using threads or async tasks optimally. For production, containerizing the server (Docker) and deploying multiple might be considered.
- **Testing**: Include a test suite for the server’s logic. For example, use pytest to test that upserting then querying returns the expected results, etc. This ensures the fusion logic and API work as intended as the project evolves.
- **Versioning & Extensibility**: As the AI system grows, we may want to add more types of memory or new endpoints. Designing the project in a modular way (as we outline next) will make it easier to extend. For example, one could add a new route `/memory/search` that only does vector search, or integrate another vector DB in future. Versioning the API (v1, v2) via URL or headers can allow non-breaking updates.
- **Documentation**: Leverage FastAPI’s automatic docs. Write clear docstrings for each endpoint and Pydantic model so that the Swagger UI generated at `/docs` is informative for developers. Additionally, a README or developer guide (possibly based on this design document) should accompany the code.

In summary, these enhancements ensure the MCP server is **secure, reliable, and maintainable** in a real-world setting. They are optional for a proof-of-concept but highly recommended for production or collaborative environments.

## Project Structure and Organization

Organizing the project well will help multiple developers work on it and keep components decoupled. Here is a suggested folder structure for the MCP memory server project:

```
mcp_memory_server/
├── app/
│   ├── main.py            # FastAPI app initialization and startup
│   ├── api/
│   │   └── memory.py      # FastAPI APIRouter with endpoints (query, upsert, delete)
│   ├── models/
│   │   └── memory_models.py  # Pydantic models for requests and responses
│   ├── services/
│   │   ├── pinecone_service.py  # Functions for Pinecone operations (connect, query, upsert, delete)
│   │   ├── graph_service.py     # Functions for Neo4j/LightRAG operations (query graph, upsert nodes)
│   │   └── memory_fusion.py     # Core logic to combine Pinecone and graph results
│   ├── utils/
│   │   └── embedding.py    # Utility to generate embeddings (using OpenAI API or local model)
│   └── config.py           # Configuration (API keys, environment variables, constants)
├── tests/
│   └── test_memory_api.py  # Test cases for the API endpoints
└── README.md               # Project readme and setup instructions
```

**Explanation of key files/directories:** 

- `app/main.py`: This is where the FastAPI app is created. It will include the routers from the `api` sub-package, set up any middleware (logging, CORS if needed), and possibly load initial resources (e.g., ensure Pinecone is initialized, connect to Neo4j). For example:
  ```python
  app = FastAPI(title="MCP Memory Server", version="1.0")
  # Include the memory router
  from app.api import memory
  app.include_router(memory.router, prefix="/memory")
  ```
  This file might also handle reading config (like Pinecone API key) on startup.

- `app/api/memory.py`: Contains the FastAPI router with the endpoint function definitions. Each function will parse the request into the Pydantic model, call the appropriate service functions, and return the response model. For example, a `@router.post("/query")` function here will create a `MemoryQueryRequest` from the JSON, then call `memory_fusion.query_memory(request)` and return a `MemoryQueryResponse`. Keeping the FastAPI-specific code here, separate from logic, makes it easier to test logic independently.

- `app/models/memory_models.py`: Defines Pydantic `BaseModel` classes for the request/response of each endpoint. For instance:
  ```python
  class MemoryQueryRequest(BaseModel):
      query: str
      top_k: int = 5

  class VectorResult(BaseModel):
      id: str
      score: float
      text: str
      metadata: dict

  class GraphResult(BaseModel):
      entity: str
      properties: dict
      neighbors: list[dict]

  class MemoryQueryResponse(BaseModel):
      query: str
      vector_results: list[VectorResult]
      graph_results: list[GraphResult]
  ```
  And similarly models for `MemoryUpsertRequest`, `MemoryUpsertResponse`, etc. Using models ensures the data going in/out conforms to the expected schema and types.

- `app/services/pinecone_service.py`: This module handles all direct interaction with Pinecone. It might have functions like `init_pinecone()` to set up the Pinecone index at startup, `upsert_vector(id, vector, metadata)`, `query_vectors(query_vector, top_k)`, and `delete_vector(id)`. It will use the `pinecone` Python client under the hood. By isolating Pinecone logic here, if we ever swap out Pinecone for another vector DB, we minimize changes elsewhere. Also, this module can include any Pinecone-specific response parsing (for example, converting Pinecone query results into our internal format).

- `app/services/graph_service.py`: Similarly, this encapsulates interactions with the knowledge graph. For example, functions like `find_entity(name)`, `upsert_entity_relations(text)`, `query_graph(query_text)` etc. It can use the LightRAG library to extract graph info from text (on upsert) and to formulate graph queries on query. Or it can use the Neo4j driver with custom Cypher queries. For instance, on upsert, it might call `lightrag.extract(text)` to get entities/relations and then use Neo4j driver to MERGE those nodes/edges. On query, it might use a combination of exact match and fuzzy match: e.g., find nodes whose name contains the query string, etc., or even utilize an NLP on the query to decide a graph query.

- `app/services/memory_fusion.py`: This is the high-level logic that the API endpoints call. For a query, a function like `query_memory(request: MemoryQueryRequest)` will coordinate `pinecone_service.query_vectors()` and `graph_service.query_graph()`, then merge the results into a `MemoryQueryResponse`. For upsert, `upsert_memory(request: MemoryUpsertRequest)` will call `embedding.embed(text)`, then `pinecone_service.upsert_vector()`, and also `graph_service.upsert_entity_relations(text)`, and finally return a response model. Essentially, this module implements the **fusion algorithms** described earlier, making use of the lower-level service modules.

- `app/utils/embedding.py`: Contains the logic for obtaining embeddings for text. This could wrap an external API call (e.g., OpenAI Embedding API) or use a local model like SentenceTransformers. It provides a function `get_embedding(text: str) -> List[float]`. By abstracting this, we can change the embedding model or vendor by editing this file alone. Also, this module could cache embeddings for repeated texts as mentioned.

- `app/config.py`: Central place for configuration. For example, Pinecone API key/environment, Neo4j connection URI and credentials, any other constants (like the embedding model name or API keys). It can use environment variables and default values. FastAPI could read these at startup. We might have something like:
  ```python
  PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
  PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west4-gcp") 
  NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
  NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
  NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
  ```
  And so on. This keeps secrets/config out of code and in environment or .env files.

- `tests/`: A directory for unit or integration tests. For example, `test_memory_api.py` could spin up a TestClient from FastAPI and simulate calls to the endpoints, using a mock Pinecone and Neo4j (or a test instance). Tests can cover that upserting then querying returns expected results, deleting removes data, etc. Given the complexity of external systems, some tests might use dummy in-memory replacements (e.g., a dict to simulate Pinecone, and NetworkX to simulate a graph) to test logic without requiring actual Pinecone/Neo4j connections.

- `README.md`: Instructions on how to set up and run the server (install requirements, set env vars, run `uvicorn app.main:app`). This is for developers or users of the MCP server.

All Python packages and modules should follow standard conventions, and the project can be installed as a package if needed (with a `pyproject.toml` or `setup.py`). This structure separates concerns:
- API layer vs. Service layer vs. Utility layer.
- Vector store logic vs. Graph logic vs. Fusion logic.

Such separation makes it easier to maintain. For instance, if we want to upgrade the Pinecone client or swap Pinecone for another vector DB, changes stay in `pinecone_service.py`. If LightRAG’s API changes or we want to try a different KG system, we adjust `graph_service.py` accordingly. The API endpoints and models likely remain unchanged in those cases.

Finally, we ensure to document each module and function. This not only helps new developers understand the codebase but also ties back to this design plan. By following this design, Python developers building AI agents can rely on a well-structured, clear, and powerful memory server that integrates seamlessly with both vector search and knowledge graphs – enabling richer context for AI reasoning than either alone could provide. The MCP Memory Server becomes the **single source of truth for agent memory**, abstracting away the complexity of multiple data stores behind a clean HTTP interface. 

