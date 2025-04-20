Perfect. I’ll create a complete step-by-step plan for building your own MCP (Model Context Protocol) server in Python using FastAPI. It will run locally for now, with the option to deploy to the cloud later. It will use OpenAI for embeddings and integrate with your existing Pinecone and Neo4j setup.

This guide will cover everything: setup, file structure, FastAPI routes, memory fusion logic, example requests, and testing. I’ll let you know as soon as it’s ready!

# Building a Personal MCP Memory Server with FastAPI

**Model Context Protocol (MCP)** is an open standard that defines how AI applications can fetch context from various data sources. Think of MCP as a “USB-C port” for AI – it standardizes connecting LLMs to tools and data ([Introduction - Model Context Protocol](https://modelcontextprotocol.io/introduction#:~:text=MCP%20is%20an%20open%20protocol,different%20data%20sources%20and%20tools)). In this guide, we’ll create a step-by-step plan to build a personal MCP server in Python using FastAPI. This server will serve as an AI “memory” system by fusing two components: a vector database (Pinecone) for semantic search and a knowledge graph (Neo4j, via the LightRAG approach) for structured knowledge. LightRAG is a technique that combines knowledge graphs with embedding-based retrieval to improve speed and accuracy of context retrieval ([LightRAG: Simple and Fast Alternative to GraphRAG](https://learnopencv.com/lightrag/#:~:text=LightRAG%20is%20an%20innovative%20approach,and%20GraphRAG%20across%20various%20benchmarks)). The goal is to develop the server locally with easy paths to future cloud deployment.

## 1. Local Environment Setup

Setting up a clean Python environment will ensure our development is isolated and reproducible:

- **Install Python 3.9+** if not already available (FastAPI and Pinecone require Python 3.9 or newer). Verify by running `python --version`.
- **Create a virtual environment** for the project. For example, in your project directory:  
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
  This keeps dependencies isolated.
- **Install required dependencies** via pip. Key libraries include:
  - **FastAPI** (web framework) and **uvicorn** (ASGI server for development) – e.g. `pip install fastapi uvicorn`.
  - **OpenAI Python SDK** (to call OpenAI API for embeddings) – `pip install openai`.
  - **Pinecone client** (to interact with Pinecone vector DB) – `pip install pinecone-client` (the Pinecone Python SDK ([Using Pinecone Vector Database: A beginner guide | by M K Pavan Kumar |  . | Medium](https://medium.com/aimonks/using-pinecone-vector-database-a-beginner-guide-6f81dc827874#:~:text=pinecone))).
  - **Neo4j driver** (for connecting to Neo4j graph database) – `pip install neo4j` (official Neo4j Python driver).
  - **LightRAG library** (for knowledge graph construction, optional) – `pip install lightrag-hku`. Optionally include the API extras if needed: `pip install "lightrag-hku[api]"` ([GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG#:~:text=)).
  - **python-dotenv** (to load API keys from a `.env` file) – `pip install python-dotenv`.
  - (Optional) **httpx** or **requests** for testing the API calls, and any logging/caching libraries as needed.
- **Set up API keys and config**: You will need credentials for Pinecone (API key and environment), OpenAI API key, and Neo4j (URL, user, password if using a database server). It’s best to store these in a `.env` file or as environment variables for security. For example, create a `.env` file with:  
  ```bash
  OPENAI_API_KEY="sk-...yourkey..."  
  PINECONE_API_KEY="your-pinecone-key"  
  PINECONE_ENVIRONMENT="your-pinecone-env"  
  NEO4J_URI="bolt://localhost:7687"  
  NEO4J_USER="neo4j"  
  NEO4J_PASSWORD="password"
  ```  
  These will be loaded by our application at runtime (never commit API secrets to code). If you use Docker later, you can pass these as env variables.

Ensure all these installations complete successfully. Now you have an isolated environment ready with FastAPI, Pinecone SDK, OpenAI SDK, and Neo4j/LightRAG tools.

## 2. Project Folder and File Structure

Organize the project files logically for clarity and maintainability. A possible structure could be:

```
mcp_memory_server/
├── main.py              # FastAPI app initialization and route inclusion
├── requirements.txt     # Python dependencies (optional, for reproducibility)
├── .env                 # Environment variables for config (gitignored)
├── app/                 # Application code package
│   ├── __init__.py
│   ├── config.py        # Configuration handling (loading keys, etc.)
│   ├── models.py        # Pydantic data models for request/response
│   ├── memory/          # Package for memory logic
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Functions to call OpenAI for embeddings
│   │   ├── vector_store.py    # Pinecone vector DB helper functions
│   │   ├── knowledge_graph.py # Neo4j (LightRAG) helper functions
│   │   └── fusion.py          # Logic to fuse vector and graph results
│   └── routes.py        # FastAPI route definitions using the above modules
└── README.md            # Documentation (optional)
```

This is just one way to structure it – for a simple project, you might keep everything in `main.py`, but separating concerns (especially as the system grows) is good practice. The `memory` package encapsulates the core logic of embeddings, vector storage, graph storage, and fusion, while `routes.py` will define the API endpoints that glue these pieces together. The `config.py` can load environment variables (using `dotenv` or `os.environ`) for API keys and database URLs so that the sensitive info is managed in one place.

Make sure to update `requirements.txt` with all needed libraries (from step 1) so others can install them easily. With the structure in place, we can start coding the server components.

## 3. Creating the FastAPI Server

First, set up the FastAPI application and ensure it runs:

- **Initialize FastAPI**: In `main.py` (or an `app/routes.py`), create the FastAPI app instance:
  ```python
  from fastapi import FastAPI
  app = FastAPI(title="MCP Memory Server", version="0.1.0")
  ```
  This creates a FastAPI app with a title and version. You can also include a description and other metadata for documentation if desired.
- **Include Routes**: If you structured routes in a separate module, import and include them. For example, if using `app.routes` module:
  ```python
  from app import routes
  app.include_router(routes.router)
  ```
  Otherwise, you will define routes directly in this file in the next step.
- **Startup Events** (optional): You might use FastAPI’s event handlers to initialize connections at startup. For example, connect to Pinecone or Neo4j when the app starts, so the connections are reused. E.g.:
  ```python
  from fastapi import Depends
  from app.memory import vector_store, knowledge_graph

  @app.on_event("startup")
  async def startup_event():
      vector_store.init_pinecone()         # initialize Pinecone client
      knowledge_graph.init_neo4j()        # connect to Neo4j (if needed)
  ```
  Ensure these `init_...` functions load keys from config and establish any global connections (like creating Pinecone index object, or Neo4j driver instance).
- **Run the Server**: Start the server for local testing. In development, use uvicorn with reload:
  ```bash
  uvicorn main:app --reload
  ```
  This should start the FastAPI server at `http://127.0.0.1:8000`. Opening that in a browser or via curl, you can check the automatic docs at `/docs` (Swagger UI) or a simple health endpoint (we’ll add one next).

At this stage, you have a running FastAPI application (though no routes yet). The console should show something like “Uvicorn running on http://127.0.0.1:8000” if successful ([Using Pinecone Vector Database: A beginner guide | by M K Pavan Kumar |  . | Medium](https://medium.com/aimonks/using-pinecone-vector-database-a-beginner-guide-6f81dc827874#:~:text=uvicorn%20main%3Aapp%20)). Now we will add the specific endpoints for the memory operations.

## 4. Creating Memory API Routes (Upsert, Query, Delete)

We’ll create three main endpoints under a `/memory` path to manage the “memory” content:

- **`POST /memory/upsert`** – Insert or update a piece of memory (semantic + graph).
- **`POST /memory/query`** – Query the memory for relevant context.
- **`DELETE /memory/delete`** – Delete a piece of memory by ID.

Using FastAPI, we can define these routes as async functions. Let’s outline each:

**Data Models**: Define Pydantic models for request/response bodies in `app.models.py` for clarity:
```python
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class UpsertRequest(BaseModel):
    id: Optional[str] = None
    text: str

class UpsertResponse(BaseModel):
    id: str
    status: str

class QueryRequest(BaseModel):
    query: str

class QueryResult(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]  # any additional info (e.g. score)

class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    knowledge: Any  # could define a structure for KG data

class DeleteRequest(BaseModel):
    id: str

class DeleteResponse(BaseModel):
    id: str
    status: str
```
These models specify what the API will expect and return. For simplicity, the query response’s knowledge part is kept generic (`Any` or a simple dict/list) as it might be a list of relationships or nodes. Adjust as needed to structure the knowledge graph output (for example, a list of triples or a subgraph).

**Route Definitions** (in `app/routes.py` or `main.py`):
```python
from fastapi import APIRouter, HTTPException
from app import models, memory

router = APIRouter()

@router.post("/memory/upsert", response_model=models.UpsertResponse)
async def upsert_memory(req: models.UpsertRequest):
    # Generate an ID if not provided
    doc_id = req.id or memory.vector_store.generate_id()
    text = req.text
    # 1. Get embedding for the text
    embedding = memory.embeddings.get_embedding(text)
    # 2. Upsert into Pinecone vector DB
    memory.vector_store.upsert_vector(doc_id, embedding, {"text": text})
    # 3. Extract knowledge and insert into Neo4j graph
    memory.knowledge_graph.insert_text(doc_id, text)
    return {"id": doc_id, "status": "upserted"}

@router.post("/memory/query", response_model=models.QueryResponse)
async def query_memory(req: models.QueryRequest):
    query_text = req.query
    # 1. Embed the query text
    query_vec = memory.embeddings.get_embedding(query_text)
    # 2. Search Pinecone for similar vectors (semantic matches)
    top_results = memory.vector_store.query_vector(query_vec, top_k=5)
    # 3. Query knowledge graph for related info
    kg_results = memory.knowledge_graph.query_knowledge(query_text, top_results)
    # 4. Combine/fuse the results
    fused = memory.fusion.combine_results(top_results, kg_results)
    return {"query": query_text, "results": fused["results"], "knowledge": fused["knowledge"]}

@router.delete("/memory/delete", response_model=models.DeleteResponse)
async def delete_memory(req: models.DeleteRequest):
    doc_id = req.id
    # Delete from Pinecone
    memory.vector_store.delete_vector(doc_id)
    # Delete from Neo4j (knowledge graph)
    memory.knowledge_graph.delete_entry(doc_id)
    return {"id": doc_id, "status": "deleted"}
```

This pseudocode demonstrates the flow:
- **Upsert**: Accepts an `id` (optional) and `text`. It generates an ID if not given (you could use a UUID or a hash). Then:
  1. Embeds the text via OpenAI.
  2. Upserts the embedding into Pinecone with the given ID and possibly store the raw text as metadata (so we can retrieve the original text later).
  3. Processes the text to extract knowledge (entities/relations) and inserts that into Neo4j, linking it to the `doc_id`.
- **Query**: Accepts a query text, then:
  1. Embeds the query text.
  2. Uses Pinecone to find top-K similar vectors (the semantic memory retrieval).
  3. Uses the knowledge graph to find related information. This might involve searching the graph for entities mentioned in the query or related to the results. We’ll detail this fusion logic in Step 8.
  4. Combines the results from Pinecone and Neo4j into a unified response.
- **Delete**: Accepts an `id`. It removes the entry from both Pinecone (vector memory) and Neo4j (graph memory). In Pinecone, deletion by ID is straightforward ([Delete vectors - Pinecone Docs](https://docs.pinecone.io/reference/api/2024-07/data-plane/delete#:~:text=,grpc%20import%20PineconeGRPC%20as%20Pinecone)). In the graph, you might remove the node(s) associated with that document and any relationships, which requires a defined strategy (for example, if each upserted text corresponds to a subgraph of knowledge, you might delete all nodes with a `doc_id` property matching the ID).

Each route returns a confirmation or the data requested. We use `HTTPException` to handle cases like not found (e.g., if deleting an ID that doesn’t exist). FastAPI automatically generates JSON responses from these return dicts or Pydantic models.

## 5. Using OpenAI API to Generate Embeddings

Our server relies on embeddings to represent text in vector form. We use OpenAI’s embedding API for this:

- **OpenAI API Key**: Ensure your OpenAI key is loaded (in code, something like `openai.api_key = os.environ["OPENAI_API_KEY"]`).
- **Choosing a Model**: Use the latest embedding model (at time of writing, `text-embedding-ada-002` is a good choice for general text). This model produces 1536-dimensional vectors.
- **Generating an Embedding**: Use the OpenAI Python SDK to get the embedding for a given text string. For example:  
  ```python
  import openai
  openai.api_key = os.environ["OPENAI_API_KEY"]
  response = openai.Embedding.create(
      input=text,
      model="text-embedding-ada-002"
  )
  embedding_vector = response['data'][0]['embedding']
  ```  
  This call returns the embedding vector for the input text ([How to generate text embeddings with OpenAI's API in Python](https://how.dev/answers/how-to-generate-text-embeddings-with-openais-api-in-python#:~:text=openai.api_key%20%3D%20os.environ%5B)). You can wrap this in a function `get_embedding(text: str) -> List[float]` in `embeddings.py`.
- **Batching and Rate Limits**: If you plan to upsert large documents, you may need to chunk the text (e.g., by paragraphs) and embed each chunk, as the API has input length limits (around 8K tokens for ada-002). For now, assume inputs are reasonably sized (a few paragraphs). Also be mindful of API rate limits and costs – cache embeddings if the same text might be embedded multiple times to avoid duplicate API calls.
- **Embedding size**: Note the length of the resulting vector (e.g., 1536). This must match the dimension of your Pinecone index (see next step). If you switch embedding models, adjust the Pinecone index accordingly.

With this in place, our memory server can convert any text or query into a numeric vector representation that Pinecone can store and search.

## 6. Integrating Pinecone for Vector Storage and Query

Pinecone will serve as the vector database for our semantic memory:

- **Account and Index Setup**: Make sure you have a Pinecone account and have obtained your API key and environment (from Pinecone’s console). Initialize the Pinecone client in code:
  ```python
  import pinecone
  pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
  ```
  This authenticates your client ([Using Pinecone Vector Database: A beginner guide | by M K Pavan Kumar |  . | Medium](https://medium.com/aimonks/using-pinecone-vector-database-a-beginner-guide-6f81dc827874#:~:text=_%20%3D%20load_dotenv%28find_dotenv%28%29%29%20%20,getenv%28%27PINECONE_ENVIRONMENT)). Next, create or connect to an index:
  ```python
  index_name = "mcp-memory"
  if index_name not in pinecone.list_indexes():
      # Create an index with dimension 1536 for ada-002 embeddings
      pinecone.create_index(name=index_name, dimension=1536, metric="cosine")
  index = pinecone.Index(index_name)
  ```
  We choose cosine similarity (you could also use "dotproduct" or "euclidean" depending on your use case) and dimension 1536 to match the embedding. This index creation is one-time (you can also do it via Pinecone UI).
- **Upserting Vectors**: To save an embedding in Pinecone:
  ```python
  vector = embedding_vector  # obtained from OpenAI
  meta = {"text": original_text}  # you can store the raw text or other metadata
  index.upsert([
      {"id": doc_id, "values": vector, "metadata": meta}
  ])
  ``` 
  This will insert (or overwrite) the vector with the given `id` ([Upsert data - Pinecone Docs](https://docs.pinecone.io/guides/data/upsert-data#:~:text=index.upsert%28%20vectors%3D%5B%20%7B%20,B)). You may also specify a namespace if you want to partition data (e.g., by user or by type) – by default it goes into the default namespace. The `metadata` is optional but useful (here we store the original text for retrieval; you could also store tags, timestamps, etc.).
- **Querying Vectors**: To find similar items given a query vector:
  ```python
  result = index.query(vector=query_vector, top_k=5, include_metadata=True)
  ```
  This returns the top 5 closest matches, including their IDs, scores, and metadata ([Using Pinecone Vector Database: A beginner guide | by M K Pavan Kumar |  . | Medium](https://medium.com/aimonks/using-pinecone-vector-database-a-beginner-guide-6f81dc827874#:~:text=def%20query%28self%2C%20query_vector%29%3A%20,3)). From the result, you can extract `matches` list, where each match has an `id`, `score` (similarity), and any stored `metadata` (like the text).
- **Deleting Vectors**: Pinecone provides a delete operation to remove vectors by ID or by filters. For example:
  ```python
  index.delete(ids=[doc_id])
  ``` 
  will delete the vector with that ID ([Delete vectors - Pinecone Docs](https://docs.pinecone.io/reference/api/2024-07/data-plane/delete#:~:text=,grpc%20import%20PineconeGRPC%20as%20Pinecone)). If you used namespaces or filters, you can specify those too (e.g., `index.delete(ids=[...], namespace="my-ns")`).
- **Handling Pinecone Connections**: The Pinecone client maintains internal connections, but it’s fine to use the `index` object across requests. You might initialize the `index` at app startup (as in Step 3’s startup event) and store it in a global or in a FastAPI dependency for reuse, rather than initializing on every request (to improve performance).

At this point, our `/memory/upsert` route can call `upsert_vector(id, embedding)` to store data, and `/memory/query` can call `query_vector(query_vec)` to get relevant vectors. Pinecone operates as the long-term semantic memory, retrieving texts similar in meaning to the query.

## 7. Using LightRAG and Neo4j for Knowledge Graph Storage

Besides raw text search, we want to store and retrieve structured knowledge – facts and relationships extracted from the texts. We’ll use a knowledge graph in Neo4j for this purpose, leveraging the **LightRAG** approach to build the graph:

 ([GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG)) *LightRAG indexing pipeline: when inserting a document, it is chunked, embedded, and an entity-relation extraction step identifies knowledge to store in a graph and a vector store. In our case, Pinecone is the vector store and Neo4j will hold the extracted knowledge.*

- **Neo4j Setup**: For local development, running Neo4j is straightforward (e.g., via the Neo4j Desktop app or Docker container ([GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG#:~:text=Using%20Neo4J%20for%20Storage))). Ensure it’s running at `neo4j://localhost:7687` (the default bolt port) and you have the credentials. In production, you might use a hosted Neo4j Aura DB or similar.
- **Connect to Neo4j**: Using the Neo4j Python driver, initialize a driver and session:
  ```python
  from neo4j import GraphDatabase
  uri = os.environ["NEO4J_URI"]      # e.g., "neo4j://localhost:7687"
  user = os.environ["NEO4J_USER"]    # e.g., "neo4j"
  pwd = os.environ["NEO4J_PASSWORD"] 
  driver = GraphDatabase.driver(uri, auth=(user, pwd))
  session = driver.session()
  ```
  You might establish this in a startup event or on first use. With a session, you can run Cypher queries to create and query nodes/relationships.
- **Extracting Knowledge (LightRAG)**: Manually extracting entities and relations from text can be complex, but LightRAG provides a framework to do this automatically (using language models and heuristics). If you use the LightRAG library:
  - Initialize LightRAG to use Neo4j as its graph storage:  
    ```python
    from lightrag import LightRAG, setup_logger
    setup_logger("lightrag", level="INFO")  # optional: enable LightRAG logging
    rag = LightRAG(graph_storage="Neo4JStorage")  # use Neo4j for KG ([GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG#:~:text=,by%20specifying%20kg%3D%22Neo4JStorage))
    await rag.initialize_storages()  # if LightRAG uses async init for storages
    ```  
    (LightRAG might also need a working directory and an LLM model for extraction – consult LightRAG docs for specifics. It may use a smaller model or OpenAI to extract entities and relations.)
  - Insert documents into LightRAG:  
    ```python
    documents = [text]
    rag.insert(documents)
    ```  
    Under the hood, this will chunk the text, extract entities and relations, and store them in the Neo4j database (since we configured Neo4JStorage). The knowledge graph will consist of nodes for entities and relationships between them derived from the text ([LightRAG: Simple and Fast Alternative to GraphRAG](https://learnopencv.com/lightrag/#:~:text=LightRAG%20is%20an%20innovative%20approach,and%20GraphRAG%20across%20various%20benchmarks)). LightRAG ensures the graph is populated with the content of the text in structured form.
  - (If not using LightRAG library) **Custom extraction**: As an alternative, you could implement a simpler extraction:
    - Use an NLP library or OpenAI to identify named entities in the text (people, organizations, dates, etc.).
    - Use sentence parsing or prompts to identify relationships between these entities (e.g., "X works for Y", "X located in Y").
    - Create nodes for each unique entity (with labels like `:Person`, `:Place`, etc.) and relationships for each relation found (with appropriate types).
    - Tag each node/relation with the `doc_id` or source, so you know which document contributed that knowledge.
    This approach is much more involved, which is why LightRAG is valuable – it automates this extraction and graph construction step.
- **Insert into Neo4j**: If using LightRAG’s `rag.insert`, it will do it for you. If doing manually via the driver:
  ```python
  session.run(
      "MERGE (e:Entity {name: $name}) "
      "ON CREATE SET e.type=$type, e.description=$desc, e.doc_id=$doc_id",
      {"name": ent_name, "type": ent_type, "desc": ent_description, "doc_id": doc_id}
  )
  # Similarly create other entity nodes and relationships
  session.run(
      "MATCH (a:Entity {name:$source, doc_id:$doc_id}), (b:Entity {name:$target, doc_id:$doc_id}) "
      "MERGE (a)-[r:RELATED {relationship:$rel, doc_id:$doc_id}]->(b)",
      {"source": src_entity, "target": tgt_entity, "rel": relation_desc, "doc_id": doc_id}
  )
  ```
  This Cypher pseudocode merges entity nodes and a relationship. We attach `doc_id` so that later we can retrieve all knowledge from a given document or remove it easily. (LightRAG internally does something similar; it even supports deleting by document ID for cleanup ([GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG#:~:text=,2024.11.09%5DIntroducing%20the%20LightRAG%20Gui)).)
- **Graph Schema**: Decide how to model knowledge. A generic approach: each unique entity (with a name or identifier) is a node, and relationships are edges with a type or description. For example, inserting the sentence *"Alice works at Acme Corp."* would create nodes `(:Person {name:"Alice"})`, `(:Organization {name:"Acme Corp"})` and a relationship `(:Person {name:"Alice"})-[:WORKS_AT]->(:Organization {name:"Acme Corp"})`. You may also store textual evidence or attributes on nodes/edges (e.g., a snippet of text as description, or a confidence score).
- **Verify insertion**: After an upsert, you can query Neo4j (using Cypher in Neo4j browser or via the driver) to ensure the knowledge was saved. For example: `MATCH (n) RETURN n LIMIT 10;` to see some nodes.

With the knowledge graph building in place, our upsert route now not only stores the text vector but also populates the Neo4j graph with facts from the text. This structured data will be used to enrich query results and provide a more robust context to the LLM.

## 8. Memory Fusion Logic – Combining Vector and Graph Results

The core advantage of a fused memory system is being able to draw both from unstructured text embeddings and structured knowledge graph data. In the query workflow, after obtaining results from Pinecone and Neo4j, we need to **fuse** them into a coherent answer.

 ([GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG)) *LightRAG retrieval flow: both vector-similar text chunks and knowledge graph relations are retrieved, combined into a final context for the query. This dual retrieval helps capture both direct textual matches and indirectly related knowledge.*

Here’s how we can implement the fusion logic:

- **Retrieve from Pinecone**: From Pinecone’s query (Step 6), we have the top K vector matches for the query. Each match might include:
  - `id` (the document ID or chunk ID that was inserted),
  - `score` (similarity score),
  - `metadata` (which includes the original text snippet).
- **Retrieve from Neo4j**: We want knowledge that’s relevant to the query. Two approaches (which can be combined):
  1. **By Document ID**: For each top result from Pinecone, fetch the knowledge graph nodes and relations that came from the same source document. Because in the upsert we tagged knowledge with `doc_id`, we can do a Cypher query like:  
     ```cypher
     MATCH (e:Entity)-[r]->(f:Entity) 
     WHERE e.doc_id IN $ids 
     RETURN e.name as source, r.relationship as rel, f.name as target
     ```  
     with `$ids` being the list of top Pinecone result IDs. This will pull all facts extracted from those top-matching documents. These facts might reinforce or elaborate on the content of those documents.
  2. **By Query Terms/Entities**: Parse the query itself for any entity names or keywords that might exist in the graph. For example, if the query is "Where does Alice work?", identify "Alice" as a likely entity. Then query Neo4j for that entity and its relationships:  
     ```cypher
     MATCH (a:Entity {name:$name})-[r]->(b:Entity) 
     RETURN a.name as source, r.relationship as rel, b.name as target
     ```  
     If "Alice" exists in the graph (from any document), this would retrieve relations like "Alice WORKS_AT Acme Corp".
  In practice, you can do both: get knowledge from relevant docs and directly relevant to query terms. LightRAG’s retrieval does a form of this dual-step: it finds “topic entities” related to the query and uses them to pull a subgraph ([LightRAG: The Ultimate Guide to Fast, Graph-Based Retrieval ...](https://medium.com/@divyanshbhatiajm19/lightrag-the-ultimate-guide-to-fast-graph-based-retrieval-augmented-generation-2025-42edff346ab2#:~:text=LightRAG%3A%20The%20Ultimate%20Guide%20to,LLMs%20to%20access%20information)).
- **Combine Results**: Now merge the semantic and graph results. The output structure could be:
  ```json
  {
    "query": "...",
    "results": [
       { "id": "doc1", "text": "snippet from doc1...", "score": 0.89 },
       { "id": "doc2", "text": "snippet from doc2...", "score": 0.85 }
    ],
    "knowledge": [
       { "source": "Alice", "rel": "WORKS_AT", "target": "Acme Corp" },
       { "source": "Bob", "rel": "MANAGES", "target": "Alice" }
    ]
  }
  ``` 
  The `results` list comes from Pinecone (textual memory), and the `knowledge` list comes from Neo4j (structured memory). In your `memory.fusion.combine_results` function, you can format data this way. You might also remove duplicate facts or limit the knowledge results to avoid overload.
- **Ranking or Filtering**: In some cases, you may want to prioritize certain info. For example, if Pinecone results are very strong (high similarity), those text snippets might already contain the answer. The knowledge graph facts provide supporting info or indirect info (like relationships). If the query is very factoid (e.g., asking a specific known relation), the graph might directly answer it. You could choose to merge by simply appending, or do some logic like: if an entity from the query is found in graph, prefer those facts, etc. As an initial approach, just provide all relevant info.
- **LightRAG’s method**: If using the LightRAG library’s retrieval, a lot of this is handled internally. LightRAG would use the query to retrieve a subgraph and relevant text together. But since we are doing a custom implementation, we’re essentially replicating that idea: combining top vector matches (local context) with global knowledge graph context ([LightRAG: Simple and Fast Alternative to GraphRAG](https://learnopencv.com/lightrag/#:~:text=LightRAG%20is%20an%20innovative%20approach,and%20GraphRAG%20across%20various%20benchmarks)).

The end result of fusion is that when a client (like an LLM agent) queries the MCP server, it gets a rich context: some raw text passages that closely match the query semantically, and some structured facts that might be useful. The client can then feed both into an LLM prompt (or use them to answer directly if it has logic to do so). This synergy helps reduce hallucinations and improves factual accuracy, as the LLM has both evidence and distilled knowledge.

## 9. Testing the Endpoints Locally

With the server running (via `uvicorn`), it’s important to test each route to ensure everything works as expected:

- **Health Check** (optional): If you added a `/health` GET route (like in the earlier Medium example ([Using Pinecone Vector Database: A beginner guide | by M K Pavan Kumar |  . | Medium](https://medium.com/aimonks/using-pinecone-vector-database-a-beginner-guide-6f81dc827874#:~:text=%40app.get%28,OK))), you can simply curl it:
  ```bash
  curl http://localhost:8000/health
  ```
  Expected result: `{"message":"OK"}` or similar, confirming the server is up.
- **Test Upsert**: Use `curl` or an API client to POST to `/memory/upsert`. For example:
  ```bash
  curl -X POST http://localhost:8000/memory/upsert \
       -H "Content-Type: application/json" \
       -d '{"id": "doc1", "text": "Alice works at Acme Corp."}'
  ``` 
  This should return a JSON confirming the upsert, e.g. `{"id":"doc1","status":"upserted"}`. Internally, this will have called OpenAI to embed the text, stored the vector in Pinecone, and added nodes “Alice” and “Acme Corp” with a relationship in Neo4j.
  - Try upserting a couple of different pieces of text to build a small memory. For example, another text: `{"id": "doc2", "text": "Acme Corp is located in San Francisco."}` which would add location info to the graph.
- **Test Query**: Now query the memory:
  ```bash
  curl -X POST http://localhost:8000/memory/query \
       -H "Content-Type: application/json" \
       -d '{"query": "Where does Alice work?"}'
  ``` 
  The response should be a JSON containing relevant text snippets and knowledge. For instance:
  ```json
  {
    "query": "Where does Alice work?",
    "results": [
      {"id":"doc1","text":"Alice works at Acme Corp.","score":...}
    ],
    "knowledge": [
      {"source":"Alice","rel":"WORKS_AT","target":"Acme Corp"},
      {"source":"Acme Corp","rel":"LOCATED_IN","target":"San Francisco"}
    ]
  }
  ``` 
  The exact format depends on how you implemented fusion, but we expect to see that it found "Alice works at Acme Corp." from the semantic search, and also knowledge graph facts like Alice→WORKS_AT→Acme Corp, possibly supplemented by the fact that Acme Corp is in San Francisco (from the second document). This shows the fused result containing both direct answer and related info.
- **Test Delete**: Now test deletion:
  ```bash
  curl -X DELETE http://localhost:8000/memory/delete \
       -H "Content-Type: application/json" \
       -d '{"id": "doc1"}'
  ``` 
  Response should confirm deletion: `{"id":"doc1","status":"deleted"}`. To verify:
  - If you query the memory again with the same question, the results should no longer include doc1’s info (Pinecone won’t return it). If your knowledge graph deletion removed the nodes for doc1, the "Alice works at Acme Corp" relation might be gone (though if doc2 mentioned Acme, some info on Acme might remain).
  - Also, you can try retrieving from Pinecone or Neo4j directly to confirm. For Pinecone, you could attempt a fetch by ID or see that similarity scores drop. For Neo4j, run a Cypher query to ensure the nodes with doc_id "doc1" are gone.

Using **httpx** in a Python script or interactive shell is another convenient way to test:
```python
import httpx
resp = httpx.post("http://localhost:8000/memory/upsert", json={"id":"test1","text":"Bob manages Alice at Acme Corp."})
print(resp.json())
# => {"id": "test1", "status": "upserted"}
resp = httpx.post("http://localhost:8000/memory/query", json={"query":"Who manages Alice?"})
print(resp.json())
```
This would yield a response showing that "Bob manages Alice at Acme Corp" was found and perhaps the knowledge graph indicates Bob → MANAGES → Alice, etc.

Run through various scenarios to ensure the server handles them:
- Query something that wasn’t inserted to see how it behaves (likely returns no results or an empty list).
- Upsert multiple times with the same ID to see that it updates (Pinecone will overwrite the vector for an existing ID on upsert ([Upsert data - Pinecone Docs](https://docs.pinecone.io/guides/data/upsert-data#:~:text=If%20a%20record%20ID%20already,record%2C%20update%20%20the%20record))).
- Error handling: try a delete on an ID that doesn’t exist (it should perhaps return a 404 or just status "deleted" with no effect).

Local testing ensures the logic is solid before moving to deployment.

## 10. **Optional:** Adding Authentication, Logging, and Caching

Once the basic functionality works, you may consider additional features to improve security and performance:

- **Authentication**: If this server will be accessible beyond your local machine, protect it. A simple API key auth can be implemented by requiring a header (e.g., `x-api-key`) in each request. In FastAPI, you can use `Depends` to create a dependency that checks the header against a preset key:
  ```python
  from fastapi import Depends, HTTPException, Security
  from fastapi.security import APIKeyHeader

  API_KEY = "mysecretkey"
  api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)

  def verify_api_key(key: str = Security(api_key_header)):
      if key != API_KEY:
          raise HTTPException(status_code=401, detail="Unauthorized")
      return True

  # Then use Depends in your routes:
  @router.post("/memory/upsert", dependencies=[Depends(verify_api_key)])
  async def upsert_memory(...):
      ...
  ```
  This will require clients to include `x-api-key: mysecretkey` in headers. For more robust auth, consider OAuth2 or integrating with an identity service, but an API key is a quick solution.
- **Logging**: Use Python’s `logging` to record events. For example, log each request or important actions (like “Inserted doc1 in Pinecone and Neo4j”). FastAPI’s uvicorn logs will show requests, but you can add your own for debugging. If using LightRAG, it has its internal logging which we enabled with `setup_logger` in step 7 (set level to INFO or DEBUG for more details) ([GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG#:~:text=,INFO)). Logging is crucial when running the server in the cloud to debug issues (you can send logs to stdout or a file).
- **Caching**: Caching can save time and cost:
  - **Embed caching**: Keep a dictionary or use an LRU cache for `get_embedding`. If the same text is embedded repeatedly, reuse the result instead of calling OpenAI again. For instance, you might use `functools.lru_cache` on the `get_embedding` function.
  - **Pinecone query caching**: If certain queries repeat often, you could cache the last results for a short time. This is trickier because the memory can change (after upserts or deletes), so consider cache invalidation. A simple approach: cache query results for, say, 60 seconds to handle identical consecutive queries.
  - **Graph caching**: If Neo4j queries are expensive, you might also cache knowledge query results for recent queries or entities.
  - Use caution with caching to not serve stale data after updates. Since our use-case might not need extreme performance, caching is optional. But if using this in production with heavy load, a caching layer (or even in-memory vector store for quick hits) could be beneficial.
- **Input validation and size limits**: As an enhancement, ensure the input text isn’t too large (the OpenAI embedding has limits). You can enforce a max length and return a 400 error if exceeded, or automatically chunk large inputs.
- **Rate limiting** (if public): prevent abuse by limiting how many requests per minute a single client can do. This can be done via external libraries or proxies.

Adding these features will make your MCP server more secure and robust, especially if it’s going to be used by other applications or users.

## 11. **Optional:** Deployment Notes (Docker, Cloud, etc.)

Deploying the FastAPI-based MCP server to the cloud will make it accessible to your AI applications anywhere:

- **Containerization with Docker**: Create a `Dockerfile` for your application. For example:
  ```Dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  ENV PORT=8000
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
  Build the image with `docker build -t mcp-memory-server .` and test it locally (`docker run -p 8000:8000 mcp-memory-server`). Ensure it can reach Pinecone (which is cloud-based) and Neo4j (you might link a Neo4j container or use a cloud Neo4j).
- **Hosted services**: Many options exist:
  - **Render.com** or **Railway.app**: you can deploy directly from a GitHub repo. They’ll build the Docker image or use `requirements.txt` and start uvicorn for you. These platforms allow you to set environment variables (for your API keys and DB URIs) in their dashboard.
  - **Heroku** (container stack) or **Fly.io**: also can deploy Dockerized apps easily. Fly.io even allows deploying Neo4j as a companion if needed.
  - **AWS**: You could use AWS Elastic Container Service or Fargate to run the Docker container, or AWS App Runner for a simpler setup. Alternatively, API Gateway + Lambda (using e.g. Mangum to adapt FastAPI to Lambda) is possible for low volumes.
  - **Azure or GCP**: Similar container or app services exist (Azure App Service, Google Cloud Run) which are straightforward for running FastAPI apps.
- **Neo4j in production**: If you need the knowledge graph in the cloud, consider using Neo4j Aura (Neo4j’s cloud service) or a managed Neo4j instance. Update the `NEO4J_URI` in your config to point to it. Ensure network rules allow your app to connect. If running Neo4j yourself on a VM or container, you’ll need to manage its availability and data persistence.
- **Pinecone**: Since Pinecone is a hosted service, you just need to ensure the environment variables for API key/env are set in your deployed app. There’s no additional infra needed for Pinecone.
- **Scaling considerations**: In a cloud deploy, you might run multiple instances for load balancing. Pinecone can handle concurrent requests, but be mindful of OpenAI rate limits if many queries embed simultaneously – you might need to request a rate limit increase or use a local embedding model as an improvement.
- **Monitoring**: Use logging or cloud monitoring to keep an eye on the health and performance of your server. For instance, you could integrate something like Prometheus or use the host’s monitoring.

When deploying, test the endpoints in the cloud as you did locally (perhaps with a small test script or via the provided docs UI). With Docker and modern PaaS offerings, moving a FastAPI app to production can be done in a matter of minutes.

---

**Conclusion:** Following this plan, you will have a running MCP-compatible server that provides a powerful fused memory for LLMs – combining semantic search via Pinecone with knowledge graph queries via Neo4j/LightRAG. This setup allows an AI model to retrieve both relevant textual context and structured facts, improving its ability to generate accurate and context-rich responses. The local-first development approach makes it easy to test and iterate, while the clean design (FastAPI structure and modular memory logic) facilitates future enhancements like authentication, scaling, and deployment to the cloud. Happy building!

**Sources:**

1. Pinecone vector DB usage with FastAPI and upsert/query examples ([Using Pinecone Vector Database: A beginner guide | by M K Pavan Kumar |  . | Medium](https://medium.com/aimonks/using-pinecone-vector-database-a-beginner-guide-6f81dc827874#:~:text=_%20%3D%20load_dotenv%28find_dotenv%28%29%29%20%20,getenv%28%27PINECONE_ENVIRONMENT)) ([Using Pinecone Vector Database: A beginner guide | by M K Pavan Kumar |  . | Medium](https://medium.com/aimonks/using-pinecone-vector-database-a-beginner-guide-6f81dc827874#:~:text=def%20query%28self%2C%20query_vector%29%3A%20,3))  
2. Pinecone API documentation on upserting vectors ([Upsert data - Pinecone Docs](https://docs.pinecone.io/guides/data/upsert-data#:~:text=index.upsert%28%20vectors%3D%5B%20%7B%20,B)) and deleting by ID ([Delete vectors - Pinecone Docs](https://docs.pinecone.io/reference/api/2024-07/data-plane/delete#:~:text=,grpc%20import%20PineconeGRPC%20as%20Pinecone))  
3. OpenAI API example for generating text embeddings ([How to generate text embeddings with OpenAI's API in Python](https://how.dev/answers/how-to-generate-text-embeddings-with-openais-api-in-python#:~:text=openai.api_key%20%3D%20os.environ%5B))  
4. Model Context Protocol (MCP) introduction ([Introduction - Model Context Protocol](https://modelcontextprotocol.io/introduction#:~:text=MCP%20is%20an%20open%20protocol,different%20data%20sources%20and%20tools)) – standardizing context integration for LLMs  
5. LightRAG concept of combining knowledge graphs with embedding retrieval ([LightRAG: Simple and Fast Alternative to GraphRAG](https://learnopencv.com/lightrag/#:~:text=LightRAG%20is%20an%20innovative%20approach,and%20GraphRAG%20across%20various%20benchmarks)) and Neo4j integration setup