# Nova AI Memory MCP Server

## 1. Overview

This project implements a Model Context Protocol (MCP) compliant server that encapsulates the sophisticated memory system of Nova AI. It provides a standardized REST API for AI agents or other applications to interact with a fused memory store, combining semantic vector search (via Pinecone) and structured knowledge graph retrieval (via Neo4j).

The server leverages the core logic components from the original Nova AI system (`query_router`, `hybrid_merger`, `reranker`) to ensure functional parity in memory retrieval, including Reciprocal Rank Fusion (RRF) merging and cross-encoder reranking.

## 2. Architecture

The server is built using FastAPI and follows a layered architecture:

-   **API Layer (`app/api`)**: Defines REST endpoints (`/memory/query`, `/memory/upsert`, `/memory/{id}`, `/memory/health`).
-   **Service Layer (`app/services`)**:
    -   `MemoryService`: Orchestrates memory operations, integrating all components.
    -   `PineconeClient`: Handles interaction with the Pinecone vector database.
    -   `GraphClient`: Handles interaction with the Neo4j graph database.
    -   `EmbeddingService`: Generates text embeddings using OpenAI.
    -   Reused Nova Modules (`query_router`, `hybrid_merger`, `reranker`): Provide core retrieval logic.
-   **Configuration (`app/config.py`)**: Manages settings via environment variables or a `.env` file.

(Refer to `ARCHITECTURE.md` for a detailed diagram).

## 3. Setup and Installation

### Prerequisites

-   Python 3.9+
-   Access to OpenAI API (requires API key)
-   Access to Pinecone (requires API key and environment details)
-   A running Neo4j database instance (local Docker container or cloud instance like AuraDB)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd nova_memory_mcp
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On Linux/macOS
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` was created in Task T2. Ensure it's up-to-date if dependencies change).*

4.  **Configure Environment Variables:**
    -   Copy the `.env.example` file to `.env`:
        ```bash
        cp .env.example .env
        ```
    -   Edit the `.env` file and add your actual credentials:
        -   `OPENAI_API_KEY`: Your OpenAI API key.
        -   `PINECONE_API_KEY`: Your Pinecone API key.
        -   `PINECONE_ENV`: Your Pinecone environment (e.g., `us-west1-gcp`).
        -   `NEO4J_PASSWORD`: The password for your Neo4j database user.
        -   *(Optional)* Adjust `PINECONE_INDEX`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_DATABASE` if they differ from the defaults.

5.  **Ensure Neo4j is Running:**
    -   If using Docker locally (recommended for development), you might use a `docker-compose.yml` (Task T16 will create this) or run Neo4j manually:
        ```bash
        docker run --rm -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/<your_neo4j_password> neo4j:latest
        ```
        *(Replace `<your_neo4j_password>` with the password set in your `.env` file)*.
    -   Ensure the `NEO4J_URI` in your `.env` file points to the correct Bolt URL (e.g., `bolt://localhost:7687`).

## 4. Running the Server

Once the setup is complete, you can run the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

-   `--reload`: Enables auto-reloading when code changes (useful for development).
-   `--port 8000`: Specifies the port to run on (default is 8000).

The server will start, and you should see output indicating it's running, including the initialization status of Pinecone, Neo4j, and the Reranker.

You can access the automatically generated API documentation:

-   **Swagger UI:** `http://localhost:8000/docs`
-   **ReDoc:** `http://localhost:8000/redoc`

## 5. API Endpoints

The server exposes the following endpoints under the `/memory` prefix:

-   **`POST /memory/query`**:
    -   Retrieves relevant memory items based on a query.
    -   **Request Body:** `{"query": "Your query text"}`
    -   **Response Body:** `{"results": [{"id": ..., "text": ..., "source": ..., "score": ..., "metadata": ...}, ...]}`

-   **`POST /memory/upsert`**:
    -   Adds or updates a memory item.
    -   **Request Body:** `{"id": "optional_id", "content": "Memory content", "metadata": {"key": "value"}}`
    -   **Response Body:** `{"id": "item_id", "status": "success"}`

-   **`DELETE /memory/{memory_id}`**:
    -   Deletes a memory item by its ID.
    -   **Path Parameter:** `memory_id` (string)
    -   **Response Body:** `{"id": "item_id", "status": "deleted"}`

-   **`GET /memory/health`**:
    -   Checks the health of the server and its dependencies.
    -   **Response Body:** `{"status": "ok", "pinecone": "ok", "graph": "ok", "reranker": "loaded"}` (or error details)

## 6. Connecting an AI Agent

An AI agent (or any client application) can interact with this MCP server by making standard HTTP requests to the endpoints described above.

Here's a basic Python example using the `requests` library:

```python
import requests
import json

# Base URL of the running MCP server
MCP_SERVER_URL = "http://localhost:8000" # Adjust if running elsewhere

def query_memory(query_text: str):
    """Sends a query to the MCP server."""
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/memory/query",
            json={"query": query_text}
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying memory: {e}")
        return None

def upsert_memory(content: str, item_id: str = None, metadata: dict = None):
    """Upserts a memory item."""
    payload = {"content": content}
    if item_id:
        payload["id"] = item_id
    if metadata:
        payload["metadata"] = metadata

    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/memory/upsert",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error upserting memory: {e}")
        return None

def delete_memory(item_id: str):
    """Deletes a memory item."""
    try:
        response = requests.delete(f"{MCP_SERVER_URL}/memory/{item_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error deleting memory: {e}")
        return None

# --- Example Agent Interaction ---
if __name__ == "__main__":
    # Example: Upsert some information
    upsert_info = upsert_memory(
        content="The capital of France is Paris.",
        item_id="france_capital",
        metadata={"topic": "geography"}
    )
    if upsert_info:
        print(f"Upsert successful: {upsert_info}")

    # Example: Query the memory
    query = "What is the capital of France?"
    print(f"\nQuerying: {query}")
    query_result = query_memory(query)

    if query_result and query_result.get("results"):
        print("Results:")
        for i, item in enumerate(query_result["results"]):
            print(f"  {i+1}. ID: {item['id']}, Score: {item['score']:.4f}, Text: {item['text']}")
    elif query_result:
        print("No relevant results found.")

    # Example: Delete the memory
    # delete_info = delete_memory("france_capital")
    # if delete_info:
    #     print(f"\nDelete successful: {delete_info}")

```

An agent would typically:
1.  Call `query_memory` before generating a response to retrieve relevant context.
2.  Format the retrieved `results` (text snippets) into its prompt.
3.  Generate its response.
4.  Potentially call `upsert_memory` to store new information learned during the interaction or the conversation turn itself.

## 7. Future Work

-   Add authentication/authorization for secure access.
-   Implement more sophisticated graph querying strategies.
-   Explore batch API endpoints for bulk operations.
-   Enhance monitoring and logging.
