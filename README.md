# Nova AI Fusion Memory MCP Server

## 1. Overview

This project implements a Model Context Protocol (MCP) compliant server that encapsulates the sophisticated memory system of Nova AI. It provides a standardized MCP interface for AI agents (like Roo or Claude Desktop) to interact with a fused memory store, combining semantic vector search (via Pinecone) and structured knowledge graph retrieval (via Neo4j).

The server leverages the core logic components from the original Nova AI system (`query_router`, `hybrid_merger`, `reranker`) to ensure functional parity in memory retrieval, including Reciprocal Rank Fusion (RRF) merging and cross-encoder reranking.

This version uses the official `mcp-python-sdk` (`FastMCP`) framework for handling MCP communication, simplifying the architecture compared to previous versions that used a separate REST API and adapter.

## 2. Architecture

The server now consists of a single primary process defined in `mcp_server.py`:

-   **MCP Server (`mcp_server.py`)**: Built using `FastMCP` from the `mcp-python-sdk`. It defines MCP tools (`query_memory`, `upsert_memory`, `delete_memory`, `check_health`) using decorators. It manages the lifecycle of the underlying `MemoryService` via an `asynccontextmanager` lifespan.
-   **Service Layer (`app/services`)**:
    -   `MemoryService`: Orchestrates memory operations, integrating all components. It is initialized during the MCP server's lifespan startup.
    -   `PineconeClient`: Handles interaction with the Pinecone vector database.
    -   `GraphClient`: Handles interaction with the Neo4j graph database.
    -   `EmbeddingService`: Generates text embeddings using OpenAI.
    -   Reused Nova Modules (`query_router`, `hybrid_merger`, `reranker`): Provide core retrieval logic.
-   **Configuration (`app/config.py`)**: Manages settings via environment variables or a `.env` file (loaded automatically by Pydantic settings or via Docker Compose/`docker run --env-file`).

(Refer to `ARCHITECTURE.md` for a conceptual diagram of the memory pipeline).

## 3. Setup and Installation

### Prerequisites

-   Python 3.10+ (as used in Dockerfile)
-   Docker and Docker Compose
-   Access to OpenAI API (requires API key)
-   Access to Pinecone (requires API key and environment details)
-   *(Neo4j is handled by Docker Compose)*

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Nova_AI_Fusion_Memory_MCP
    ```

2.  **(Optional) Create and activate a Python virtual environment (for local development/testing outside Docker):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On Linux/macOS
    source venv/bin/activate
    # Install dependencies if needed locally
    # pip install -r requirements.txt 
    ```

3.  **Configure Environment Variables:**
    -   Copy the `.env.example` file to `.env`:
        ```bash
        cp .env.example .env 
        # Or on Windows: copy .env.example .env
        ```
    -   Edit the `.env` file and add your actual credentials:
        -   `OPENAI_API_KEY`: Your OpenAI API key.
        -   `PINECONE_API_KEY`: Your Pinecone API key.
        -   `PINECONE_ENV`: Your Pinecone environment (e.g., `us-east-1`).
        -   `NEO4J_PASSWORD`: Optional. Set this only if Neo4j authentication is enabled.
        -   *(Optional)* Adjust `PINECONE_INDEX`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_DATABASE` if they differ from the defaults used in `docker-compose.yml` or `app/config.py`. **Note:** `NEO4J_URI` should typically remain `bolt://neo4j:7687` when using Docker Compose, as this allows the `nova-memory` service to connect to the `neo4j` service within the Docker network.

## 4. Running the Server (Docker Compose Recommended)

Docker Compose handles starting the MCP server and its Neo4j dependency.

1.  **Build the Docker images:**
    ```bash
    docker-compose build
    ```
    *(This builds the `nova-memory` service image based on the `Dockerfile`)*

2.  **Start the MCP server and Neo4j:**
    ```bash
    docker-compose --profile mcp up -d
    ```
    -   `--profile mcp`: Ensures only the `nova-memory` (MCP server) and `neo4j` services are started.
    -   `-d`: Runs the containers in detached mode (in the background).

3.  **Verify the containers are running:**
    ```bash
    docker ps
    ```
    You should see `nova_mcp_server` and `nova_neo4j_db` listed with status "Up".

4.  **View Logs:**
    ```bash
    docker logs nova_mcp_server -f # Follow logs for the MCP server
    docker logs nova_neo4j_db -f  # Follow logs for Neo4j
    ```

5.  **Stopping the Services:**
    ```bash
    docker-compose --profile mcp down
    ```

## 5. Connecting MCP Clients (Roo / Claude Desktop)

This server communicates using the Model Context Protocol over standard input/output when run via Docker. Configure your MCP client (like Roo or Claude Desktop) to connect.

### Configure Roo Integration

Locate your Roo MCP settings file:
-   Windows: `%APPDATA%\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\mcp_settings.json`
-   macOS: `~/Library/Application Support/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json`
-   Linux: `~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json`

Add or update the `nova-memory` server entry:

```json
{
  "mcpServers": {
    "nova-memory": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--env-file",
        ".env",
        "nova-memory-mcp:latest"
      ],
      "cwd": "c:/path/to/your/nova-memory-mcp",  // Replace with your actual path
      "disabled": false,
      "autoApprove": [],
      "alwaysAllow": [
        "query_memory",
        "upsert_memory",
        "delete_memory",
        "check_health"
      ],
      "tools": [
        {
          "name": "query_memory",
          "description": "Query the memory system",
          "inputSchema": {
            "type": "object",
            "properties": {
              "query": {"type": "string"},
              "top_k_vector": {"type": "integer", "minimum": 1, "default": 50},
              "top_k_final": {"type": "integer", "minimum": 1, "default": 15},
              "category": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
              "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}], "default": null},
              "min_score": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": null},
              "run_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}
            },
            "required": ["query"]
          }
        },
        {
          "name": "check_health",
          "description": "Check the health of the memory system",
          "inputSchema": {"type": "object", "properties": {}}
        },
        {
          "name": "upsert_memory",
          "description": "Add or update a memory item",
          "inputSchema": {
            "type": "object",
            "properties": {
              "id": {"type": "string", "description": "Optional ID"},
              "content": {"type": "string"},
              "metadata": {"type": "object"}
            },
            "required": ["content"]
          }
        },
        {
          "name": "delete_memory",
          "description": "Delete a memory item by ID",
          "inputSchema": {
            "type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"]
          }
        }
      ]
    }
  }
}
```

**Key Roo configuration points:**
- `cwd` **must** be set to the absolute path of the project directory so Docker can find the `.env` file
- The relative path `.env` works when `cwd` is properly set
- Restart Roo after saving the configuration

### Configure Claude Desktop Integration

Locate your Claude Desktop configuration file:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

Add or update the `nova-memory` server entry:

```json
{
  "mcpServers": {
    "nova-memory": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--network=nova-memory-mcp_nova_network",
        "--env-file",
        "c:/path/to/your/nova-memory-mcp/.env",
        "nova-memory-mcp:latest"
      ],
      "cwd": "c:/path/to/your/nova-memory-mcp",
      "transportType": "stdio",
      "disabled": false,
      "autoApprove": [],
      "alwaysAllow": [
        "query_memory",
        "upsert_memory",
        "delete_memory",
        "check_health"
      ],
      "tools": [
        {
          "name": "query_memory",
          "description": "Query the memory system",
          "path": "/memory/query",
          "inputSchema": {
            "type": "object",
            "properties": {
              "query": {"type": "string", "description": "The query text to search for in memory"},
              "top_k_vector": {"type": "integer", "minimum": 1, "default": 50},
              "top_k_final": {"type": "integer", "minimum": 1, "default": 15},
              "category": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
              "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}], "default": null},
              "min_score": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": null},
              "run_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}
            },
            "required": ["query"]
          }
        },
        {
          "name": "check_health",
          "description": "Check the health of the memory system",
          "path": "/memory/health",
          "inputSchema": {
            "type": "object",
            "properties": {}
          }
        },
        {
          "name": "upsert_memory",
          "description": "Add or update a memory item",
          "path": "/memory/upsert",
          "inputSchema": {
            "type": "object",
            "properties": {
              "id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null},
              "content": {"type": "string"},
              "metadata": {"anyOf": [{"type": "object"}, {"type": "null"}], "default": null}
            },
            "required": ["content"]
          }
        },
        {
          "name": "delete_memory",
          "description": "Delete a memory item by ID",
          "path": "/memory/%INPUT%",
          "inputSchema": {
            "type": "object",
            "properties": {
              "memory_id": {"type": "string"}
            },
            "required": ["memory_id"]
          }
        }
      ]
    }
  }
}
```

**Key Claude Desktop configuration differences:**
- For Claude Desktop, use the **absolute path** to the `.env` file: `c:/path/to/your/nova-memory-mcp/.env`
- Add the Docker network parameter: `--network=nova-memory-mcp_nova_network`
- Include `"transportType": "stdio"` in the configuration
- Restart Claude Desktop after saving the configuration

### Troubleshooting Connection Issues

If you encounter connection issues:

1. **Check the logs:**
   - Roo logs: Available in VSCode output panel
   - Claude Desktop logs: Located in `%APPDATA%\Claude\logs\mcp-server-nova-memory.log` (Windows)

2. **Common issues:**
   - Missing `.env` file or incorrect path
   - Docker network configuration issues
   - Docker container not running or already in use
   - Incorrect container name or image name

3. **Verify Docker containers:**
   ```bash
   docker ps  # Check if nova_mcp_server is running
   docker logs nova_mcp_server  # Check container logs
   ```

### Using the Memory System

Once connected, you can interact with the memory system:

- **Store Memory:** "Remember that the project kickoff is next Tuesday."
- **Query Memory:** "What do you know about the project kickoff?"
- **Check Health:** "Is the nova-memory system working properly?"

## 6. Future Work

-   Add authentication/authorization for secure access (potentially using MCP's OAuth features).
-   Implement more sophisticated graph querying strategies.
-   Explore batch operations for efficiency.
-   Enhance monitoring and logging.
-   Add support for more MCP operations and memory manipulation features.
