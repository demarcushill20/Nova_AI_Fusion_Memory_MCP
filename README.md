# Nova AI Fusion Memory MCP Server

## 1. Overview

A Model Context Protocol (MCP) server that provides AI agents with a fused memory system combining semantic vector search (Pinecone), structured knowledge graph retrieval (Neo4j), and chronological timeline indexing (Redis). Built with the official `mcp-python-sdk` (`FastMCP`).

The server uses `query_router`, `hybrid_merger`, and `reranker` modules for intelligent retrieval including Reciprocal Rank Fusion (RRF) merging and cross-encoder reranking.

## 2. Architecture

```
                         ┌──────────────────────┐
                         │    MCP Clients        │
                         │  (Claude, Roo, etc.)  │
                         └──────────┬───────────┘
                                    │ stdio/MCP
                         ┌──────────▼───────────┐
                         │    mcp_server.py      │
                         │    (FastMCP)          │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   MemoryService       │
                         │   (orchestrator)      │
                         └──┬────┬────┬────┬────┘
                            │    │    │    │
              ┌─────────────┘    │    │    └─────────────┐
              ▼                  ▼    ▼                  ▼
        ┌───────────┐   ┌──────────┐ ┌──────────┐  ┌──────────┐
        │ Pinecone  │   │  Neo4j   │ │  Redis   │  │ Embedding│
        │ (vectors) │   │ (graph)  │ │(timeline)│  │ Service  │
        └───────────┘   └──────────┘ └──────────┘  └──────────┘
```

### Service Layer (`app/services/`)

| Service | Purpose |
|---------|---------|
| `MemoryService` | Orchestrates all memory operations — upsert, query, checkpoint, temporal retrieval |
| `PineconeClient` | Semantic vector storage and retrieval |
| `GraphClient` | Neo4j knowledge graph with Session nodes, FOLLOWS chains, and INCLUDES edges |
| `RedisTimeline` | Sorted set timeline index for O(log N) chronological queries |
| `SequenceService` | Monotonic event counter — Redis INCR primary, file-based fallback |
| `EmbeddingService` | Text embeddings via OpenAI |
| `QueryRouter` | Routes queries to VECTOR, GRAPH, HYBRID, TEMPORAL, or TEMPORAL_SEMANTIC modes |
| `HybridMerger` | Reciprocal Rank Fusion across retrieval backends |
| `Reranker` | Cross-encoder reranking for final result ordering |

### Configuration (`app/config.py`)

Settings via environment variables or `.env` file, managed by Pydantic settings.

## 3. Chronological Memory System

Every memory write receives a monotonic `event_seq` and ISO 8601 `event_time`. This provides strict "what happened after what" ordering independent of clock drift or multi-agent write interleaving.

### Phase 1 — Write-Time Chronology Enforcement
- Every `upsert_memory` and `bulk_upsert_memory` call injects `event_seq`, `event_time`, and `memory_type` into metadata
- `event_seq` is system-assigned (caller cannot override) and strictly monotonic
- `memory_type` defaults to `"scratch"` if not provided

### Phase 2 — Session Checkpoint System
- `create_checkpoint` stores session boundaries as first-class memory items (`memory_type: "checkpoint"`)
- Each checkpoint snapshots the current `event_seq` as `last_event_seq`
- `get_last_checkpoint` retrieves the most recent checkpoint, optionally filtered by project/thread

### Phase 3 — Temporal Retrieval Tools
- `get_recent_events` returns the N most recent events ordered by `event_seq` DESC
- Supports filtering by project, thread_id, memory_type, since_seq, and since_time
- Pinecone backend: over-fetch with 5x factor, client-side sort (Pinecone can't sort)
- Neo4j backend: native `ORDER BY` for server-side sorting

### Phase 4 — Temporal-First Query Router
- `QueryRouter` detects recency-intent keywords ("last session", "most recent", "where were we", "catch up", etc.)
- **TEMPORAL** mode: pure recency, no semantic similarity
- **TEMPORAL_SEMANTIC** mode: temporal window first, then semantic refinement within that window
- Temporal detection takes priority — won't be downgraded to HYBRID

### Phase 5 — Graph Time Model
- Neo4j Session nodes with `session_id`, `started_at`, `last_event_seq`
- `INCLUDES` edges: Session → MemoryItem (event linking)
- `FOLLOWS` edges: Session → PreviousSession (temporal chain)
- `get_session_events` retrieves all events for a given session via graph traversal

### Phase 6 — Redis Timeline Store
- Redis sorted sets (`nova:timeline:{scope}`) with `event_seq` as score
- O(log N) recency queries via `ZREVRANGE`, replacing Pinecone dummy-vector over-fetch
- Dedicated checkpoint index (`nova:checkpoints:{scope}`) with O(1) latest lookup
- Dual-scope: project-specific timelines + automatic global aggregation
- `SequenceService` dual-backend: Redis `INCR` primary with file-based fallback
- Bidirectional counter sync on startup, graceful degradation on Redis failure

## 4. MCP Tools

| Tool | Description |
|------|-------------|
| `query_memory` | Semantic/temporal/hybrid query with intelligent routing |
| `upsert_memory` | Store or update a memory item (auto-assigns event_seq) |
| `bulk_upsert_memory` | Batch store multiple items with consecutive event_seq allocation |
| `delete_memory` | Delete a memory item by ID |
| `create_checkpoint` | Create a session checkpoint with event_seq snapshot |
| `get_last_checkpoint` | Retrieve the most recent checkpoint |
| `get_recent_events` | Get N most recent events by event_seq (temporal retrieval) |
| `get_session_events` | Get all events linked to a session via graph traversal |
| `check_health` | Health check across all backends (Pinecone, Neo4j, Redis) |

## 5. Setup and Installation

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- OpenAI API key
- Pinecone API key and environment
- *(Neo4j and Redis are handled by Docker Compose)*

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Nova_AI_Fusion_Memory_MCP
    ```

2.  **Configure Environment Variables:**
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and set:
    - `OPENAI_API_KEY` — Your OpenAI API key
    - `PINECONE_API_KEY` — Your Pinecone API key
    - `PINECONE_ENV` — Your Pinecone environment (e.g., `us-east-1`)
    - `NEO4J_PASSWORD` — Optional, only if Neo4j auth is enabled
    - `REDIS_URL` — Defaults to `redis://redis_db:6379/0` (Docker internal)
    - `REDIS_ENABLED` — Defaults to `true`

3.  **(Optional) Local development:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    pip install -r requirements.txt
    ```

## 6. Running the Server (Docker Compose)

```bash
# Build images
docker-compose build

# Start all services (MCP server, Neo4j, Redis)
docker-compose --profile mcp up -d

# Verify
docker ps
# Expected: nova_mcp_server, nova_neo4j_db, redis_db

# View logs
docker logs nova_mcp_server -f

# Stop
docker-compose --profile mcp down
```

## 7. Connecting MCP Clients

### Roo (VS Code)

Settings file location:
- Windows: `%APPDATA%\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\mcp_settings.json`
- macOS: `~/Library/Application Support/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json`
- Linux: `~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json`

```json
{
  "mcpServers": {
    "nova-memory": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "--env-file", ".env",
        "nova-memory-mcp:latest"
      ],
      "cwd": "/path/to/Nova_AI_Fusion_Memory_MCP",
      "disabled": false,
      "alwaysAllow": [
        "query_memory", "upsert_memory", "bulk_upsert_memory",
        "delete_memory", "create_checkpoint", "get_last_checkpoint",
        "get_recent_events", "get_session_events", "check_health"
      ]
    }
  }
}
```

### Claude Desktop

Settings file location:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "nova-memory": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "--network=nova-memory-mcp_nova_network",
        "--env-file", "/absolute/path/to/.env",
        "nova-memory-mcp:latest"
      ],
      "transportType": "stdio"
    }
  }
}
```

**Note:** Claude Desktop requires the absolute path to `.env` and the `--network` flag.

### Troubleshooting

1. Check logs: `docker logs nova_mcp_server`
2. Verify containers: `docker ps`
3. Common issues: missing `.env`, Docker network misconfiguration, stale containers

## 8. Testing

```bash
# Run full test suite (202 tests)
python3 -m pytest tests/test_sequence_service.py \
  tests/test_chronology_injection.py \
  tests/test_checkpoint.py \
  tests/test_temporal_retrieval.py \
  tests/test_temporal_router.py \
  tests/test_graph_session.py \
  tests/test_redis_timeline.py -v

# Install fakeredis for full Redis coverage (14 extra tests)
pip install fakeredis
```

| Test File | Phase | Tests |
|-----------|-------|-------|
| `test_sequence_service.py` | 1 — Monotonic counter | 24 |
| `test_chronology_injection.py` | 1 — Write-time injection | 23 |
| `test_checkpoint.py` | 2 — Session checkpoints | 28 |
| `test_temporal_retrieval.py` | 3 — Temporal retrieval | 38 |
| `test_temporal_router.py` | 4 — Query routing | 32 |
| `test_graph_session.py` | 5 — Graph time model | 28 |
| `test_redis_timeline.py` | 6 — Redis timeline | 29 |

## 9. Memory Governance

- `MEMORY_SCHEMA.md` — Canonical metadata contract for all durable writes
- `OPENCLAW_MEMORY_RULES.md` — Multi-agent write/read discipline

## 10. Future Work

- Authentication/authorization (MCP OAuth)
- LLM-based query intent classification (upgrade from keyword matching)
- Redis Streams for real-time event subscription
- Cross-session memory consolidation and compaction
- Enhanced monitoring and observability dashboards
