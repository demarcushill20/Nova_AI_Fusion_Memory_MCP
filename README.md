# Nova AI Fusion Memory MCP Server

A Model Context Protocol (MCP) server providing AI agents with fused persistent memory: semantic vector search (Pinecone), knowledge graph traversal (Neo4j), and chronological timeline indexing (Redis). Built with `FastMCP` (mcp-python-sdk).

## Architecture

```
                         ┌──────────────────────┐
                         │    MCP Clients        │
                         │ (Claude Code, Claude  │
                         │  Desktop, Roo, etc.)  │
                         └──────────┬───────────┘
                                    │ stdio / JSON-RPC
                         ┌──────────▼───────────┐
                         │    mcp_server.py      │
                         │    (FastMCP v1.26)    │
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
        │ Pinecone  │   │  Neo4j   │ │  Redis   │  │ OpenAI   │
        │ (vectors) │   │ (graph)  │ │(timeline)│  │(embed/3s)│
        └───────────┘   └──────────┘ └──────────┘  └──────────┘
```

### Service Layer (`app/services/`)

| Service | Purpose |
|---------|---------|
| `MemoryService` | Orchestrates all memory operations — upsert, query, checkpoint, temporal retrieval |
| `PineconeClient` | Semantic vector storage and retrieval (text-embedding-3-small, 1536 dims) |
| `GraphClient` | Neo4j knowledge graph — Session nodes, FOLLOWS chains, INCLUDES edges |
| `RedisTimeline` | Sorted set timeline index for O(log N) chronological queries |
| `SequenceService` | Monotonic event counter — Redis INCR primary, file-based fallback |
| `EmbeddingService` | Text embeddings via OpenAI (text-embedding-3-small) |
| `QueryRouter` | Routes queries to VECTOR, GRAPH, HYBRID, TEMPORAL, or TEMPORAL_SEMANTIC modes |
| `HybridMerger` | Reciprocal Rank Fusion (RRF) across retrieval backends |
| `Reranker` | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) |

## MCP Tools

| Tool | Description |
|------|-------------|
| `upsert_memory` | Store or update a memory item with auto-assigned `event_seq` and embeddings |
| `bulk_upsert_memory` | Batch store multiple items with consecutive `event_seq` allocation |
| `query_memory` | Semantic/temporal/hybrid query with intelligent routing and reranking |
| `delete_memory` | Delete a memory item by ID from all backends |
| `create_checkpoint` | Create a session checkpoint with `event_seq` snapshot and graph links |
| `get_last_checkpoint` | Retrieve the most recent checkpoint (Redis O(1) primary, Pinecone fallback) |
| `get_recent_events` | Get N most recent events by `event_seq` (Redis sorted set) |
| `get_session_events` | Get all events linked to a session via Neo4j graph traversal |
| `check_health` | Health check across all backends (Pinecone, Neo4j, Redis, Reranker) |

## Chronological Memory System

Every memory write receives a monotonic `event_seq` and ISO 8601 `event_time`, providing strict "what happened after what" ordering independent of clock drift or multi-agent interleaving.

### Phase 1 — Write-Time Chronology Enforcement
- Every `upsert_memory` and `bulk_upsert_memory` injects `event_seq`, `event_time`, and `memory_type`
- `event_seq` is system-assigned (caller cannot override), strictly monotonic
- `memory_type` defaults to `"scratch"` if not provided

### Phase 2 — Session Checkpoint System
- `create_checkpoint` stores session boundaries (`memory_type: "checkpoint"`)
- Each checkpoint snapshots the current `event_seq` as `last_event_seq`
- `get_last_checkpoint` retrieves the most recent checkpoint via Redis O(1) lookup

### Phase 3 — Temporal Retrieval
- `get_recent_events` returns the N most recent events ordered by `event_seq` DESC
- Supports filtering by project, thread_id, memory_type, since_seq, since_time
- Redis primary path: O(log N) via `ZREVRANGE` / `ZRANGEBYSCORE`
- Pinecone fallback: over-fetch with 5x factor, client-side sort

### Phase 4 — Temporal-First Query Router
- Detects recency-intent keywords ("last session", "most recent", "where were we")
- **TEMPORAL** mode: pure recency, no semantic similarity
- **TEMPORAL_SEMANTIC** mode: temporal window first, then semantic refinement
- Temporal detection takes priority over HYBRID

### Phase 5 — Graph Time Model
- Neo4j Session nodes with `session_id`, `started_at`, `last_event_seq`
- `INCLUDES` edges: Session -> MemoryItem (auto-created on upsert when `session_id` is provided)
- `FOLLOWS` edges: Session -> PreviousSession (temporal chain)
- `get_session_events` retrieves all events for a session via graph traversal

### Phase 6 — Redis Timeline Store
- Redis sorted sets (`nova:timeline:{scope}`) with `event_seq` as score
- O(log N) recency queries via `ZREVRANGE`, replacing Pinecone dummy-vector over-fetch
- Dedicated checkpoint index (`nova:checkpoints:{scope}`) with O(1) latest lookup
- Dual-scope: project-specific timelines + automatic global aggregation
- `SequenceService` dual-backend: Redis `INCR` primary with file-based fallback
- Bidirectional counter sync on startup, graceful degradation on Redis failure

## Setup

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for Neo4j and Redis)
- OpenAI API key (for embeddings)
- Pinecone API key and environment (for vector storage)

### Environment Variables

```bash
cp .env.example .env
```

Required in `.env`:
| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for text-embedding-3-small |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_ENV` | Pinecone environment (e.g. `aped-4627-b74a`) |
| `PINECONE_INDEX` | Pinecone index name (must be 1536 dims, cosine) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` (default) |
| `NEO4J_URI` | Neo4j bolt URI (default: `bolt://neo4j:7687`) |
| `REDIS_URL` | Redis URL (default: `redis://redis_db:6379/0`) |
| `REDIS_ENABLED` | Enable Redis timeline (default: `true`) |

### Start Backend Services

```bash
# Start Neo4j and Redis (Docker Compose)
docker compose up -d neo4j_db redis_db

# Verify
docker ps
# Expected: nova_neo4j_db (ports 7474, 7687), nova_redis (port 6379)
```

### Run the MCP Server

**Option A: Docker (all-in-one)**
```bash
docker compose --profile mcp up -d
```

**Option B: Local (connect to Docker backends)**
```bash
pip install -r requirements.txt
bash run_mcp_local.sh
```

## Connecting MCP Clients

### Claude Code

Add to `~/.claude/settings.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "nova-memory": {
      "command": "bash",
      "args": ["/path/to/Nova_AI_Fusion_Memory_MCP/run_mcp_local.sh"],
      "env": {}
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

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

### Roo (VS Code)

Add to MCP settings:

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

### Troubleshooting

- **MCP connection fails**: All logging must go to stderr. If you see garbled JSON-RPC responses, check that `logging.basicConfig` runs BEFORE importing app modules in `mcp_server.py`.
- **Containers not starting**: `docker compose up -d neo4j_db redis_db` then verify with `docker ps`.
- **Pinecone dimension mismatch**: Index must be 1536 dims with cosine metric for text-embedding-3-small.

## Testing

```bash
# Run full test suite (202 tests)
python3 -m pytest tests/ -v

# Individual phases
python3 -m pytest tests/test_sequence_service.py -v       # Phase 1: monotonic counter
python3 -m pytest tests/test_chronology_injection.py -v   # Phase 1: write-time injection
python3 -m pytest tests/test_checkpoint.py -v             # Phase 2: session checkpoints
python3 -m pytest tests/test_temporal_retrieval.py -v     # Phase 3: temporal retrieval
python3 -m pytest tests/test_temporal_router.py -v        # Phase 4: query routing
python3 -m pytest tests/test_graph_session.py -v          # Phase 5: graph time model
python3 -m pytest tests/test_redis_timeline.py -v         # Phase 6: Redis timeline

# Install fakeredis for full Redis coverage
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

## Memory Governance

- `MEMORY_SCHEMA.md` — Canonical metadata contract for all durable writes
- `OPENCLAW_MEMORY_RULES.md` — Multi-agent write/read discipline

## Future Work

- Authentication/authorization (MCP OAuth)
- LLM-based query intent classification (upgrade from keyword matching)
- Redis Streams for real-time event subscription
- Cross-session memory consolidation and compaction
- Enhanced monitoring and observability dashboards
