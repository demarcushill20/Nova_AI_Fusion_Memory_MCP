# TASK.md - Nova AI Memory MCP Server Implementation Tasks

## Phase 1: Project Setup and Core Configuration (P1)

**Goal:** Establish the project structure, environment, dependencies, and basic configuration loading.

---

**Task T1: Initialize Project Structure** [COMPLETED - 2025-04-20]
- **Description:** Create the main project directory (`nova_memory_mcp/`) and the internal application structure (`app/`, `app/api/`, `app/services/`, etc.) as defined in `PLANNING.md` (Section: Project Structure) and `ARCHITECTURE.md` (Section 4 & 5).
- **Priority:** High
- **Estimated Time:** 1h
- **Difficulty:** Low
- **Dependencies:** None
- **Related Architectural Components:** Overall project structure.
- **Acceptance Criteria:** Directory structure matching the plan exists. Basic `__init__.py` files are created where necessary.

---

**Task T2: Setup Python Environment and Dependencies** [COMPLETED - 2025-04-20]
- **Description:** Create a virtual environment, install necessary Python packages (FastAPI, Uvicorn, OpenAI SDK, Pinecone client, Neo4j driver, LightRAG, python-dotenv, etc.), and generate `requirements.txt`.
- **Subtasks:**
    - T2.1: Create and activate Python virtual environment (`venv`).
    - T2.2: Install core dependencies using `pip`.
    - T2.3: Freeze dependencies into `requirements.txt`.
- **Priority:** High
- **Estimated Time:** 1.5h
- **Difficulty:** Low
- **Dependencies:** T1
- **Related Architectural Components:** Python Environment, Dependency Management.
- **Relevant Research/Planning Notes:** `RESEARCH2.md` (Section 1), `PLANNING.md` (requirements.txt).
- **Acceptance Criteria:** Virtual environment is active. All listed dependencies are installed. `requirements.txt` accurately reflects dependencies.

---

**Task T3: Implement Configuration Loading** [COMPLETED - 2025-04-20]
- **Description:** Create `app/config.py` to load settings (API keys, DB URIs, index names) from environment variables and a `.env` file using Pydantic `BaseSettings`. Define necessary configuration variables.
- **Subtasks:**
    - T3.1: Define Pydantic `Settings` class in `app/config.py`.
    - T3.2: Include fields for OpenAI, Pinecone (Key, Env, Index), Neo4j (URI, User, Pass, DB), and potentially Reranker model name.
    - T3.3: Configure `BaseSettings` to load from a `.env` file.
    - T3.4: Create a sample `.env.example` file listing required variables.
- **Priority:** High
- **Estimated Time:** 2h
- **Difficulty:** Medium
- **Dependencies:** T1, T2
- **Related Architectural Components:** Configuration (`app/config.py`).
- **Relevant Research/Planning Notes:** `PLANNING.md` (Section: Configuration), `ARCHITECTURE.md` (Section 5.10, 9.1).
- **Acceptance Criteria:** `app/config.py` exists and can load settings from environment variables or `.env`. A `.env.example` file is present.

---

**Task T4: Basic FastAPI Application Setup** [COMPLETED - 2025-04-20]
- **Description:** Create `app/main.py` to initialize the FastAPI application instance, include basic middleware (if any), and set up placeholder startup/shutdown events.
- **Priority:** High
- **Estimated Time:** 1h
- **Difficulty:** Low
- **Dependencies:** T1, T2, T3
- **Related Architectural Components:** FastAPI App (`app/main.py`).
- **Relevant Research/Planning Notes:** `RESEARCH2.md` (Section 3), `PLANNING.md` (app/main.py).
- **Acceptance Criteria:** `app/main.py` exists. Running `uvicorn app.main:app` starts the server without errors (though with no routes yet).

---

## Phase 2: Core Services and Integrations (P2)

**Goal:** Implement the core memory service logic, integrate Nova AI modules, and establish connections to external services (OpenAI, Pinecone, Neo4j).

---

**Task T5: Integrate Nova AI Memory Modules** [COMPLETED - 2025-04-20]
- **Description:** Integrate the existing Nova AI Python modules (`query_router.py`, `hybrid_merger.py`, `reranker.py`) into the project structure (e.g., copy into `app/services/` or install if packaged). Ensure they are importable.
- **Priority:** High
- **Estimated Time:** 1h
- **Difficulty:** Medium
- **Dependencies:** T1
- **Related Architectural Components:** Query Router, Hybrid Merger, Reranker.
- **Relevant Research/Planning Notes:** `PLANNING.md` (Section: Reusing Nova AI Memory Modules), `ARCHITECTURE.md` (Section 5.7, 5.8, 5.9).
- **Acceptance Criteria:** Nova AI modules are present in the project structure and can be imported without errors from within the `app` package.

---

**Task T6: Implement Embedding Service** [COMPLETED - 2025-04-20]
- **Description:** Create `app/services/embedding_service.py` to handle text embedding generation using the OpenAI API. Include LRU caching.
- **Subtasks:**
    - T6.1: Implement `get_embedding(text: str)` function using `openai.Embedding.create`.
    - T6.2: Load OpenAI API key from configuration (T3).
    - T6.3: Add `functools.lru_cache` to `get_embedding`.
- **Priority:** High
- **Estimated Time:** 2h
- **Difficulty:** Medium
- **Dependencies:** T3, T5 (uses config)
- **Related Architectural Components:** Embedding Service (`app/services/embedding_service.py`).
- **Relevant Research/Planning Notes:** `RESEARCH2.md` (Section 5), `ARCHITECTURE.md` (Section 5.6).
- **Acceptance Criteria:** Service can generate embeddings for given text using OpenAI. Caching is functional.
- **Test Status:**
  - Test Creation Date: 2025-04-20 (Developer Agent)
  - Test Execution Date: 2025-04-20
  - Test Results: ✅ PASSED (tests/test_embedding_service.py)
  - Debugging Activities: None needed.
  - Resolution Details: N/A

---

**Task T7: Implement Pinecone Client Service** [COMPLETED - 2025-04-20]
- **Description:** Create `app/services/pinecone_client.py` to manage interactions with Pinecone.
- **Subtasks:**
    - T7.1: Implement `initialize()` function to connect using config (T3) and get the `Index` object. Handle index creation if not exists (using configured dimension 1536, cosine metric).
    - T7.2: Implement `upsert_vector(id, vector, metadata)` function.
    - T7.3: Implement `query_vector(query_vector, top_k)` function.
    - T7.4: Implement `delete_vector(id)` function.
    - T7.5: Implement `ping()` or `check_connection()` for health checks.
- **Priority:** High
- **Estimated Time:** 3h
- **Difficulty:** Medium
- **Dependencies:** T3, T5 (uses config)
- **Related Architectural Components:** Pinecone Client (`app/services/pinecone_client.py`).
- **Relevant Research/Planning Notes:** `RESEARCH2.md` (Section 6), `PLANNING.md` (Section: Configuration), `ARCHITECTURE.md` (Section 5.4).
- **Acceptance Criteria:** Service can connect to Pinecone, perform upsert, query, delete operations, and check connection status.
- **Test Status:**
  - Test Creation Date: 2025-04-20 (Developer Agent)
  - Test Execution Date: 2025-04-20
  - Test Results: ✅ PASSED (tests/test_pinecone_client_integration.py)
  - Debugging Activities: Updated Pinecone package name in requirements.txt; Corrected index listing logic in pinecone_client.py; Increased indexing delay in integration test.
  - Resolution Details: Test passed after applying fixes and increasing delay.

---

**Task T8: Implement Graph Client Service (Neo4j/LightRAG)** [COMPLETED - 2025-04-20]
- **Description:** Create `app/services/graph_client.py` to manage interactions with Neo4j, potentially using LightRAG's `Neo4JStorage`.
- **Subtasks:**
    - T8.1: Implement `initialize()` function to connect using config (T3). Decide whether to use `Neo4JStorage` or the raw `neo4j` driver based on LightRAG integration feasibility.
    - T8.2: Implement `upsert_graph_data(id, content, metadata)` function (e.g., creating/merging `:Base` nodes).
    - T8.3: Implement `query_graph(query_text_or_embedding)` function to retrieve relevant graph data. Define the query strategy (e.g., based on LightRAG methods, embedding similarity if available, or keyword matching).
    - T8.4: Implement `delete_graph_data(id)` function (e.g., `MATCH (n:Base {id: $id}) DETACH DELETE n`).
    - T8.5: Implement `ping()` or `check_connection()` for health checks.
- **Priority:** High
- **Estimated Time:** 4h
- **Difficulty:** High (Requires understanding Neo4j/Cypher and potentially LightRAG internals)
- **Dependencies:** T3, T5, T6 (needs embedding function for init/query)
- **Related Architectural Components:** Graph Client (`app/services/graph_client.py`).
- **Relevant Research/Planning Notes:** `RESEARCH2.md` (Section 7), `PLANNING.md` (Section: Reusing Nova AI Memory Modules, Configuration), `ARCHITECTURE.md` (Section 5.5).
- **Acceptance Criteria:** Service can connect to Neo4j, perform upsert, query, delete operations based on the chosen strategy, and check connection status.

---

**Task T9: Implement Core Memory Service** [COMPLETED - 2025-04-20]
- **Description:** Create `app/services/memory_service.py` to orchestrate the memory operations by integrating all underlying components.
- **Subtasks:**
    - T9.1: Initialize instances of `EmbeddingService`, `PineconeClient`, `GraphClient`, `QueryRouter`, `HybridMerger`, `Reranker` upon service instantiation, passing necessary configurations.
    - T9.2: Implement `perform_query(query_text)` method:
        - Call `EmbeddingService` for query embedding.
        - Call `QueryRouter` (optional logging).
        - Call `PineconeClient.query_vector`.
        - Call `GraphClient.query_graph`.
        - Call `HybridMerger.merge_results`.
        - Call `Reranker.rerank`.
        - Format and return final results.
    - T9.3: Implement `perform_upsert(id, content, metadata)` method:
        - Determine/generate ID (e.g., MD5 hash).
        - Call `EmbeddingService` for content embedding.
        - Call `PineconeClient.upsert_vector`.
        - Call `GraphClient.upsert_graph_data`.
        - Handle potential errors/inconsistencies between the two upserts.
    - T9.4: Implement `perform_delete(id)` method:
        - Call `PineconeClient.delete_vector`.
        - Call `GraphClient.delete_graph_data`.
        - Handle potential errors.
    - T9.5: Implement helper methods for health checks using client pings.
- **Priority:** High
- **Estimated Time:** 6h
- **Difficulty:** High
- **Dependencies:** T3, T5, T6, T7, T8
- **Related Architectural Components:** Memory Service (`app/services/memory_service.py`).
- **Relevant Research/Planning Notes:** `PLANNING.md` (Section: FastAPI API Endpoints), `ARCHITECTURE.md` (Section 5.3, 6.2, 6.3).
- **Acceptance Criteria:** `MemoryService` class exists. Methods correctly orchestrate calls to underlying services and Nova modules for query, upsert, and delete operations according to the defined pipeline.

---

## Phase 3: API Endpoint Implementation (P3)

**Goal:** Expose the core memory service functionality via FastAPI REST endpoints.

---

**Task T10: Define API Data Models** [COMPLETED - 2025-04-20]
- **Description:** Create `app/models.py` with Pydantic models for API request and response validation.
- **Subtasks:**
    - T10.1: Define `QueryRequest` (query: str).
    - T10.2: Define `MemoryItem` (id: str, text: str, source: str, score: float, metadata: Optional[dict]).
    - T10.3: Define `QueryResponse` (results: List[MemoryItem]).
    - T10.4: Define `UpsertRequest` (id: Optional[str], content: str, metadata: Optional[dict]).
    - T10.5: Define `UpsertResponse` (id: str, status: str).
    - T10.6: Define `DeleteResponse` (id: str, status: str).
- **Priority:** High
- **Estimated Time:** 1.5h
- **Difficulty:** Low
- **Dependencies:** T1
- **Related Architectural Components:** Models (`app/models.py`).
- **Relevant Research/Planning Notes:** `RESEARCH.md` (Section: API Input/Output), `RESEARCH2.md` (Section 4), `ARCHITECTURE.md` (Section 5.11, 7).
- **Acceptance Criteria:** Pydantic models matching the API specification are defined in `app/models.py`.

---

**Task T11: Implement API Routes** [COMPLETED - 2025-04-20]
- **Description:** Create `app/api/memory_routes.py` and implement the FastAPI routes using the `MemoryService`.
- **Subtasks:**
    - T11.1: Create an `APIRouter` instance.
    - T11.2: Implement `POST /query` endpoint:
        - Inject `MemoryService`.
        - Validate request using `QueryRequest` model.
        - Call `memory_service.perform_query`.
        - Return response using `QueryResponse` model.
    - T11.3: Implement `POST /upsert` endpoint:
        - Inject `MemoryService`.
        - Validate request using `UpsertRequest` model.
        - Call `memory_service.perform_upsert`.
        - Return response using `UpsertResponse` model.
    - T11.4: Implement `DELETE /memory/{id}` endpoint:
        - Inject `MemoryService`.
        - Get `id` from path parameter.
        - Call `memory_service.perform_delete`.
        - Return response using `DeleteResponse` model.
    - T11.5: Implement `GET /health` endpoint:
        - Inject `MemoryService`.
        - Call health check helper in `MemoryService`.
        - Return appropriate status and details.
    - T11.6: Include the router in `app/main.py`.
- **Priority:** High
- **Estimated Time:** 4h
- **Difficulty:** Medium
- **Dependencies:** T4, T9, T10
- **Related Architectural Components:** API Router (`app/api/memory_routes.py`), FastAPI App (`app/main.py`).
- **Relevant Research/Planning Notes:** `PLANNING.md` (Section: FastAPI API Endpoints), `ARCHITECTURE.md` (Section 5.2, 7).
- **Acceptance Criteria:** All API endpoints (`/query`, `/upsert`, `/memory/{id}`, `/health`) are implemented and functional when tested locally. Requests and responses match the defined Pydantic models.

---

## Phase 4: Testing and Documentation (P4)

**Goal:** Ensure the server functions correctly and is well-documented.

---

**Task T12: Implement Unit and Integration Tests** [COMPLETED - 2025-04-20]
- **Description:** Write tests for core services and integrations (using test harness scripts per custom instructions).
- **Subtasks:**
    - T12.1: Write unit tests for `EmbeddingService` (mocking OpenAI API).
    - T12.2: Write unit tests for `MemoryService` logic (mocking clients and Nova modules).
    - T12.3: Write integration tests for `PineconeClient` and `GraphClient` (potentially against local/test instances or mocks).
- **Priority:** Medium
- **Estimated Time:** 6h
- **Difficulty:** Medium
- **Dependencies:** T6, T7, T8, T9
- **Related Architectural Components:** Testing framework.
- **Relevant Research/Planning Notes:** `ARCHITECTURE.md` (Section 14).
- **Acceptance Criteria:** Unit and integration tests cover critical service logic. Tests pass reliably.

---

**Task T13: Implement API Tests** [COMPLETED - 2025-04-20]
- **Description:** Write tests for the FastAPI endpoints using `httpx` against a running server with mocked service layer (using test harness script per custom instructions).
- **Subtasks:**
    - T13.1: Set up `pytest` and `TestClient`.
    - T13.2: Write tests for the `/query` endpoint with various inputs.
    - T13.3: Write tests for the `/upsert` endpoint (including ID generation/update).
    - T13.4: Write tests for the `/memory/{id}` endpoint (including deleting existing/non-existing IDs).
    - T13.5: Write tests for the `/health` endpoint (checking success/failure scenarios).
- **Priority:** High
- **Estimated Time:** 4h
- **Difficulty:** Medium
- **Dependencies:** T11
- **Related Architectural Components:** API Router, Testing framework.
- **Relevant Research/Planning Notes:** `ARCHITECTURE.md` (Section 14).
- **Acceptance Criteria:** API tests cover all endpoints and common scenarios. Tests pass reliably against a running test instance.

---

**Task T14: Create Project Documentation** [COMPLETED - 2025-04-20]
- **Description:** Write a `README.md` file explaining setup, configuration, running the server locally, and API usage.
- **Subtasks:**
    - T14.1: Document project purpose and architecture overview.
    - T14.2: Provide instructions for setting up the environment (`venv`, `requirements.txt`).
    - T14.3: Explain configuration using `.env` file and list required variables.
    - T14.4: Detail how to run the server locally (Uvicorn, Docker Compose).
    - T14.5: Document API endpoints with request/response examples (or point to auto-generated Swagger docs).
- **Priority:** Medium
- **Estimated Time:** 3h
- **Difficulty:** Low
- **Dependencies:** T1, T2, T3, T11
- **Related Architectural Components:** Documentation (`README.md`).
- **Acceptance Criteria:** `README.md` is comprehensive, clear, and accurate.

---

## Phase 5: Containerization and Deployment Prep (P5)

**Goal:** Prepare the application for deployment using Docker.

---

**Task T15: Create Dockerfile** [COMPLETED - 2025-04-20]
- **Description:** Write a `Dockerfile` to containerize the FastAPI application.
- **Subtasks:**
    - T15.1: Choose a suitable Python base image (e.g., `python:3.10-slim`).
    - T15.2: Set up working directory, copy necessary files (`requirements.txt`, `app/`, `main.py`).
    - T15.3: Install dependencies from `requirements.txt`.
    - T15.4: Expose the application port (e.g., 8000).
    - T15.5: Define the `CMD` to run Uvicorn.
- **Priority:** Medium
- **Estimated Time:** 2h
- **Difficulty:** Medium
- **Dependencies:** T2, T4
- **Related Architectural Components:** Deployment Architecture, Dockerfile.
- **Relevant Research/Planning Notes:** `PLANNING.md` (Section: Development and Deployment), `ARCHITECTURE.md` (Section 9).
- **Acceptance Criteria:** `Dockerfile` exists and can successfully build a container image using `docker build`.

---

**Task T16: Create Docker Compose for Local Development** [COMPLETED - 2025-04-20]
- **Description:** Create `docker-compose.yml` to orchestrate the application container and a local Neo4j container for easy local testing.
- **Subtasks:**
    - T16.1: Define a service for the FastAPI application (`nova-ai`) using the `Dockerfile` (T15). Map ports and potentially mount volumes for hot-reloading if desired. Configure environment variables from `.env`.
    - T16.2: Define a service for Neo4j (`neo4j`) using the official image. Configure ports, volumes for data persistence, and environment variables for authentication (matching `.env`).
    - T16.3: Configure dependencies (e.g., app depends on neo4j being healthy).
- **Priority:** Medium
- **Estimated Time:** 3h
- **Difficulty:** Medium
- **Dependencies:** T3, T15
- **Related Architectural Components:** Deployment Architecture, Docker Compose.
- **Relevant Research/Planning Notes:** `PLANNING.md` (Section: Development and Deployment), `ARCHITECTURE.md` (Section 9.1).
- **Acceptance Criteria:** `docker-compose.yml` exists. Running `docker-compose up` successfully starts the application and Neo4j containers. The application container can connect to the Neo4j container.

---