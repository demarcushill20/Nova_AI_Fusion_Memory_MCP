import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import configuration first to ensure it's loaded
try:
    from .config import settings
except ImportError:
    print("Error: Could not import settings from app.config. Ensure the file exists and is configured.")
    # Handle the error appropriately, maybe exit or use defaults
    # For now, re-raise to make the issue clear during development
    raise

# Import API routers and the shared service instance
from .api import memory_routes
from .api.memory_routes import memory_service_instance # Import the shared instance
from .observability.metrics import REGISTRY as METRICS_REGISTRY

from prometheus_client import make_asgi_app

logger = logging.getLogger(__name__)

# --- Application Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle application startup and shutdown logic.
    """
    print("--- Starting Application ---")
    logger.info("Application startup: Initializing resources...")

    # Initialize the shared MemoryService instance
    logger.info("Initializing shared MemoryService...")
    await memory_service_instance.initialize()
    logger.info("Shared MemoryService initialized.")

    # Remove the placeholder print now that real initialization happens
    # print("--- Application Resources Initialized (Placeholders) ---")

    yield # Application runs here

    print("--- Shutting Down Application ---")
    logger.info("Application shutdown: Cleaning up resources...")

    # Close the shared MemoryService instance resources
    logger.info("Closing shared MemoryService resources...")
    await memory_service_instance.close()
    logger.info("Shared MemoryService resources closed.")

    print("--- Application Shutdown Complete ---")


# --- FastAPI Application Instance ---
app = FastAPI(
    title="Nova AI Memory MCP Server",
    description="Provides a REST API for interacting with Nova AI's fused memory system (Pinecone + Neo4j).",
    version="0.1.0",
    lifespan=lifespan # Use the lifespan context manager
)

# --- Include API Routers ---
# Include the memory router (prefix '/memory' is defined within the router itself)
app.include_router(memory_routes.router)

# --- Prometheus metrics endpoint (PLAN-0759 Phase 8b, Sprint 18) ---
# The /metrics endpoint has no auth middleware. Exposure is gated at the
# network boundary: the process MUST bind to 127.0.0.1 (never 0.0.0.0).
# Scrape config runs on the host. If a reverse proxy fronts this service,
# the /metrics path MUST be excluded from the public route or allowlisted
# to the scrape host only. The direct-launch block below defaults to
# 127.0.0.1; production deploys under uvicorn/systemd MUST pass --host
# 127.0.0.1 explicitly (see runbook).
app.mount("/metrics", make_asgi_app(registry=METRICS_REGISTRY))

# --- Root Endpoint (Optional) ---
@app.get("/", tags=["General"])
async def read_root():
    """
    Root endpoint providing basic information about the API.
    """
    return {
        "message": "Welcome to the Nova AI Memory MCP Server API",
        "version": app.version,
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# Placeholder for other potential middleware or configurations

if __name__ == "__main__":
    # Direct-launch debug path: `python app/main.py`. Binds to 127.0.0.1
    # only — /metrics has no auth, so external exposure would leak
    # internal SLO / latency signals. Override via METRICS_HOST env var
    # if you know what you're doing (e.g., a container network where the
    # only reachable callers are sibling services behind an allowlist).
    import os
    import uvicorn
    host = os.environ.get("METRICS_HOST", "127.0.0.1")
    port = int(os.environ.get("METRICS_PORT", "8000"))
    print(f"Running FastAPI app directly on {host}:{port} (debug mode)...")
    uvicorn.run(app, host=host, port=port)