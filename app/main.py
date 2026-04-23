import logging
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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

# --- /metrics loopback enforcement (PLAN-0759 Sprint 21 Phase 3) ---
# Defense-in-depth: request-layer loopback gate that enforces the same
# policy the socket bind does, so a misconfigured bind (0.0.0.0) cannot
# silently expose internal SLO/latency signals.
_METRICS_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_METRICS_ALLOW_PUBLIC = os.environ.get("METRICS_ALLOW_PUBLIC", "").strip().lower() == "1"
_METRICS_REJECTION_LOG_LIMIT = 10
_metrics_rejection_count = 0

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


# --- /metrics loopback enforcement middleware (PLAN-0759 Sprint 21 Phase 3) ---
# Applied before the /metrics ASGI mount so non-loopback requests to
# /metrics are rejected at the request layer regardless of socket bind.
@app.middleware("http")
async def _metrics_loopback_guard(request: Request, call_next):
    path = request.url.path
    if path == "/metrics" or path.startswith("/metrics/"):
        # Defensive: request.client can be None in test/ASGI-internal paths;
        # treat that as local so the harness and internal probes work.
        client = request.client
        client_host = client.host if client is not None else None
        allow = (
            client_host is None
            or client_host.lower() in _METRICS_LOOPBACK_HOSTS
            or _METRICS_ALLOW_PUBLIC
        )
        if not allow:
            global _metrics_rejection_count
            _metrics_rejection_count += 1
            if _metrics_rejection_count <= _METRICS_REJECTION_LOG_LIMIT:
                logger.warning(
                    "Rejected non-loopback /metrics request from %s (count=%d)",
                    client_host,
                    _metrics_rejection_count,
                )
            else:
                logger.debug(
                    "Rejected non-loopback /metrics request from %s (count=%d)",
                    client_host,
                    _metrics_rejection_count,
                )
            return JSONResponse(
                status_code=403,
                content={"error": "metrics endpoint not accessible from this client"},
            )
    return await call_next(request)


# --- Prometheus metrics endpoint (PLAN-0759 Phase 8b, Sprint 18; guarded Sprint 21 Phase 3) ---
# Defense in depth:
#  (1) Request layer — the `_metrics_loopback_guard` middleware above
#      rejects non-loopback clients with 403 unless METRICS_ALLOW_PUBLIC=1.
#  (2) Socket layer — the process MUST bind to 127.0.0.1 (never 0.0.0.0).
# Either control alone would contain accidental exposure; both together
# mean a bind misconfiguration still cannot leak internal SLO/latency
# signals. Scrape config runs on the host. If a reverse proxy fronts
# this service, the /metrics path MUST still be excluded from the public
# route or allowlisted to the scrape host only. The direct-launch block
# below defaults to 127.0.0.1; production deploys under uvicorn/systemd
# MUST pass --host 127.0.0.1 explicitly (see runbook).
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