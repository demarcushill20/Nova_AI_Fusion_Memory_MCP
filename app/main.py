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

# Import API routers
from .api import memory_routes

logger = logging.getLogger(__name__)

# --- Application Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle application startup and shutdown logic.
    """
    print("--- Starting Application ---")
    logger.info("Application startup: Initializing resources...")

    # Placeholder: Initialize database connections, load models, etc.
    # Example: await initialize_database_connections()
    # Example: await load_machine_learning_models()
    # These will be implemented in detail within service modules later (T7, T8, T9)

    print("--- Application Resources Initialized (Placeholders) ---")
    yield # Application runs here
    print("--- Shutting Down Application ---")
    logger.info("Application shutdown: Cleaning up resources...")

    # Placeholder: Clean up resources, close connections
    # Example: await close_database_connections()

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
    # This block allows running the app directly using `python app/main.py`
    # However, it's more common to run using Uvicorn: `uvicorn app.main:app --reload`
    import uvicorn
    print("Running FastAPI app directly using Uvicorn (for debugging)...")
    # Note: Configuration loading happens when config.py is imported.
    # Ensure .env file is present or environment variables are set.
    uvicorn.run(app, host="0.0.0.0", port=8000)