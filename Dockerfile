# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set the port the application will run on (though not directly used by stdio MCP)
ENV PORT 8000 
# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and tools
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Ensure pip is up-to-date and install requirements
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code needed by mcp_server.py
COPY ./app ./app
COPY .env.example .
COPY mcp_server.py .
COPY hybrid_merger.py .
COPY query_router.py .
COPY reranker.py .

COPY docker-mcp-entrypoint.sh .
RUN chmod +x docker-mcp-entrypoint.sh

# Use the entrypoint script to run the server in the background and keep container alive
ENTRYPOINT ["./docker-mcp-entrypoint.sh"]
# CMD is removed as ENTRYPOINT now handles the execution