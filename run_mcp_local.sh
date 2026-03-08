#!/bin/bash
# Run MCP server locally (outside Docker), pointing to Docker services on localhost
cd /home/nova/Nova_AI_Fusion_Memory_MCP

# Load vars from .env
set -a
source .env
set +a

# Override Docker-internal hostnames with localhost
export NEO4J_URI="bolt://localhost:7687"
export REDIS_URL="redis://localhost:6379/0"
export EVENT_SEQ_FILE="/home/nova/Nova_AI_Fusion_Memory_MCP/data/event_seq.counter"

exec python3 mcp_server.py
