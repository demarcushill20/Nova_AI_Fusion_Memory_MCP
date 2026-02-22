# VERIFY_GITHUB_RAW

This document proves that GitHub serves multiline, parse-valid files for the pinned commit and that local validation checks pass.

## Pinned Commit

```bash
git fetch origin
git rev-parse origin/main
```

```text
28dcaba80116d4e3bdfb28e2193c1f751b7416d7
```

## Raw Output: mcp_server.py

```bash
curl -L https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/28dcaba80116d4e3bdfb28e2193c1f751b7416d7/mcp_server.py | nl -ba | sed -n '1,30p'
```

```text
     1	import asyncio
     2	import logging
     3	import sys
     4	from collections.abc import AsyncIterator
     5	from contextlib import asynccontextmanager
     6	from dataclasses import dataclass  # Import dataclass
     7	from typing import Any, Dict, List, Optional
     8	
     9	from mcp.server.fastmcp import FastMCP, Context
    10	
    11	# ToolParam is not needed; FastMCP uses type hints directly
    12	
    13	# Assuming your MemoryService and config are structured appropriately
    14	# Adjust imports based on your actual project structure
    15	from app.services.memory_service import MemoryService
    16	
    17	# Configure logging
    18	logging.basicConfig(
    19	    level=logging.INFO,
    20	    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    21	    handlers=[logging.StreamHandler(sys.stderr)],
    22	)
    23	logger = logging.getLogger("nova-memory-mcp-server")
    24	
    25	# Global instance (or manage via lifespan context)
    26	# Using a global instance might be simpler if lifespan context proves tricky
    27	# memory_service_instance: Optional[MemoryService] = None
    28	
    29	
    30	# Define a context structure for type hinting if needed
```

```bash
git rev-parse 28dcaba80116d4e3bdfb28e2193c1f751b7416d7:mcp_server.py
git hash-object <downloaded_raw_bytes_for_mcp_server.py>
```

```text
git blob: 973b2bb45915eb64c8cfa3eb6579094f9993ce69
raw blob: 973b2bb45915eb64c8cfa3eb6579094f9993ce69
status: MATCH
```

## Raw Output: app/config.py

```bash
curl -L https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/28dcaba80116d4e3bdfb28e2193c1f751b7416d7/app/config.py | nl -ba | sed -n '1,30p'
```

```text
     1	from typing import Optional
     2	
     3	from pydantic import model_validator
     4	from pydantic_settings import BaseSettings, SettingsConfigDict
     5	
     6	
     7	class Settings(BaseSettings):
     8	    """
     9	    Application settings loaded from environment variables or .env file.
    10	    """
    11	
    12	    model_config = SettingsConfigDict(
    13	        env_file=".env",
    14	        env_file_encoding="utf-8",
    15	        case_sensitive=False,
    16	        extra="ignore",
    17	    )
    18	
    19	    # OpenAI Configuration
    20	    OPENAI_API_KEY: Optional[str] = None
    21	
    22	    # Pinecone Configuration
    23	    PINECONE_API_KEY: str
    24	    PINECONE_ENV: str
    25	    PINECONE_INDEX: str = "nova-ai-memory"  # Default index name from Nova_AI.py
    26	
    27	    # Neo4j Configuration
    28	    NEO4J_URI: str = "bolt://neo4j:7687"  # Default Docker Compose service URI
    29	    NEO4J_USER: str = "neo4j"
    30	    NEO4J_PASSWORD: Optional[str] = None  # Optional when Neo4j auth is disabled
```

```bash
git rev-parse 28dcaba80116d4e3bdfb28e2193c1f751b7416d7:app/config.py
git hash-object <downloaded_raw_bytes_for_app/config.py>
```

```text
git blob: 5a2b15e34b9594729c69bbce8c383f395c985dd6
raw blob: 5a2b15e34b9594729c69bbce8c383f395c985dd6
status: MATCH
```

## Raw Output: docker-compose.yml

```bash
curl -L https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/28dcaba80116d4e3bdfb28e2193c1f751b7416d7/docker-compose.yml | nl -ba | sed -n '1,80p'
```

```text
     1	services:
     2	  neo4j:
     3	    image: neo4j:5.19 # Use a specific stable version
     4	    container_name: nova_neo4j_db # Renamed container to avoid conflicts
     5	    ports:
     6	      - "7474:7474" # HTTP interface
     7	      - "7687:7687" # Bolt interface (for driver connection)
     8	    volumes:
     9	      - neo4j_data:/data # Persist Neo4j data
    10	      # Optional: Mount directories for plugins or logs if needed
    11	      # - ./neo4j/plugins:/plugins
    12	      # - ./neo4j/logs:/logs
    13	    environment:
    14	      # Authentication disabled for local Docker usage
    15	      NEO4J_AUTH: none
    16	      # Optional: Configure memory limits if needed
    17	      # NEO4J_server_memory_pagecache_size: 1G
    18	      # NEO4J_server_memory_heap_initial__size: 1G
    19	      # NEO4J_server_memory_heap_max__size: 2G
    20	    # Healthcheck removed as it was causing startup issues
    21	    restart: unless-stopped
    22	    networks:
    23	      nova_network:
    24	        aliases:
    25	          - neo4j_db
    26	
    27	  # Combined MCP Server (using mcp-python-sdk)
    28	  # This service now directly handles MCP communication via stdio
    29	  nova-memory:
    30	    image: nova-memory-mcp:latest # Stable tag independent of compose project naming
    31	    build:
    32	      context: .
    33	      dockerfile: Dockerfile # Uses the updated Dockerfile running mcp_server.py
    34	    container_name: nova_mcp_server # Keep container name consistent for now
    35	    stdin_open: true # Keep STDIN open for the MCP SDK server
    36	    env_file:
    37	      - .env # Load environment variables for mcp_server.py (API keys etc.)
    38	    # No RUNTIME_MODE needed
    39	    # No ports needed for stdio MCP
    40	    depends_on: # Still depends on Neo4j being ready
    41	      neo4j:
    42	        condition: service_started
    43	    restart: unless-stopped
    44	    # Keep the profile so it's not started by default `docker-compose up`
    45	    profiles:
    46	      - mcp
    47	    networks:
    48	      nova_network:
    49	        aliases:
    50	          - nova_memory_mcp # Alias for potential internal communication if needed
    51	
    52	# Define networks for container communication
    53	networks:
    54	  nova_network:
    55	    driver: bridge
    56	
    57	volumes:
    58	  neo4j_data: # Define the named volume for Neo4j data persistence
```

```bash
git rev-parse 28dcaba80116d4e3bdfb28e2193c1f751b7416d7:docker-compose.yml
git hash-object <downloaded_raw_bytes_for_docker-compose.yml>
```

```text
git blob: 353a860ba6388287cb17e0f61e5ccbfcb910b115
raw blob: 353a860ba6388287cb17e0f61e5ccbfcb910b115
status: MATCH
```

## Local Validation Commands

```bash
python -m py_compile mcp_server.py app/config.py
docker compose -f docker-compose.yml config
```

```text
py_compile exit code: 0
docker compose config exit code: 0
docker compose config (first 80 lines):
name: nova_ai_fusion_memory_mcp
services:
  neo4j:
    container_name: nova_neo4j_db
    environment:
      NEO4J_AUTH: none
    image: neo4j:5.19
    networks:
      nova_network:
        aliases:
          - neo4j_db
    ports:
      - mode: ingress
        target: 7474
        published: "7474"
        protocol: tcp
      - mode: ingress
        target: 7687
        published: "7687"
        protocol: tcp
    restart: unless-stopped
    volumes:
      - type: volume
        source: neo4j_data
        target: /data
        volume: {}
networks:
  nova_network:
    name: nova_ai_fusion_memory_mcp_nova_network
    driver: bridge
volumes:
  neo4j_data:
    name: nova_ai_fusion_memory_mcp_neo4j_data
```

## Conclusion

All pinned raw files are multiline and hash-match Git blobs at commit 28dcaba80116d4e3bdfb28e2193c1f751b7416d7. Python compile checks and Docker Compose config checks pass.
