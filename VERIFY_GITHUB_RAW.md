# VERIFY_GITHUB_RAW

This report is generated from commit-pinned GitHub raw URLs and local
validation commands. It is saved as UTF-8 without BOM with LF newlines.

## Pinned Commit

```bash
git fetch origin
git rev-parse origin/main
```

```text
15d711cdc0d883ab95655f04aaf58b10a00ddb5a
```

## Raw Output: mcp_server.py

```bash
curl -L https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/15d711cdc0d883ab95655f04aaf58b10a00ddb5a/mcp_server.py | nl -ba | sed -n '1,20p'
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
```

```bash
git rev-parse 15d711cdc0d883ab95655f04aaf58b10a00ddb5a:mcp_server.py
git hash-object <downloaded_raw_bytes_for_mcp_server.py>
```

```text
git blob: 973b2bb45915eb64c8cfa3eb6579094f9993ce69
raw blob: 973b2bb45915eb64c8cfa3eb6579094f9993ce69
status: MATCH
```

## Raw Output: app/config.py

```bash
curl -L https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/15d711cdc0d883ab95655f04aaf58b10a00ddb5a/app/config.py | nl -ba | sed -n '1,20p'
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
```

```bash
git rev-parse 15d711cdc0d883ab95655f04aaf58b10a00ddb5a:app/config.py
git hash-object <downloaded_raw_bytes_for_app/config.py>
```

```text
git blob: 5a2b15e34b9594729c69bbce8c383f395c985dd6
raw blob: 5a2b15e34b9594729c69bbce8c383f395c985dd6
status: MATCH
```

## Raw Output: docker-compose.yml

```bash
curl -L https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/15d711cdc0d883ab95655f04aaf58b10a00ddb5a/docker-compose.yml | nl -ba | sed -n '1,40p'
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
```

```bash
git rev-parse 15d711cdc0d883ab95655f04aaf58b10a00ddb5a:docker-compose.yml
git hash-object <downloaded_raw_bytes_for_docker-compose.yml>
```

```text
git blob: 353a860ba6388287cb17e0f61e5ccbfcb910b115
raw blob: 353a860ba6388287cb17e0f61e5ccbfcb910b115
status: MATCH
```

## UTF-8 BOM Check

```bash
python - << 'PY'
from pathlib import Path
import urllib.request
sha = '15d711cdc0d883ab95655f04aaf58b10a00ddb5a'
raw_main = urllib.request.urlopen('https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/main/VERIFY_GITHUB_RAW.md').read()
raw_sha = urllib.request.urlopen(f'https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/{sha}/VERIFY_GITHUB_RAW.md').read()
local = Path('VERIFY_GITHUB_RAW.md').read_bytes()
print('local first4bytes:', local[:4])
print('local has_bom:', local.startswith(b'\xef\xbb\xbf'))
print('main first4bytes:', raw_main[:4])
print('main has_bom:', raw_main.startswith(b'\xef\xbb\xbf'))
print('sha first4bytes:', raw_sha[:4])
print('sha has_bom:', raw_sha.startswith(b'\xef\xbb\xbf'))
PY
```

```text
local first4bytes: b'# VE'
local has_bom: False
main first4bytes: b'# VE'
main has_bom: False
sha first4bytes: b'# VE'
sha has_bom: False
```

## Local Validation Commands

```bash
python -m py_compile mcp_server.py app/config.py
docker compose -f docker-compose.yml config
```

```text
py_compile exit code: 0
docker compose config exit code: 0
docker compose config (first 60 lines):
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

All pinned raw files are multiline and hash-match Git blobs at commit 15d711cdc0d883ab95655f04aaf58b10a00ddb5a.
`VERIFY_GITHUB_RAW.md` is UTF-8 without BOM and uses LF newlines.
Python compile checks and Docker Compose config checks pass.
