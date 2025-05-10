# MCP Connection Error Fix Plan

## Identified Issues
1. **Docker Image Name Mismatch**: MCP settings is looking for `nova-memory-mcp_mcp-server:latest` but the container/image appears to be using a different name.
2. **Docker Container Not Running**: The MCP server container appears to be either not running or not accessible using the expected name.
3. **Docker Profile Configuration**: The MCP server in docker-compose.yml uses a profile that might not be activated.
4. **Network Configuration Inconsistency**: Network settings differ between docker-compose.yml and mcp_settings.json.

## Detailed Fix Plan

### Step 1: Verify and Fix Docker Image Names
1. Check existing images:
```bash
docker images
```

2. If the image exists but with a different name, tag it with the expected name:
```bash
docker tag nova_mcp_server:latest nova-memory-mcp_mcp-server:latest
```

### Step 2: Correct MCP Settings Configuration
1. Update MCP settings to match actual Docker container/image names:
   - Edit: `C:\Users\black\AppData\Roaming\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\mcp_settings.json`
   - Modify the "nova-memory" section to match your actual Docker container name and network configuration:

```json
"nova-memory": {
  "command": "docker",
  "args": [
    "run",
    "-i",
    "--rm",
    "--network=nova_network",
    "nova_mcp_server:latest"
  ],
  "disabled": false,
  "autoApprove": [],
  "alwaysAllow": [
    "query_memory",
    "upsert_memory",
    "delete_memory",
    "check_health"
  ]
}
```

### Step 3: Ensure MCP Server Container Is Running
1. Check running containers:
```bash
docker ps
```

2. If the MCP server container is not running, start it with the correct profile:
```bash
docker-compose --profile mcp up -d
```

### Step 4: Verify Docker Network Configuration
1. Check Docker networks:
```bash
docker network ls
```

2. Ensure the container is connected to the correct network:
```bash
docker network inspect nova_network
```

3. If needed, modify docker-compose.yml to use host networking if that's what your MCP settings expects:
```yml
mcp-server:
  # other settings...
  network_mode: "host"
  # Or remove the following if switching to host network
  # networks:
  #   nova_network:
  #     aliases:
  #       - mcp_server
```

### Step 5: Rebuild and Restart
1. Rebuild the Docker images:
```bash
docker-compose build --no-cache
```

2. Stop and remove existing containers:
```bash
docker-compose down
```

3. Start with the MCP profile:
```bash
docker-compose --profile mcp up -d
```

### Step 6: Verify MCP Connection
1. Check if Claude/Roo-Code can now connect to the MCP server
2. Verify container logs for any errors:
```bash
docker logs nova_mcp_server
```

## Progress Tracking

| Date | Actions Taken | Results | Next Steps |
|------|---------------|---------|------------|
| 5/7/2025 | Initial diagnosis | Identified Docker image not found error | Begin with Step 1 |
| 5/7/2025 | Updated `mcp_settings.json` with correct image name (`nova-memory-mcp-mcp-server:latest`) and network (`nova_network`). Verified network exists. Rebuilt images (`--no-cache`) and restarted containers (`docker-compose down; docker-compose --profile mcp up -d`). Checked logs. | `nova_mcp_server` running in MCP mode. `mcp_settings.json` updated. | Verify connection in Roo (Step 6). |
