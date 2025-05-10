# Detailed MCP Connection Error Analysis

## Root Issue Identified
After thorough analysis of the codebase, I've identified the exact cause of the MCP error -32000: Connection closed.

### Critical Image Name Mismatch
In `mcp_settings.json`, the nova-memory MCP server is configured to use the following Docker image:
```
"nova-memory-mcp-mcp-server:latest"
```

But this image doesn't exist in your Docker environment for two reasons:

1. **Double "mcp" in name**: The name contains "mcp" twice ("nova-memory-mcp-mcp-server")
2. **Docker Compose naming convention**: Docker Compose would create image names as `nova-memory-mcp_mcp-server:latest` (with underscore, not hyphen)

When examining your docker-compose.yml, it shows:
```yaml
mcp-server:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: nova_mcp_server
  # ...other configuration...
  profiles:
    - mcp
```

Docker Compose would automatically name this image as `nova-memory-mcp_mcp-server:latest` (project directory name + service name), but your mcp_settings.json is looking for `nova-memory-mcp-mcp-server:latest` (with a hyphen instead of underscore, and a duplicated "mcp").

### Profile Not Activated
The service is defined with `profiles: [mcp]` which means it's not started by default when running `docker-compose up`. You need to explicitly include the profile:
```
docker-compose --profile mcp up
```

## Detailed Fix Plan

### 1. Correct the MCP Settings Configuration
Edit `C:\Users\black\AppData\Roaming\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\mcp_settings.json`:

Change:
```json
"args": [
  "run",
  "-i",
  "--rm",
  "--network=nova_network",
  "nova-memory-mcp-mcp-server:latest"
]
```

To:
```json
"args": [
  "run",
  "-i",
  "--rm",
  "--network=nova_network",
  "nova-memory-mcp_mcp-server:latest"
]
```

### 2. Start All Required Docker Containers with the Correct Profile
```
cd C:\Users\black\Desktop\Local_MCP\Nova-Memory-Mcp
docker-compose --profile mcp up -d
```

### 3. Verify Docker Network Configuration
The Docker network configuration in your mcp_settings.json uses `--network=nova_network`, which requires:
1. The network to exist
2. The container to be allowed to connect to this network

Check that the nova_network exists:
```
docker network ls
```

If needed, create the network:
```
docker network create nova_network
```

### 4. Check API Connectivity
The MCP adapter is configured to connect to the API server at:
```
API_BASE_URL = "http://nova_mcp_server_api:8000"
```

This hostname resolution requires both containers to be on the same Docker network. If there are connectivity issues, you can verify the API server is running properly by:

1. Check if the API container is running:
```
docker ps | findstr nova_mcp_server_api
```

2. Check logs for the API container:
```
docker logs nova_mcp_server_api
```

3. Test the API directly:
```
docker exec -it nova_mcp_server curl http://nova_mcp_server_api:8000/memory/health
```

### 5. Advanced Debugging Options

#### Option 1: Run MCP Container with Shell for Debugging
```
docker-compose run --rm --entrypoint /bin/bash mcp-server
```
This will give you a shell inside the container where you can try:
```
curl http://nova_mcp_server_api:8000/memory/health
```

#### Option 2: Tag an Existing Image with the Expected Name
If you've already built the image but it has a different name, you can create a tag:
```
docker tag nova-memory-mcp_mcp-server:latest nova-memory-mcp-mcp-server:latest
```

## Progress Tracking

| Date | Actions Taken | Results | Next Steps |
|------|---------------|---------|------------|
| 5/7/2025 | Deep code analysis | Found image name mismatch and profile issues | Apply fixes 1-2 |
