# MCP Server Debug Diary

## Issue Summary
- Error: Unable to find image 'nova-memory-mcp_mcp-server:latest' locally
- Error code: MCP error -32000: Connection closed
- Environment: Local Docker setup with MCP servers
- Recent changes made to mcp_settings.json

## Diagnosis
The main issue appears to be that Docker is unable to locate the required image locally. This could be due to one of several issues:

1. The Docker image hasn't been built correctly or at all
2. The image name in mcp_settings.json doesn't match what's being built by docker-compose
3. The Docker build process is failing silently
4. Recent changes to mcp_settings.json have introduced configuration errors

## Debug Plan

### Step 1: Verify Docker Images
- Run `docker images` to check if the expected images exist locally
- Check if there are any images with names similar to 'nova-memory-mcp_mcp-server'
- Verify if the tag 'latest' is present

### Step 2: Review Docker Compose Configuration
- Examine docker-compose.yml to verify service names
- Check if the service name matches the expected image name
- Verify the build context and Dockerfile locations

### Step 3: Review Recent mcp_settings.json Changes
- Compare current mcp_settings.json with previous versions (if available)
- Check for discrepancies in Docker command parameters
- Verify if "archon-mcp:latest" vs "nova-memory-mcp_mcp-server:latest" naming is causing confusion

### Step 4: Rebuild Docker Images
- Run `docker-compose build --no-cache` to force rebuild all images
- Check for any build errors in the output
- Verify the images are created correctly with `docker images`

### Step 5: Check Docker Network
- Run `docker network ls` to verify networks exist
- Check if containers can communicate through the network
- Verify if any network-related parameters have changed in the configuration

### Step 6: Review Docker Logs
- Check logs of any running containers with `docker logs <container_id>`
- Look for error messages relating to connectivity or configuration issues

### Step 7: Fix Implementation

#### Potential Solutions:
1. If image naming is inconsistent:
   - Update mcp_settings.json to use the correct image name
   - Or rename the built image to match the expected name using `docker tag`

2. If build is failing:
   - Fix any issues in the Dockerfile or build context
   - Check for missing dependencies or build arguments

3. If network connectivity is the issue:
   - Ensure all containers are on the same Docker network
   - Verify host.docker.internal resolves correctly

4. If configuration changes caused the issue:
   - Revert specific changes in mcp_settings.json that introduced the issue
   - Update the configuration to match the current Docker setup

## Progress Tracking

| Date | Actions Taken | Results | Next Steps |
|------|---------------|---------|------------|
| 5/7/2025 | Initial diagnosis | Identified Docker image not found error | Begin with Step 1 |

## Resources
- Docker commands cheatsheet:
  - `docker images`: List all images
  - `docker ps -a`: List all containers
  - `docker-compose build --no-cache`: Rebuild all services
  - `docker logs <container_id>`: View container logs
  - `docker network ls`: List Docker networks
