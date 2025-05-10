# Nova Memory MCP Server - Troubleshooting Guide

This guide helps diagnose and fix common issues when connecting Roo (or Claude Desktop) to the Nova Memory MCP server running in Docker.

## Common Error: "Unable to find image '...' locally" or "Connection closed"

This usually indicates a mismatch between the Docker image/container name configured in your MCP settings and the actual running container, or the container isn't running correctly.

### Step 1: Verify Docker Image Name

1.  **Check available Docker images**:
    ```bash
    docker images
    ```
    Look for images related to `nova-memory-mcp`. Note the exact `REPOSITORY` and `TAG`. Common names created by Docker Compose are `<project-name>_<service-name>` (e.g., `nova-memory-mcp_mcp-server`).

2.  **Check running containers**:
    ```bash
    docker ps
    ```
    Identify the running container for the MCP server (likely named `nova_mcp_server`). Note the `IMAGE` it's using.

### Step 2: Verify MCP Settings Configuration

1.  **Locate your Roo MCP settings file**:
    *   Windows: `%APPDATA%\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\mcp_settings.json`
    *   macOS: `~/Library/Application Support/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json`
    *   Linux: `~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/mcp_settings.json`

2.  **Edit the file** and find the `"nova-memory"` server configuration.

3.  **Ensure the image name is correct**:
    *   In the `"args"` array, the last element should be the exact image name found in Step 1 (including the tag, usually `:latest`).
    *   Example Correct Name: `"nova-memory-mcp_mcp-server:latest"`

4.  **Ensure the network configuration is correct**:
    *   The `--network` argument in `"args"` should match the network used by your running containers (usually `nova_network` as defined in `docker-compose.yml`).
    *   Example Correct Network: `"--network=nova_network"`
    *   *(Note: Using `--network=host` might work in some setups but using the specific Docker Compose network is generally more reliable for inter-container communication if needed).*

5.  **Save the file** and **restart Roo**.

### Step 3: Ensure Containers are Running Correctly

1.  **Use the provided startup script**:
    *   Navigate to the `Nova-Memory-Mcp` project directory in your terminal.
    *   Run the batch file:
        ```bash
        .\start_mcp_server.bat
        ```
    *   This script ensures the containers are stopped and restarted using the correct `--profile mcp` flag.

2.  **Verify running containers**:
    ```bash
    docker ps
    ```
    You should see at least `nova_mcp_server`, `nova_mcp_server_api`, and `nova_neo4j_db` running.

### Step 4: Check Container Logs

1.  **View logs for the MCP server**:
    ```bash
    docker logs nova_mcp_server
    ```
    Look for:
    *   `Starting in MCP mode...` (Indicates the correct mode)
    *   Any error messages related to startup or connecting to the API/database.

2.  **View logs for the API server** (if MCP server logs show connection issues):
    ```bash
    docker logs nova_mcp_server_api
    ```

### Step 5: Verify Network Connectivity (Advanced)

1.  **Inspect the Docker network**:
    ```bash
    docker network inspect nova-memory-mcp_nova_network
    ```
    Confirm that `nova_mcp_server` (or the container ID) is listed under the `Containers` section. If it's missing, it wasn't started correctly on this network.

### Step 6: Rebuild Images (If Necessary)

If you suspect code changes haven't been reflected or there are build issues:

1.  **Force a rebuild**:
    ```bash
    docker-compose build --no-cache
    ```
2.  **Restart using the script**:
    ```bash
    .\start_mcp_server.bat
    ```

### Step 7: Retry Connection

After performing the relevant steps, try connecting from Roo again.

If issues persist, review the logs carefully for specific error messages.
