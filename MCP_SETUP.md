# Nova AI Memory MCP Server - Docker Setup

This document explains how to set up and use the Nova AI Memory MCP Server with Claude Desktop using Docker.

## Architecture Overview

The Nova AI Memory MCP Server has been containerized to support two operating modes:

1. **API Mode** - Exposes a REST API on port 8000 for direct HTTP requests (default mode)
2. **MCP Mode** - Communicates via stdin/stdout following the Model Context Protocol (MCP) for Claude Desktop integration

The MCP Mode uses an adapter layer that translates between the MCP protocol and the internal REST API, allowing seamless integration with Claude Desktop while maintaining the original API functionality.

## Prerequisites

- Docker and Docker Compose installed
- Claude Desktop application installed (for MCP integration)
- Neo4j database (automatically set up by docker-compose)

## Getting Started

### Starting the Nova Memory MCP Server

#### Option 1: Running in API Mode (Default)

To run the server in API mode (exposing the REST API on port 8000):

```bash
docker-compose up -d
```

This will start two containers:
- `nova_neo4j_db` - Neo4j database
- `nova_mcp_server_api` - The Nova Memory server in API mode

You can access the API directly at `http://localhost:8000/memory/health` to check if it's running.

#### Option 2: Running in MCP Mode (For Claude Desktop)

To run the server in MCP mode for Claude Desktop integration:

```bash
docker-compose --profile mcp up -d
```

This will start two containers:
- `nova_neo4j_db` - Neo4j database
- `nova_mcp_server` - The Nova Memory server in MCP mode

## Configuring Claude Desktop

To use the Nova Memory MCP Server with Claude Desktop, you need to update your Claude Desktop configuration:

1. Locate your Claude Desktop configuration file:
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. Add the following configuration to the file:

```json
{
  "mcpServers": {
    "nova-memory": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "--network=host", "nova-memory-mcp_mcp-server"],
      "disabled": false,
      "autoApprove": [],
      "alwaysAllow": [
        "query_memory",
        "upsert_memory",
        "delete_memory",
        "check_health"
      ]
    }
  }
}
```

3. Restart Claude Desktop to apply the changes.

## Available MCP Tools

The Nova Memory MCP Server provides the following tools for Claude Desktop:

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `query_memory` | Query the memory system | `{"query": "your query text"}` |
| `upsert_memory` | Add or update a memory item | `{"text": "memory content", "id": "optional-id", "metadata": {}}` |
| `delete_memory` | Delete a memory item by ID | `{"id": "memory-id-to-delete"}` |
| `check_health` | Check the health of the memory system | `{}` |

## Example Usage in Claude Desktop

Once configured, you can use the Nova Memory MCP Server in Claude Desktop with prompts like:

```
Can you store this information in my memory: "The deadline for the project is November 15th"
```

Claude will use the `upsert_memory` tool to store this information.

```
What do you know about the project deadline?
```

Claude will use the `query_memory` tool to retrieve relevant memories.

## Troubleshooting

If you encounter issues:

1. Check if the Neo4j container is running: `docker ps | grep nova_neo4j_db`
2. Check if the MCP server container is running: `docker ps | grep nova_mcp_server`
3. Check the logs: `docker logs nova_mcp_server`
4. Ensure your Claude Desktop configuration file is correctly formatted
5. Try restarting the containers: `docker-compose --profile mcp down && docker-compose --profile mcp up -d`

## Building the Docker Image Manually

If you need to build the Docker image manually:

```bash
docker build -t nova-memory-mcp:latest .
```

## Advanced Configuration

The Docker container accepts the following environment variables:

- `RUNTIME_MODE` - Set to `api` for REST API mode or `mcp` for Claude Desktop integration
- `PORT` - Port number for the API server (default: 8000)

You can customize these in the `.env` file or in the `docker-compose.yml` file.