# Nova AI Memory MCP Server - Docker Setup

This document explains how to set up and use the Nova AI Memory MCP Server with Claude Desktop using Docker.

## Architecture Overview

The current containerized deployment runs a single **MCP stdio server** powered by `FastMCP` (`mcp_server.py`).
There is no runtime adapter layer required for MCP operation.

## Prerequisites

- Docker and Docker Compose installed
- Claude Desktop application installed (for MCP integration)
- Neo4j database (automatically set up by docker-compose)

## Getting Started

### Starting the Nova Memory MCP Server

To run the MCP server for Claude Desktop integration:

```bash
docker-compose --profile mcp up -d
```

This will start two containers:
- `nova_neo4j_db` - Neo4j database
- `nova_mcp_server` - The Nova Memory MCP server (stdio transport)

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
      "args": [
        "run",
        "-i",
        "--rm",
        "--network=nova-memory-mcp_nova_network",
        "--env-file",
        "c:/path/to/your/nova-memory-mcp/.env",
        "nova-memory-mcp_mcp-server:latest"
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
  }
}
```

Update the `--network` and `--env-file` values for your local project path and Docker network name.

3. Restart Claude Desktop to apply the changes.

## Available MCP Tools

The Nova Memory MCP Server provides the following tools for Claude Desktop:

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `query_memory` | Query the memory system | `{"query": "your query text"}` |
| `upsert_memory` | Add or update a memory item | `{"content": "memory content", "id": "optional-id", "metadata": {}}` |
| `delete_memory` | Delete a memory item by ID | `{"memory_id": "memory-id-to-delete"}` |
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

Configure service credentials and model/database settings in `.env`.
The MCP server runs directly as `python mcp_server.py` in the container entrypoint.
