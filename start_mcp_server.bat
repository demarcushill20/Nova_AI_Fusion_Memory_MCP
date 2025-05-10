@echo off
echo Starting Nova Memory MCP Server with MCP profile...
cd /d "%~dp0"
docker-compose --profile mcp down
docker-compose --profile mcp up -d
echo.
echo Containers started. Use 'docker ps' to verify.
pause
