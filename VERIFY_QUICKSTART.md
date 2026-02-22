# VERIFY_QUICKSTART

Use this to quickly verify that `main` serves multiline runtime files and that local parse checks pass.

```bash
git fetch origin && git rev-parse origin/main
```

```bash
SHA=$(git rev-parse origin/main)
```

```bash
curl -L "https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/$SHA/mcp_server.py" | nl -ba | sed -n '1,30p'
curl -L "https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/$SHA/app/config.py" | nl -ba | sed -n '1,30p'
curl -L "https://raw.githubusercontent.com/demarcushill20/Nova_AI_Fusion_Memory_MCP/$SHA/docker-compose.yml" | nl -ba | sed -n '1,80p'
```

```bash
python -m py_compile mcp_server.py app/config.py
docker compose -f docker-compose.yml config
```