# VERIFY_QUICKSTART

If `.../main/...` looks stale, use `.../refs/heads/main/...` or a commit-pinned URL.
This avoids raw alias cache inconsistencies in some environments.

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

```bash
python - << 'PY'
from pathlib import Path
p = "VERIFY_GITHUB_RAW.md"
b = Path(p).read_bytes()[:4]
print("first4bytes:", b)
print("has_bom:", b.startswith(b"\xef\xbb\xbf"))
PY
```
