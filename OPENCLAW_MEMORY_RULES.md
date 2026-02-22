# OPENCLAW_MEMORY_RULES

This document defines memory discipline for OpenClaw agents using Nova Fusion Memory MCP.

## Core Rule

```text
IF information may be useful later:
    upsert_memory(...) or bulk_upsert_memory(...)
ELSE:
    keep local only
```

## Write Rules

1. Every durable write must include canonical metadata from `MEMORY_SCHEMA.md`.
2. Always set `run_id` so experiments can be reconstructed end-to-end.
3. Use `bulk_upsert_memory` for multi-item ingestion from research pages, logs, or sweeps.
4. Write concise, atomic memories (one decision/finding/error per item).

## Read Rules

1. Always filter queries by `run_id` when working inside one experiment loop.
2. Use `category` and `tags` to prevent cross-agent memory contamination.
3. Set `min_score` for high-precision tasks (risk checks, deployment gates).
4. Use tuned `top_k_vector`/`top_k_final` by agent role.

## Agent Policy Examples

| Agent | Preferred `category` reads | Typical write categories |
|---|---|---|
| `research_agent` | `research_note`, `backtest_result` | `research_note`, `backtest_result` |
| `pine_architect` | `strategy_spec`, `compile_error` | `strategy_spec` |
| `qa_agent` | `compile_error`, `risk_rule` | `compile_error`, `postmortem` |
| `risk_supervisor` | `risk_rule`, `postmortem` | `risk_rule` |
| `deployment_agent` | `deployment_config`, `risk_rule` | `deployment_config`, `postmortem` |

## Minimal Workflow

1. Query scoped memory for the current `run_id`.
2. Perform task-specific reasoning.
3. Persist durable outputs with schema-compliant metadata.
4. On completion, write a `postmortem` summary for future runs.
