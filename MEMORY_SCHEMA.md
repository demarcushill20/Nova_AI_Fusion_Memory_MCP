# MEMORY_SCHEMA

This document defines the canonical metadata contract for Nova Fusion Memory MCP.
Use this schema for both `upsert_memory` and `bulk_upsert_memory`.

## System-Enforced Fields (Phase 1 — Chronological Ordering)

These fields are **automatically injected** by the server on every write.
Agents do not need to set them — the system guarantees their presence.

| Field | Type | Injected By | Description |
|---|---|---|---|
| `event_seq` | `integer` | **system** (always) | Monotonic sequence number. The canonical ordering key. Never caller-provided. |
| `event_time` | `string` (ISO 8601 UTC) | **system** (if missing) | Write timestamp. Caller may provide; system fills if absent. |
| `memory_type` | `string` | **system** (if missing) | Memory class: `scratch`, `decision`, `artifact`, `finding`, `checkpoint`, `constraint`. Defaults to `scratch`. |

> **Why `event_seq` matters:** Timestamps can drift, tool calls can arrive out of order, and multi-agent writes can interleave. `event_seq` guarantees a strict "what happened after what" ordering even if timestamps are messy.

## Canonical Metadata Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `category` | `string` | yes | Memory class (for retrieval policy separation). |
| `run_id` | `string` | yes | Unique experiment/run identifier for reproducibility. |
| `agent` | `string` | yes | Agent that produced the memory (e.g., `research_agent`). |
| `source` | `string` | yes | Origin of information (`url`, `file`, `manual`, `agent`). |
| `timestamp` | `string` (ISO 8601 UTC) | yes | Creation/update time (`2026-02-22T10:30:00Z`). |
| `thread_id` | `string` | recommended | Conversation/run grouping identifier. |
| `session_id` | `string` | recommended | Session grouping identifier. |
| `project` | `string` | recommended | Project scope (e.g., `NovaTrade`, `NovaSHIFT`). |
| `symbol` | `string` | no | Instrument symbol when relevant (`BTCUSD`, `AAPL`). |
| `timeframe` | `string` | no | Strategy/market timeframe (`1m`, `15m`, `1h`, `1d`). |
| `tags` | `array[string]` | no | Additional retrieval tags (`["risk", "breakout"]`). |
| `market` | `string` | no | Market venue/context (`crypto`, `forex`, `equities`). |

## Recommended Category Values

- `strategy_spec`
- `backtest_result`
- `risk_rule`
- `compile_error`
- `deployment_config`
- `postmortem`
- `research_note`

## Recommended memory_type Values

- `scratch` — default, general-purpose memory
- `decision` — a decision or conclusion reached
- `artifact` — a produced output or deliverable
- `finding` — a research finding or observation
- `checkpoint` — session checkpoint (see Phase 2)
- `constraint` — a rule or constraint to remember

## Validation Rules

1. `event_seq` and `event_time` are always present (system-enforced — do not set `event_seq` manually).
2. `category`, `run_id`, `agent`, `source`, and `timestamp` should be present on all durable memories.
3. `timestamp` should be ISO 8601 in UTC (suffix `Z`).
4. `tags` should be lowercase strings for consistent filtering.
5. Keep metadata JSON-serializable.

## Example Metadata

```json
{
  "event_seq": 42,
  "event_time": "2026-02-22T10:30:00+00:00",
  "memory_type": "finding",
  "category": "strategy_spec",
  "run_id": "run_20260222_001",
  "agent": "pine_architect",
  "source": "agent",
  "timestamp": "2026-02-22T10:30:00Z",
  "project": "NovaTrade",
  "thread_id": "thread_001",
  "symbol": "BTCUSD",
  "timeframe": "15m",
  "tags": ["breakout", "volatility"],
  "market": "crypto"
}
```

> Note: `event_seq` is shown for reference — it is always system-assigned, never caller-provided.

## Example Upsert Payload

```json
{
  "content": "Use ATR stop of 1.8x for BTC breakout strategy.",
  "metadata": {
    "category": "strategy_spec",
    "run_id": "run_20260222_001",
    "agent": "research_agent",
    "source": "agent",
    "timestamp": "2026-02-22T10:30:00Z",
    "symbol": "BTCUSD",
    "timeframe": "15m",
    "tags": ["risk", "atr"]
  }
}
```

## Example Bulk Upsert Payload

```json
{
  "items": [
    {
      "content": "Compile error: undeclared identifier at line 42.",
      "metadata": {
        "category": "compile_error",
        "run_id": "run_20260222_001",
        "agent": "qa_agent",
        "source": "agent",
        "timestamp": "2026-02-22T10:35:00Z",
        "symbol": "BTCUSD",
        "timeframe": "15m",
        "tags": ["compile", "pine"]
      }
    },
    {
      "content": "Backtest Sharpe improved to 1.74 after stop update.",
      "metadata": {
        "category": "backtest_result",
        "run_id": "run_20260222_001",
        "agent": "research_agent",
        "source": "agent",
        "timestamp": "2026-02-22T10:42:00Z",
        "symbol": "BTCUSD",
        "timeframe": "15m",
        "tags": ["backtest", "sharpe"]
      }
    }
  ]
}
```
