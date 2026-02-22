# MEMORY_SCHEMA

This document defines the canonical metadata contract for Nova Fusion Memory MCP.
Use this schema for both `upsert_memory` and `bulk_upsert_memory`.

## Canonical Metadata Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `category` | `string` | yes | Memory class (for retrieval policy separation). |
| `run_id` | `string` | yes | Unique experiment/run identifier for reproducibility. |
| `agent` | `string` | yes | Agent that produced the memory (e.g., `research_agent`). |
| `source` | `string` | yes | Origin of information (`url`, `file`, `manual`, `agent`). |
| `timestamp` | `string` (ISO 8601 UTC) | yes | Creation/update time (`2026-02-22T10:30:00Z`). |
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

## Validation Rules

1. `category`, `run_id`, `agent`, `source`, and `timestamp` should be present on all durable memories.
2. `timestamp` should be ISO 8601 in UTC (suffix `Z`).
3. `tags` should be lowercase strings for consistent filtering.
4. Keep metadata JSON-serializable.

## Example Metadata

```json
{
  "category": "strategy_spec",
  "run_id": "run_20260222_001",
  "agent": "pine_architect",
  "source": "agent",
  "timestamp": "2026-02-22T10:30:00Z",
  "symbol": "BTCUSD",
  "timeframe": "15m",
  "tags": ["breakout", "volatility"],
  "market": "crypto"
}
```

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
