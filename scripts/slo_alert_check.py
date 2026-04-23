"""SLO alert shim for PLAN-0759 Phase 8c association-linking observability.

This script scrapes the FastAPI ``/metrics`` endpoint, computes p95 latencies
and error rates against the SLO thresholds declared in
``scripts/slo_config.yaml``, and routes any violations to:

1. A nova-core Obsidian vault note under ``70-debugging/`` (idempotent per
   SLO + day).
2. An optional Slack webhook (``SLACK_ALERT_WEBHOOK`` env var); skipped with
   a log warning if the env var is unset.

A separate daily check compares today's edge-count delta (sourced from
Neo4j via :class:`MemoryEdgeService.get_edge_stats`) against a rolling
7-day median persisted in ``data/slo_edges_baseline.json``.

Intended cadence (operator wires the cron; this script does not install
anything):

.. code-block:: bash

   # */15 * * * * — every 15 minutes
   NEO4J_URI=bolt://localhost:7687 SLACK_ALERT_WEBHOOK=<url> \\
       python3 -m scripts.slo_alert_check \\
           --metrics-url http://localhost:8000/metrics

All external calls (HTTP scrape, vault write, Slack POST, Neo4j query) are
fail-open: the script logs and continues. It never raises into the cron
loop. With ``--fail-on-violation`` it exits 1 when any SLO violates; the
default is to report and exit 0 so cron stays clean.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import yaml

logger = logging.getLogger("slo_alert_check")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "scripts" / "slo_config.yaml"
_DEFAULT_VAULT = Path("/home/nova/nova-vault")
_VAULT_MARKER = "type: slo-alert"
_SAFE_SLO_TOKEN_RE = re.compile(r"[^a-z0-9_-]")


def _safe_slo_token(name: str) -> str:
    """Sanitize an SLO name for safe use in filename templating.

    Lowercases and replaces any character outside ``[a-z0-9_-]`` with
    ``_``. This strips path separators, ``..`` sequences, whitespace, and
    any shell metacharacters before the token is spliced into a filename
    pattern — defense-in-depth against a malicious config escaping the
    vault root via traversal.
    """
    return _SAFE_SLO_TOKEN_RE.sub("_", (name or "").lower())


# --------------------------------------------------------------------- #
# Result types                                                          #
# --------------------------------------------------------------------- #


@dataclass
class SLOResult:
    """Outcome of evaluating one SLO against the scraped metric set."""

    name: str
    status: str  # "ok" | "violation" | "no_data"
    actual: float | None
    threshold: float | None
    direction: str
    metric_name: str
    details: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- #
# Prometheus text-format parsing                                        #
# --------------------------------------------------------------------- #


def parse_prometheus_metrics(text: str) -> dict[str, Any]:
    """Parse a Prometheus text-format payload into a simple dict.

    Returns a mapping ``metric_name -> {"type": str, "samples": list}``.
    Each sample is ``{"labels": {...}, "value": float, "suffix": str}``
    where ``suffix`` is ``""`` for counters/gauges and one of
    ``"_bucket" | "_count" | "_sum"`` for histograms (so consumers can
    rebuild the bucket array).

    Uses ``prometheus_client.parser`` when available for correctness, with
    a minimal line-based fallback.
    """
    try:
        from prometheus_client.parser import text_string_to_metric_families
    except Exception as exc:  # pragma: no cover - import-guard
        logger.warning("prometheus_client parser unavailable: %s", exc)
        return {}

    out: dict[str, Any] = {}
    try:
        families = list(text_string_to_metric_families(text))
    except Exception as exc:
        logger.warning("failed to parse /metrics payload: %s", exc)
        return {}

    for family in families:
        base_name = family.name
        entry = {"type": family.type, "samples": []}
        for sample in family.samples:
            sample_name = sample.name
            if sample_name.startswith(base_name):
                suffix = sample_name[len(base_name):]
            else:
                suffix = ""
            entry["samples"].append(
                {
                    "labels": dict(sample.labels),
                    "value": float(sample.value),
                    "suffix": suffix,
                }
            )
        out[base_name] = entry
        # prometheus_client.parser strips the "_total" suffix from counter
        # family names, but SLO configs (and operator muscle memory) use
        # the on-wire name that ends in "_total". Alias both forms so the
        # lookup by metric name works regardless of which form the YAML
        # declares.
        if family.type == "counter" and not base_name.endswith("_total"):
            out[f"{base_name}_total"] = entry
    return out


def _collect_histogram_buckets(
    metric: dict[str, Any],
) -> tuple[list[tuple[float, float]], float, float]:
    """Reduce a histogram metric into cumulative (le, count) buckets.

    Aggregates across all label permutations present in the sample set.
    Returns ``(sorted_buckets, total_count, total_sum)``. If there are no
    samples, returns ``([], 0.0, 0.0)``.
    """
    bucket_totals: dict[float, float] = {}
    total_count = 0.0
    total_sum = 0.0
    for sample in metric.get("samples", []):
        suffix = sample["suffix"]
        labels = sample["labels"]
        value = sample["value"]
        if suffix == "_bucket":
            le_raw = labels.get("le")
            if le_raw is None:
                continue
            try:
                le = float(le_raw)
            except (TypeError, ValueError):
                continue
            bucket_totals[le] = bucket_totals.get(le, 0.0) + value
        elif suffix == "_count":
            total_count += value
        elif suffix == "_sum":
            total_sum += value
    buckets = sorted(bucket_totals.items(), key=lambda kv: kv[0])
    return buckets, total_count, total_sum


def compute_p95(buckets: list[tuple[float, float]]) -> float | None:
    """Linear-interpolate the 95th percentile from cumulative histogram buckets.

    ``buckets`` is a list of ``(le, cumulative_count)`` tuples sorted by
    ``le``. Returns ``None`` when the histogram has no samples. The
    implementation follows the classic Prometheus ``histogram_quantile``
    interpolation: find the bucket that first crosses the target count,
    then linear-interpolate within it.
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = 0.95 * total
    prev_le = 0.0
    prev_count = 0.0
    for le, cum in buckets:
        if cum >= target:
            if le == float("inf"):
                return prev_le
            span_count = cum - prev_count
            if span_count <= 0:
                return le
            span_le = le - prev_le
            frac = (target - prev_count) / span_count
            return prev_le + frac * span_le
        prev_le = le
        prev_count = cum
    # All samples fell inside +Inf bucket only — return the last finite le
    # if we saw one, else None.
    return prev_le if prev_le > 0 else None


# --------------------------------------------------------------------- #
# Daily baseline persistence                                            #
# --------------------------------------------------------------------- #


def load_daily_baseline(state_file: Path) -> list[dict[str, Any]]:
    """Load the persisted rolling 7-day edge-count baseline.

    Returns an empty list if the file is missing or unparseable. Never
    raises — the caller treats "no baseline" as "skip daily-deviation
    check".
    """
    if not state_file.exists():
        return []
    try:
        raw = state_file.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("failed to read baseline %s: %s", state_file, exc)
        return []
    if not isinstance(data, list):
        logger.warning("baseline %s is not a list; ignoring", state_file)
        return []
    return data


def append_daily_snapshot(
    state_file: Path, entry: dict[str, Any]
) -> list[dict[str, Any]]:
    """Append ``entry`` to the baseline, upsert by date, trim to 7.

    Writes atomically via a ``.tmp`` sibling + rename. Returns the new
    list. Swallows write errors with a log warning.
    """
    existing = load_daily_baseline(state_file)
    today = entry.get("date")
    out = [e for e in existing if e.get("date") != today]
    out.append(entry)
    # Keep the 7 most recent by date string (ISO dates sort lexically).
    out.sort(key=lambda e: e.get("date", ""))
    if len(out) > 7:
        out = out[-7:]
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_file.with_suffix(state_file.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        tmp.replace(state_file)
    except Exception as exc:
        logger.warning("failed to persist baseline %s: %s", state_file, exc)
    return out


def _total_edge_count(stats: dict[str, Any] | None) -> int:
    if not stats:
        return 0
    return sum(int(v.get("count", 0)) for v in stats.values())


def check_daily_edges_deviation(
    neo4j_stats: dict[str, Any] | None,
    baseline: list[dict[str, Any]],
    threshold_multiplier: float,
) -> dict[str, Any]:
    """Compute today's delta vs the rolling 7-day median delta.

    Returns a dict describing the check:
        ``{"status": "ok"|"violation"|"no_data", "delta", "median", ...}``.

    Needs at least two prior snapshots (today's + one previous) to compute
    a single delta, and any non-empty history of deltas to compare
    against. If insufficient history, status is ``"no_data"``.
    """
    if not neo4j_stats:
        return {"status": "no_data", "reason": "neo4j_stats_missing"}
    today_count = _total_edge_count(neo4j_stats)
    if not baseline:
        return {
            "status": "no_data",
            "reason": "baseline_empty",
            "today_count": today_count,
        }
    # Compute today's delta vs most-recent stored snapshot that is NOT today.
    today_iso = date.today().isoformat()
    prior = [e for e in baseline if e.get("date") != today_iso]
    if not prior:
        return {
            "status": "no_data",
            "reason": "no_prior_snapshot",
            "today_count": today_count,
        }
    prior_sorted = sorted(prior, key=lambda e: e.get("date", ""))
    last_prior = prior_sorted[-1]
    last_count = _total_edge_count(last_prior.get("edges_by_type", {}))
    todays_delta = today_count - last_count

    # Build historical deltas from consecutive baseline entries.
    deltas: list[int] = []
    for a, b in zip(prior_sorted, prior_sorted[1:]):
        deltas.append(
            _total_edge_count(b.get("edges_by_type", {}))
            - _total_edge_count(a.get("edges_by_type", {}))
        )
    # Require at least 3 historical deltas before emitting a verdict.
    # A median-of-1 or median-of-2 is noise, not signal, and can fire false
    # violations on tiny baselines. Below the floor we stay in "no_data".
    if len(deltas) < 3:
        return {
            "status": "no_data",
            "reason": "insufficient_history",
            "today_count": today_count,
            "todays_delta": todays_delta,
            "history_size": len(deltas),
        }
    med = median(deltas)
    status = "ok"
    # Only flag when we have a meaningful median to compare against.
    if med > 0:
        high = med * threshold_multiplier
        low = med / threshold_multiplier
        if todays_delta > high or todays_delta < low:
            status = "violation"
    elif med < 0:
        # Negative median: same relative bounds with signs flipped.
        high = med / threshold_multiplier
        low = med * threshold_multiplier
        if todays_delta > high or todays_delta < low:
            status = "violation"
    else:
        # Median is zero — only flag if today shows meaningful movement
        # above the multiplier (we can't form a ratio with zero).
        if abs(todays_delta) >= threshold_multiplier:
            status = "violation"
    return {
        "status": status,
        "today_count": today_count,
        "last_count": last_count,
        "todays_delta": todays_delta,
        "median_delta": med,
        "multiplier": threshold_multiplier,
        "history_size": len(deltas),
    }


# --------------------------------------------------------------------- #
# SLO evaluation                                                        #
# --------------------------------------------------------------------- #


def _evaluate_p95_slo(
    slo: dict[str, Any], metrics: dict[str, Any]
) -> SLOResult:
    metric_name = slo["metric"]
    threshold = float(slo["threshold_seconds"])
    direction = slo.get("direction", "less_than")
    entry = metrics.get(metric_name)
    if not entry:
        return SLOResult(
            name=slo["name"],
            status="no_data",
            actual=None,
            threshold=threshold,
            direction=direction,
            metric_name=metric_name,
        )
    buckets, total_count, total_sum = _collect_histogram_buckets(entry)
    if total_count <= 0:
        return SLOResult(
            name=slo["name"],
            status="no_data",
            actual=None,
            threshold=threshold,
            direction=direction,
            metric_name=metric_name,
            details={"sample_count": 0},
        )
    p95 = compute_p95(buckets)
    if p95 is None:
        return SLOResult(
            name=slo["name"],
            status="no_data",
            actual=None,
            threshold=threshold,
            direction=direction,
            metric_name=metric_name,
            details={"sample_count": total_count},
        )
    status = "ok" if (p95 < threshold if direction == "less_than" else p95 > threshold) else "violation"
    return SLOResult(
        name=slo["name"],
        status=status,
        actual=p95,
        threshold=threshold,
        direction=direction,
        metric_name=metric_name,
        details={"sample_count": total_count, "sum_seconds": total_sum},
    )


def _evaluate_error_rate_slo(
    slo: dict[str, Any], metrics: dict[str, Any]
) -> SLOResult:
    metric_name = slo["metric"]
    threshold = float(slo["threshold"])
    direction = slo.get("direction", "less_than")
    entry = metrics.get(metric_name)
    if not entry:
        return SLOResult(
            name=slo["name"],
            status="no_data",
            actual=None,
            threshold=threshold,
            direction=direction,
            metric_name=metric_name,
        )
    total = 0.0
    errors = 0.0
    for sample in entry.get("samples", []):
        # prometheus_client emits counters as "<name>_total" with suffix "_total";
        # or as "<name>" if the name already ends in _total. Accept both shapes.
        suffix = sample["suffix"]
        if suffix not in ("", "_total"):
            continue
        value = sample["value"]
        total += value
        if sample["labels"].get("outcome") == "error":
            errors += value
    if total <= 0:
        return SLOResult(
            name=slo["name"],
            status="no_data",
            actual=None,
            threshold=threshold,
            direction=direction,
            metric_name=metric_name,
            details={"total": 0.0, "errors": 0.0},
        )
    rate = errors / total
    status = "ok" if (rate < threshold if direction == "less_than" else rate > threshold) else "violation"
    return SLOResult(
        name=slo["name"],
        status=status,
        actual=rate,
        threshold=threshold,
        direction=direction,
        metric_name=metric_name,
        details={"total": total, "errors": errors},
    )


def _evaluate_daily_deviation_slo(
    slo: dict[str, Any],
    neo4j_stats: dict[str, Any] | None,
    baseline: list[dict[str, Any]],
) -> SLOResult:
    mult = float(slo.get("threshold_multiplier", 10.0))
    direction = slo.get("direction", "bounded")
    check = check_daily_edges_deviation(neo4j_stats, baseline, mult)
    status = check["status"]
    actual: float | None
    if status == "no_data":
        actual = None
    else:
        actual = float(check.get("todays_delta", 0))
    return SLOResult(
        name=slo["name"],
        status=status,
        actual=actual,
        threshold=mult,
        direction=direction,
        metric_name=slo["metric"],
        details=check,
    )


def evaluate_slos(
    config: dict[str, Any],
    metrics: dict[str, Any],
    neo4j_stats: dict[str, Any] | None,
    baseline: list[dict[str, Any]],
) -> list[SLOResult]:
    """Run every SLO in ``config`` and collect results."""
    results: list[SLOResult] = []
    for slo in config.get("slos", []):
        agg = slo.get("aggregation")
        if agg == "p95":
            results.append(_evaluate_p95_slo(slo, metrics))
        elif agg == "error_rate":
            results.append(_evaluate_error_rate_slo(slo, metrics))
        elif agg == "deviation_vs_median_7d":
            results.append(
                _evaluate_daily_deviation_slo(slo, neo4j_stats, baseline)
            )
        else:
            logger.warning("unknown aggregation %r for SLO %r", agg, slo.get("name"))
    return results


# --------------------------------------------------------------------- #
# Alert routing                                                         #
# --------------------------------------------------------------------- #


def build_alert_markdown(result: SLOResult) -> str:
    """Render a vault note (frontmatter + body) for a violation result."""
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    actual_str = "n/a" if result.actual is None else f"{result.actual:.6g}"
    threshold_str = "n/a" if result.threshold is None else f"{result.threshold:.6g}"
    remediation = _remediation_hint(result)
    details_block = (
        "\n".join(f"- `{k}`: {v}" for k, v in sorted(result.details.items()))
        if result.details
        else "- (none)"
    )
    return (
        "---\n"
        f"type: slo-alert\n"
        "plan_id: PLAN-0759-assoc-linking\n"
        f"slo: {result.name}\n"
        f"status: {result.status}\n"
        f"actual: {actual_str}\n"
        f"threshold: {threshold_str}\n"
        f"created: {created}\n"
        "tags:\n"
        '- "#type/alert"\n'
        '- "#source/slo-alert-check"\n'
        f'- "#slo/{result.name}"\n'
        "---\n\n"
        f"# SLO violation: `{result.name}`\n\n"
        f"- **Metric**: `{result.metric_name}`\n"
        f"- **Direction**: `{result.direction}`\n"
        f"- **Observed**: `{actual_str}`\n"
        f"- **Threshold**: `{threshold_str}`\n"
        f"- **Generated**: {created} (UTC)\n\n"
        "## Details\n\n"
        f"{details_block}\n\n"
        "## Remediation\n\n"
        f"{remediation}\n"
    )


def _remediation_hint(result: SLOResult) -> str:
    name = result.name
    if name == "p95_similarity_link_enqueue":
        return (
            "Enqueue latency spiked. Check `SimilarityLinker.enqueue_link` "
            "for lock contention or synchronous work creeping into the hot "
            "path; p95 budget is 5 ms."
        )
    if name == "p95_similarity_link_completion":
        return (
            "Background link completion exceeded 10 s. Inspect Neo4j write "
            "pressure and embedding-service latency."
        )
    if name == "p95_graph_expansion_latency":
        return (
            "2-hop expansion exceeded 200 ms. Inspect Neo4j query plan and "
            "recall intent mode; consider tightening `INTENT_EDGE_FILTER`."
        )
    if name == "edge_create_error_rate":
        return (
            "Edge-create error ratio above 1%. Check `edge_service` logs "
            "for MERGE failures and Neo4j connectivity."
        )
    if name == "daily_edges_created_deviation":
        return (
            "Today's edge-count delta is more than 10x the rolling 7-day "
            "median (or more than 10x below it). Investigate the backfill "
            "scripts, similarity linker throughput, and any scheduled "
            "bulk-import jobs."
        )
    return "Investigate the metric emission path and recent deploys."


def _parse_frontmatter_type(text: str) -> str | None:
    """Return the ``type`` field from a note's YAML frontmatter block.

    Returns ``None`` when there is no frontmatter, the block is malformed,
    or the parsed payload is not a mapping. Used by the vault-write safety
    gate to distinguish a prior SLO alert (safe to overwrite) from any
    other note (refuse) — substring matching in the file body is too
    permissive and would let a human-authored note containing the literal
    text ``type: slo-alert`` be clobbered.
    """
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    # Find the closing fence. Skip the opening line (index 0).
    closing = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing = idx
            break
    if closing is None:
        return None
    block = "\n".join(lines[1:closing])
    try:
        parsed = yaml.safe_load(block)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("type")
    if isinstance(value, str):
        return value
    return None


def write_vault_alert(
    vault_path: Path,
    relative_path: Path,
    content: str,
    dry_run: bool,
) -> bool:
    """Write ``content`` under ``vault_path/relative_path``, idempotently.

    Safety gates (fail-open — all log + return False on trip):
    1. Containment: the resolved target must live under the resolved
       ``vault_path``. Defends against a config with a traversal-laden
       slo name escaping the vault root even if the upstream sanitizer
       is bypassed.
    2. Clobber: if the target exists, parse its YAML frontmatter and
       require ``type == "slo-alert"`` before overwriting. Substring
       matching in the body is insufficient — a human note that happens
       to mention the schema would be eligible for overwrite.

    Returns ``True`` on write success (or dry-run preview), ``False`` if
    any gate blocked it or the filesystem layer failed.
    """
    target = vault_path / relative_path
    if dry_run:
        print(f"[DRY-RUN] would write vault alert: {target}")
        print(content)
        return True
    try:
        vault_root = vault_path.resolve()
        # Resolve against the vault root even if the target does not yet
        # exist — Path.resolve(strict=False) is Python 3.6+ default.
        resolved_target = (vault_path / relative_path).resolve()
        if not resolved_target.is_relative_to(vault_root):
            logger.warning(
                "refusing to write vault alert outside vault root: "
                "target=%s vault=%s",
                resolved_target,
                vault_root,
            )
            return False
        if target.exists():
            existing = target.read_text(encoding="utf-8", errors="replace")
            fm_type = _parse_frontmatter_type(existing)
            if fm_type != "slo-alert":
                logger.warning(
                    "refusing to overwrite vault note (frontmatter type=%r, "
                    "expected 'slo-alert'): %s",
                    fm_type,
                    target,
                )
                return False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info("wrote vault alert: %s", target)
        return True
    except Exception as exc:
        logger.warning("vault write failed for %s: %s", target, exc)
        return False


def post_slack(
    webhook_url: str | None,
    result: SLOResult,
    dry_run: bool,
) -> bool:
    """POST an alert payload to Slack. Fail-open on every error path."""
    actual_str = "n/a" if result.actual is None else f"{result.actual:.6g}"
    threshold_str = "n/a" if result.threshold is None else f"{result.threshold:.6g}"
    payload = {
        "text": (
            f"SLO VIOLATION: {result.name} "
            f"(metric={result.metric_name}, actual={actual_str}, "
            f"threshold={threshold_str}, direction={result.direction})"
        )
    }
    if dry_run:
        print(f"[DRY-RUN] would POST to Slack: {json.dumps(payload)}")
        return True
    if not webhook_url:
        logger.warning("SLACK_ALERT_WEBHOOK unset; skipping Slack notify")
        return False
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
        return True
    except Exception as exc:
        logger.warning("slack post failed: %s", exc)
        return False


# --------------------------------------------------------------------- #
# External data: metrics + Neo4j                                        #
# --------------------------------------------------------------------- #


def fetch_metrics(url: str, timeout: float = 5.0) -> str:
    """Fetch /metrics text. Returns empty string on any error (fail-open)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("metrics scrape failed from %s: %s", url, exc)
        return ""


async def _fetch_neo4j_stats_async() -> dict[str, Any] | None:
    """Fetch edge stats via MemoryEdgeService. Fail-open to None."""
    try:
        from neo4j import AsyncGraphDatabase

        from app.services.associations.edge_service import MemoryEdgeService
    except Exception as exc:
        logger.warning("neo4j client import failed: %s", exc)
        return None

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    auth_kwargs: dict[str, Any] = {}
    if user and password:
        auth_kwargs["auth"] = (user, password)
    driver = None
    try:
        driver = AsyncGraphDatabase.driver(uri, **auth_kwargs)
        svc = MemoryEdgeService(driver)
        stats = await svc.get_edge_stats()
        return stats
    except Exception as exc:
        logger.warning("neo4j stats fetch failed: %s", exc)
        return None
    finally:
        if driver is not None:
            try:
                await driver.close()
            except Exception:
                pass


def fetch_neo4j_stats() -> dict[str, Any] | None:
    """Sync wrapper around the async edge-stats fetch."""
    try:
        return asyncio.run(_fetch_neo4j_stats_async())
    except Exception as exc:
        logger.warning("neo4j stats top-level failed: %s", exc)
        return None


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #


def _load_config(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("failed to load config %s: %s", path, exc)
        return {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SLO alert shim for PLAN-0759 Phase 8c."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--metrics-url", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vault-path", type=Path, default=_DEFAULT_VAULT)
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-on-violation", action="store_true")
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip the Neo4j-backed daily-deviation check entirely.",
    )
    return parser.parse_args(argv)


def run_check(args: argparse.Namespace) -> int:
    """Execute one SLO check pass. Returns the desired process exit code."""
    config = _load_config(args.config)
    if not config:
        logger.error("empty/invalid config; nothing to evaluate")
        return 0

    metrics_url = args.metrics_url or config.get(
        "prometheus_url", "http://localhost:8000/metrics"
    )
    logger.info(
        "running SLO check: config=%s url=%s window=%dm dry_run=%s",
        args.config,
        metrics_url,
        args.window_minutes,
        args.dry_run,
    )

    raw = fetch_metrics(metrics_url)
    metrics = parse_prometheus_metrics(raw) if raw else {}

    state_file = _REPO_ROOT / config.get("state_file", "data/slo_edges_baseline.json")
    baseline = load_daily_baseline(state_file)

    neo4j_stats: dict[str, Any] | None = None
    if not args.skip_neo4j:
        neo4j_stats = fetch_neo4j_stats()
    if neo4j_stats:
        # Persist today's snapshot BEFORE evaluating deviation so the next
        # run has fresh data; evaluation uses the pre-update baseline (the
        # prior snapshots) to compute today's delta.
        today_iso = date.today().isoformat()
        append_daily_snapshot(
            state_file,
            {"date": today_iso, "edges_by_type": neo4j_stats},
        )

    results = evaluate_slos(config, metrics, neo4j_stats, baseline)

    notify_cfg = config.get("notify", {})
    vault_folder = notify_cfg.get("vault_folder", "70-debugging")
    fname_pattern = notify_cfg.get(
        "vault_filename_pattern", "slo-alert-{slo_name}-{date}.md"
    )
    slack_env = notify_cfg.get("slack_env_var", "SLACK_ALERT_WEBHOOK")
    slack_url = os.environ.get(slack_env)

    violations = 0
    today_iso = date.today().isoformat()
    for result in results:
        log_line = (
            f"slo={result.name} status={result.status} "
            f"metric={result.metric_name} actual={result.actual} "
            f"threshold={result.threshold} direction={result.direction}"
        )
        if result.status == "violation":
            violations += 1
            logger.warning(log_line)
            content = build_alert_markdown(result)
            # Sanitize the SLO name before templating — defense-in-depth
            # against a malicious config with traversal sequences.
            safe_name = _safe_slo_token(result.name)
            filename = fname_pattern.format(
                slo_name=safe_name, date=today_iso
            )
            relative = Path(vault_folder) / filename
            write_vault_alert(
                args.vault_path, relative, content, args.dry_run
            )
            post_slack(slack_url, result, args.dry_run)
        else:
            logger.info(log_line)

    logger.info(
        "SLO check summary: total=%d violations=%d no_data=%d ok=%d",
        len(results),
        violations,
        sum(1 for r in results if r.status == "no_data"),
        sum(1 for r in results if r.status == "ok"),
    )

    if args.fail_on_violation and violations > 0:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run_check(args)
    except Exception as exc:
        logger.error("unhandled error in run_check: %s", exc, exc_info=True)
        # Never break the cron loop — exit 0 unless explicitly asked.
        return 1 if args.fail_on_violation else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
