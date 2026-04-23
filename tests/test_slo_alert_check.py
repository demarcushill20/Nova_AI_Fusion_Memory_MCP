"""Hermetic tests for scripts/slo_alert_check.py (PLAN-0759 Sprint 19).

These tests exercise the SLO shim end-to-end without touching the network,
Neo4j, or the real Obsidian vault. Every external surface
(``urllib.request.urlopen``, Neo4j stats, vault filesystem) is either
mocked or redirected to a ``tmp_path`` sandbox.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts import slo_alert_check as shim


# --------------------------------------------------------------------- #
# Metric-text fixtures                                                  #
# --------------------------------------------------------------------- #


def _hist_lines(
    metric: str,
    buckets: list[tuple[str, float]],
    count: float,
    total: float,
    labels: str = "",
) -> str:
    lbl_inner = labels
    label_comma = f"{lbl_inner}," if lbl_inner else ""
    body = [f"# HELP {metric} test", f"# TYPE {metric} histogram"]
    for le, cum in buckets:
        body.append(
            f'{metric}_bucket{{{label_comma}le="{le}"}} {cum}'
        )
    if lbl_inner:
        body.append(f"{metric}_count{{{lbl_inner}}} {count}")
        body.append(f"{metric}_sum{{{lbl_inner}}} {total}")
    else:
        body.append(f"{metric}_count {count}")
        body.append(f"{metric}_sum {total}")
    return "\n".join(body) + "\n"


def _green_metrics_text() -> str:
    enqueue = _hist_lines(
        "similarity_link_enqueue_seconds",
        [
            ("0.0005", 50.0),
            ("0.001", 95.0),
            ("0.002", 100.0),
            ("0.005", 100.0),
            ("+Inf", 100.0),
        ],
        count=100.0,
        total=0.08,
    )
    completion = _hist_lines(
        "similarity_link_completion_seconds",
        [
            ("0.5", 10.0),
            ("1.0", 80.0),
            ("2.5", 100.0),
            ("10.0", 100.0),
            ("+Inf", 100.0),
        ],
        count=100.0,
        total=90.0,
    )
    expansion = _hist_lines(
        "graph_expansion_latency_seconds",
        [
            ("0.01", 5.0),
            ("0.05", 50.0),
            ("0.1", 95.0),
            ("0.2", 100.0),
            ("+Inf", 100.0),
        ],
        count=100.0,
        total=5.0,
        labels='mode="general"',
    )
    # Error rate 0.1% (5/5000) — green.
    counter = (
        "# HELP edges_created_total test\n"
        "# TYPE edges_created_total counter\n"
        'edges_created_total{edge_type="SIMILAR_TO",outcome="success"} 4995.0\n'
        'edges_created_total{edge_type="SIMILAR_TO",outcome="error"} 5.0\n'
    )
    return enqueue + completion + expansion + counter


def _enqueue_violation_text() -> str:
    # Push most samples above 0.005 so p95 lands at ~0.008.
    enqueue = _hist_lines(
        "similarity_link_enqueue_seconds",
        [
            ("0.0005", 0.0),
            ("0.001", 0.0),
            ("0.002", 10.0),
            ("0.005", 30.0),
            ("0.01", 100.0),
            ("+Inf", 100.0),
        ],
        count=100.0,
        total=0.6,
    )
    return enqueue


def _completion_violation_text() -> str:
    completion = _hist_lines(
        "similarity_link_completion_seconds",
        [
            ("0.5", 0.0),
            ("1.0", 0.0),
            ("2.5", 0.0),
            ("5.0", 10.0),
            ("10.0", 50.0),
            ("25.0", 100.0),
            ("+Inf", 100.0),
        ],
        count=100.0,
        total=1500.0,
    )
    return completion


def _expansion_violation_text() -> str:
    expansion = _hist_lines(
        "graph_expansion_latency_seconds",
        [
            ("0.01", 0.0),
            ("0.05", 0.0),
            ("0.1", 10.0),
            ("0.2", 50.0),
            ("0.5", 100.0),
            ("+Inf", 100.0),
        ],
        count=100.0,
        total=30.0,
        labels='mode="general"',
    )
    return expansion


def _error_rate_violation_text() -> str:
    # 50 errors / 5000 total = 1% → fails "strictly less than 0.01".
    return (
        "# HELP edges_created_total test\n"
        "# TYPE edges_created_total counter\n"
        'edges_created_total{edge_type="SIMILAR_TO",outcome="success"} 4950.0\n'
        'edges_created_total{edge_type="SIMILAR_TO",outcome="error"} 50.0\n'
    )


def _empty_metrics_text() -> str:
    return _hist_lines(
        "similarity_link_enqueue_seconds",
        [("0.0005", 0.0), ("+Inf", 0.0)],
        count=0.0,
        total=0.0,
    )


# --------------------------------------------------------------------- #
# Config helpers                                                        #
# --------------------------------------------------------------------- #


def _load_real_config() -> dict:
    with open(shim._DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------- #
# Parser tests                                                          #
# --------------------------------------------------------------------- #


def test_parse_prometheus_metrics_empty_string_returns_empty_dict():
    assert shim.parse_prometheus_metrics("") == {}


def test_parse_prometheus_metrics_histogram_roundtrip():
    text = _green_metrics_text()
    parsed = shim.parse_prometheus_metrics(text)
    assert "similarity_link_enqueue_seconds" in parsed
    entry = parsed["similarity_link_enqueue_seconds"]
    assert entry["type"] == "histogram"
    assert any(s["suffix"] == "_bucket" for s in entry["samples"])
    assert any(s["suffix"] == "_count" for s in entry["samples"])


def test_compute_p95_with_interpolation():
    buckets = [
        (0.001, 0.0),
        (0.005, 5.0),
        (0.01, 10.0),
        (float("inf"), 10.0),
    ]
    p95 = shim.compute_p95(buckets)
    assert p95 is not None
    assert 0.008 < p95 < 0.01


def test_compute_p95_empty_histogram_returns_none():
    assert shim.compute_p95([]) is None
    assert shim.compute_p95([(0.1, 0.0), (float("inf"), 0.0)]) is None


# --------------------------------------------------------------------- #
# SLO evaluation tests                                                  #
# --------------------------------------------------------------------- #


def test_all_green_no_violations():
    config = _load_real_config()
    metrics = shim.parse_prometheus_metrics(_green_metrics_text())
    results = shim.evaluate_slos(config, metrics, None, [])
    statuses = {r.name: r.status for r in results}
    # p95 SLOs all "ok", error rate "ok", daily deviation "no_data".
    assert statuses["p95_similarity_link_enqueue"] == "ok"
    assert statuses["p95_similarity_link_completion"] == "ok"
    assert statuses["p95_graph_expansion_latency"] == "ok"
    assert statuses["edge_create_error_rate"] == "ok"
    assert statuses["daily_edges_created_deviation"] == "no_data"


def test_enqueue_p95_violation():
    config = _load_real_config()
    metrics = shim.parse_prometheus_metrics(_enqueue_violation_text())
    results = shim.evaluate_slos(config, metrics, None, [])
    enq = next(r for r in results if r.name == "p95_similarity_link_enqueue")
    assert enq.status == "violation"
    assert enq.actual is not None and enq.actual >= 0.005


def test_completion_p95_violation():
    config = _load_real_config()
    metrics = shim.parse_prometheus_metrics(_completion_violation_text())
    results = shim.evaluate_slos(config, metrics, None, [])
    comp = next(r for r in results if r.name == "p95_similarity_link_completion")
    assert comp.status == "violation"
    assert comp.actual is not None and comp.actual > 10.0


def test_expansion_p95_violation():
    config = _load_real_config()
    metrics = shim.parse_prometheus_metrics(_expansion_violation_text())
    results = shim.evaluate_slos(config, metrics, None, [])
    exp = next(r for r in results if r.name == "p95_graph_expansion_latency")
    assert exp.status == "violation"
    assert exp.actual is not None and exp.actual > 0.2


def test_error_rate_violation():
    config = _load_real_config()
    metrics = shim.parse_prometheus_metrics(_error_rate_violation_text())
    results = shim.evaluate_slos(config, metrics, None, [])
    err = next(r for r in results if r.name == "edge_create_error_rate")
    assert err.status == "violation"
    assert err.actual is not None
    assert err.actual >= 0.01


def test_no_data_when_empty_histogram():
    config = _load_real_config()
    metrics = shim.parse_prometheus_metrics(_empty_metrics_text())
    results = shim.evaluate_slos(config, metrics, None, [])
    enq = next(r for r in results if r.name == "p95_similarity_link_enqueue")
    assert enq.status == "no_data"


def test_no_data_when_metric_absent():
    config = _load_real_config()
    results = shim.evaluate_slos(config, {}, None, [])
    for r in results[:4]:  # p95 + error-rate SLOs
        assert r.status == "no_data"


# --------------------------------------------------------------------- #
# Daily deviation tests                                                 #
# --------------------------------------------------------------------- #


def _stub_stats(total_count: int) -> dict:
    return {
        "SIMILAR_TO": {
            "count": total_count,
            "avg_weight": None,
            "min_weight": None,
            "max_weight": None,
        }
    }


def test_daily_deviation_no_data_empty_baseline():
    check = shim.check_daily_edges_deviation(_stub_stats(1000), [], 10.0)
    assert check["status"] == "no_data"


def test_daily_deviation_violation_when_10x_median(tmp_path):
    # Build a 5-day baseline where each day adds ~100 edges (median delta 100).
    # Today's delta is 1500 -> well above 10x.
    baseline = []
    counts = [1000, 1100, 1200, 1300, 1400]
    for i, c in enumerate(counts):
        baseline.append(
            {
                "date": f"2026-04-{18 + i:02d}",
                "edges_by_type": _stub_stats(c),
            }
        )
    # Today's stats show 2900 -> delta vs last (1400) = 1500.
    check = shim.check_daily_edges_deviation(_stub_stats(2900), baseline, 10.0)
    assert check["status"] == "violation"
    assert check["todays_delta"] == 1500
    assert check["median_delta"] == 100


def test_daily_deviation_ok_within_multiplier():
    # Need ≥3 historical deltas after the insufficient-history floor; use 5
    # baseline entries (4 prior deltas) so the median-of-deltas is valid.
    baseline = [
        {"date": "2026-04-17", "edges_by_type": _stub_stats(900)},
        {"date": "2026-04-18", "edges_by_type": _stub_stats(1000)},
        {"date": "2026-04-19", "edges_by_type": _stub_stats(1100)},
        {"date": "2026-04-20", "edges_by_type": _stub_stats(1200)},
        {"date": "2026-04-21", "edges_by_type": _stub_stats(1300)},
    ]
    # Today = 1400 -> delta 100, median 100 -> OK.
    check = shim.check_daily_edges_deviation(_stub_stats(1400), baseline, 10.0)
    assert check["status"] == "ok"


def test_daily_deviation_no_data_when_history_too_short():
    # Two baseline entries -> one historical delta -> below the floor of 3.
    # New behavior: return "no_data" with reason "insufficient_history"
    # instead of emitting a noisy verdict from a single-delta median.
    baseline = [
        {"date": "2026-04-19", "edges_by_type": _stub_stats(1000)},
        {"date": "2026-04-20", "edges_by_type": _stub_stats(1100)},
    ]
    # Today's delta would be 1900 (3x median of 100) — with a healthy
    # history this would be ok/violation; with a 1-delta history we refuse
    # to emit a verdict.
    check = shim.check_daily_edges_deviation(_stub_stats(3000), baseline, 10.0)
    assert check["status"] == "no_data"
    assert check.get("reason") == "insufficient_history"
    assert check.get("history_size") == 1


def test_daily_deviation_no_data_when_stats_missing():
    check = shim.check_daily_edges_deviation(None, [], 10.0)
    assert check["status"] == "no_data"


def test_append_daily_snapshot_upsert_and_trim(tmp_path):
    state = tmp_path / "baseline.json"
    for i in range(10):
        shim.append_daily_snapshot(
            state,
            {"date": f"2026-04-{10 + i:02d}", "edges_by_type": _stub_stats(i)},
        )
    out = shim.load_daily_baseline(state)
    assert len(out) == 7  # trimmed
    # Same-date re-run overwrites (idempotent).
    shim.append_daily_snapshot(
        state,
        {"date": "2026-04-19", "edges_by_type": _stub_stats(999)},
    )
    out2 = shim.load_daily_baseline(state)
    dates = [e["date"] for e in out2]
    assert dates.count("2026-04-19") == 1


def test_load_daily_baseline_missing_file_returns_empty(tmp_path):
    assert shim.load_daily_baseline(tmp_path / "nope.json") == []


def test_load_daily_baseline_corrupt_file_returns_empty(tmp_path):
    state = tmp_path / "bad.json"
    state.write_text("not-json", encoding="utf-8")
    assert shim.load_daily_baseline(state) == []


# --------------------------------------------------------------------- #
# Vault write + clobber protection                                      #
# --------------------------------------------------------------------- #


def test_vault_write_fresh_file(tmp_path):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    result = shim.SLOResult(
        name="p95_similarity_link_enqueue",
        status="violation",
        actual=0.01,
        threshold=0.005,
        direction="less_than",
        metric_name="similarity_link_enqueue_seconds",
    )
    content = shim.build_alert_markdown(result)
    ok = shim.write_vault_alert(
        vault, Path("70-debugging") / "slo-alert-foo-2026-04-23.md",
        content, dry_run=False,
    )
    assert ok
    written = (vault / "70-debugging" / "slo-alert-foo-2026-04-23.md").read_text()
    assert "type: slo-alert" in written


def test_vault_write_refuses_to_clobber_human_note(tmp_path):
    vault = tmp_path / "vault"
    folder = vault / "70-debugging"
    folder.mkdir(parents=True)
    target = folder / "slo-alert-foo-2026-04-23.md"
    target.write_text(
        "---\ntype: research\nauthor: human\n---\n\n# Do not overwrite me\n",
        encoding="utf-8",
    )
    result = shim.SLOResult(
        name="foo",
        status="violation",
        actual=0.01,
        threshold=0.005,
        direction="less_than",
        metric_name="foo",
    )
    content = shim.build_alert_markdown(result)
    ok = shim.write_vault_alert(
        vault, Path("70-debugging") / "slo-alert-foo-2026-04-23.md",
        content, dry_run=False,
    )
    assert not ok
    assert "Do not overwrite me" in target.read_text()


def test_vault_write_overwrites_prior_slo_alert(tmp_path):
    vault = tmp_path / "vault"
    folder = vault / "70-debugging"
    folder.mkdir(parents=True)
    target = folder / "slo-alert-foo-2026-04-23.md"
    target.write_text(
        "---\ntype: slo-alert\nslo: foo\nstatus: violation\n---\n\nOld body\n",
        encoding="utf-8",
    )
    result = shim.SLOResult(
        name="foo",
        status="violation",
        actual=0.02,
        threshold=0.005,
        direction="less_than",
        metric_name="foo",
    )
    content = shim.build_alert_markdown(result)
    ok = shim.write_vault_alert(
        vault, Path("70-debugging") / "slo-alert-foo-2026-04-23.md",
        content, dry_run=False,
    )
    assert ok
    body = target.read_text()
    assert "Old body" not in body
    assert "type: slo-alert" in body


def test_vault_write_dry_run_no_side_effects(tmp_path, capsys):
    vault = tmp_path / "vault"
    ok = shim.write_vault_alert(
        vault, Path("70-debugging") / "slo-alert-foo.md",
        "---\ntype: slo-alert\n---\nbody\n", dry_run=True,
    )
    assert ok
    assert not (vault / "70-debugging" / "slo-alert-foo.md").exists()
    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out


# --------------------------------------------------------------------- #
# Slack routing                                                         #
# --------------------------------------------------------------------- #


def _violation_result() -> shim.SLOResult:
    return shim.SLOResult(
        name="edge_create_error_rate",
        status="violation",
        actual=0.02,
        threshold=0.01,
        direction="less_than",
        metric_name="edges_created_total",
    )


def test_slack_skip_when_webhook_missing(monkeypatch):
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)
    with patch("scripts.slo_alert_check.urllib.request.urlopen") as mock_open:
        mock_open.side_effect = AssertionError("should not be called")
        ok = shim.post_slack(None, _violation_result(), dry_run=False)
        assert ok is False
        assert not mock_open.called


def test_slack_failure_does_not_raise():
    with patch(
        "scripts.slo_alert_check.urllib.request.urlopen",
        side_effect=RuntimeError("boom"),
    ):
        ok = shim.post_slack("https://example/hook", _violation_result(), dry_run=False)
    assert ok is False  # fail-open


def test_slack_dry_run_prints_payload(capsys):
    ok = shim.post_slack("https://example/hook", _violation_result(), dry_run=True)
    assert ok
    out = capsys.readouterr().out
    assert "[DRY-RUN]" in out and "SLO VIOLATION" in out


# --------------------------------------------------------------------- #
# run_check orchestration (dry-run, no-side-effect path)                #
# --------------------------------------------------------------------- #


def test_run_check_dry_run_no_http_no_writes(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)

    def _fake_fetch(url, timeout=5.0):
        return _enqueue_violation_text()

    def _fake_neo4j():
        return None

    monkeypatch.setattr(shim, "fetch_metrics", _fake_fetch)
    monkeypatch.setattr(shim, "fetch_neo4j_stats", _fake_neo4j)
    # Any urlopen attempt would fail this test.
    with patch(
        "scripts.slo_alert_check.urllib.request.urlopen",
        side_effect=AssertionError("dry-run must not HTTP"),
    ):
        args = shim._parse_args(
            [
                "--vault-path",
                str(vault),
                "--dry-run",
                "--metrics-url",
                "http://unused/metrics",
                "--skip-neo4j",
            ]
        )
        exit_code = shim.run_check(args)
    assert exit_code == 0
    # No real vault writes despite a violation in the fake metrics.
    assert not any((vault / "70-debugging").iterdir())


def test_run_check_fail_on_violation_exit_code(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.setattr(
        shim, "fetch_metrics", lambda url, timeout=5.0: _enqueue_violation_text()
    )
    monkeypatch.setattr(shim, "fetch_neo4j_stats", lambda: None)
    args = shim._parse_args(
        [
            "--vault-path",
            str(vault),
            "--dry-run",
            "--fail-on-violation",
            "--skip-neo4j",
        ]
    )
    rc = shim.run_check(args)
    assert rc == 1


def test_run_check_green_exit_code_zero(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.setattr(
        shim, "fetch_metrics", lambda url, timeout=5.0: _green_metrics_text()
    )
    monkeypatch.setattr(shim, "fetch_neo4j_stats", lambda: None)
    args = shim._parse_args(
        [
            "--vault-path",
            str(vault),
            "--fail-on-violation",
            "--skip-neo4j",
        ]
    )
    rc = shim.run_check(args)
    assert rc == 0


def test_fetch_metrics_fail_open_on_http_error(monkeypatch):
    with patch(
        "scripts.slo_alert_check.urllib.request.urlopen",
        side_effect=RuntimeError("network down"),
    ):
        assert shim.fetch_metrics("http://unused/metrics") == ""


# --------------------------------------------------------------------- #
# Markdown shape                                                        #
# --------------------------------------------------------------------- #


def test_build_alert_markdown_has_required_frontmatter():
    content = shim.build_alert_markdown(_violation_result())
    # Key fields the vault expects.
    for needle in (
        "type: slo-alert",
        "plan_id: PLAN-0759-assoc-linking",
        "slo: edge_create_error_rate",
        "status: violation",
        '"#type/alert"',
        '"#source/slo-alert-check"',
    ):
        assert needle in content, f"missing {needle!r} in alert markdown"


# --------------------------------------------------------------------- #
# Path-traversal defense (Sprint 19 review follow-ups)                  #
# --------------------------------------------------------------------- #


def test_safe_slo_token_sanitizes_traversal():
    # A malicious config with traversal sequences must not survive into a
    # filename: path separators, dots, and anything outside [a-z0-9_-] get
    # replaced with underscores, and the token is lowercased.
    token = shim._safe_slo_token("../../etc/passwd")
    assert "/" not in token
    assert ".." not in token
    # `../../etc/passwd` -> 6 non-safe chars (../../ -> ______) + etc + _
    # + passwd = "______etc_passwd".
    assert token == "______etc_passwd"
    # Uppercase + whitespace + shell metacharacters all collapse.
    assert shim._safe_slo_token("Foo BAR;rm -rf /") == "foo_bar_rm_-rf__"


def test_write_vault_alert_rejects_escape(tmp_path):
    # A relative path that resolves outside the vault root must be rejected
    # even if the sanitizer is somehow bypassed. The file must not land on
    # disk anywhere, inside or outside the vault.
    vault = tmp_path / "vault"
    vault.mkdir()
    outside_canary = tmp_path / "outside.md"
    assert not outside_canary.exists()
    ok = shim.write_vault_alert(
        vault,
        Path("..") / "outside.md",
        "---\ntype: slo-alert\n---\nbody\n",
        dry_run=False,
    )
    assert ok is False
    assert not outside_canary.exists()


def test_vault_write_refuses_human_note_with_literal_type_string(tmp_path):
    # A human-authored note whose body contains the literal text
    # `type: slo-alert` (e.g., a design doc quoting the schema) must NOT
    # be clobbered. Only the frontmatter's `type` field counts.
    vault = tmp_path / "vault"
    folder = vault / "70-debugging"
    folder.mkdir(parents=True)
    target = folder / "slo-alert-foo-2026-04-23.md"
    target.write_text(
        "---\n"
        "type: research\n"
        "author: human\n"
        "---\n\n"
        "# Design doc\n\n"
        "The alert frontmatter looks like `type: slo-alert` and is used "
        "by the shim to gate clobbers. Do not overwrite me.\n",
        encoding="utf-8",
    )
    content = shim.build_alert_markdown(_violation_result())
    ok = shim.write_vault_alert(
        vault,
        Path("70-debugging") / "slo-alert-foo-2026-04-23.md",
        content,
        dry_run=False,
    )
    assert ok is False
    assert "Do not overwrite me" in target.read_text()


# --------------------------------------------------------------------- #
# End-to-end non-dry run (Sprint 19 review follow-ups)                  #
# --------------------------------------------------------------------- #


def test_run_check_writes_violation_file_end_to_end(tmp_path, monkeypatch):
    from datetime import date as _date

    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)

    monkeypatch.setattr(
        shim, "fetch_metrics", lambda url, timeout=5.0: _enqueue_violation_text()
    )
    monkeypatch.setattr(shim, "fetch_neo4j_stats", lambda: None)
    args = shim._parse_args(
        [
            "--vault-path",
            str(vault),
            "--skip-neo4j",
        ]
    )
    rc = shim.run_check(args)
    # No --fail-on-violation -> exit 0 even though a violation fires.
    assert rc == 0

    today_iso = _date.today().isoformat()
    expected = (
        vault
        / "70-debugging"
        / f"slo-alert-p95_similarity_link_enqueue-{today_iso}.md"
    )
    assert expected.exists(), f"violation file not written at {expected}"

    body = expected.read_text(encoding="utf-8")
    # Frontmatter parses and carries the expected schema fields.
    fm_type = shim._parse_frontmatter_type(body)
    assert fm_type == "slo-alert"
    assert "status: violation" in body
    assert "slo: p95_similarity_link_enqueue" in body
    assert body.startswith("---\n")
