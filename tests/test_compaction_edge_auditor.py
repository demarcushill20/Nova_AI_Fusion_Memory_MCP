"""Hermetic tests for scripts/compaction_edge_auditor.py (Sprint 21 Phase 4).

These tests exercise the audit shim end-to-end without touching Neo4j,
the real vault, or Slack. All external surfaces are monkeypatched.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import compaction_edge_auditor as aud


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _run(argv, monkeypatch, counts_or_none):
    """Invoke ``run_audit`` with ``fetch_compaction_counts`` stubbed.

    ``counts_or_none`` is returned verbatim from the fetch stub. Pass
    ``None`` to simulate a Neo4j connection error / skipped fetch.
    """
    captured: dict = {}

    def _fake_fetch(since_iso):
        captured["since_iso"] = since_iso
        return counts_or_none

    monkeypatch.setattr(aud, "fetch_compaction_counts", _fake_fetch)
    args = aud._parse_args(argv)
    rc = aud.run_audit(args)
    return rc, captured


def _parse_summary(captured_stdout: str) -> dict:
    """Parse the ``compaction_audit ...`` stdout summary line into a dict."""
    match = re.search(
        r"compaction_audit recent_(\d+)h=(\d+) total=(\d+) status=(\S+)",
        captured_stdout,
    )
    assert match, f"summary line not found in stdout: {captured_stdout!r}"
    return {
        "lookback_hours": int(match.group(1)),
        "recent": int(match.group(2)),
        "total": int(match.group(3)),
        "status": match.group(4),
    }


# --------------------------------------------------------------------- #
# classify_status                                                       #
# --------------------------------------------------------------------- #


def test_classify_status_recent_positive_is_ok():
    assert aud.classify_status({"recent": 3, "total": 10}) == "ok"


def test_classify_status_dead_letter_when_all_zero():
    assert aud.classify_status({"recent": 0, "total": 0}) == "alert_dead_letter"


def test_classify_status_regression_when_recent_zero_total_positive():
    assert (
        aud.classify_status({"recent": 0, "total": 42}) == "alert_regression"
    )


def test_classify_status_none_counts_is_ok():
    # Connection errors must NOT alert — operator sees the log.
    assert aud.classify_status(None) == "ok"


# --------------------------------------------------------------------- #
# End-to-end orchestration: three status branches                       #
# --------------------------------------------------------------------- #


def test_zero_recent_zero_total_triggers_dead_letter(
    tmp_path, monkeypatch, capsys
):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)

    rc, _ = _run(
        ["--vault-path", str(vault)],
        monkeypatch,
        {"recent": 0, "total": 0, "sample_run_ids": []},
    )
    assert rc == 0

    out = capsys.readouterr().out
    parsed = _parse_summary(out)
    assert parsed["status"] == "alert_dead_letter"
    assert parsed["recent"] == 0
    assert parsed["total"] == 0

    # Vault note landed with the dead-letter body marker.
    from datetime import date

    today_iso = date.today().isoformat()
    expected = (
        vault
        / "70-debugging"
        / f"compaction-audit-alert_dead_letter-{today_iso}.md"
    )
    assert expected.exists()
    body = expected.read_text(encoding="utf-8")
    assert "type: compaction-audit" in body
    assert "status: alert_dead_letter" in body
    assert "dead-letter" in body


def test_zero_recent_positive_total_triggers_regression(
    tmp_path, monkeypatch, capsys
):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)

    rc, _ = _run(
        ["--vault-path", str(vault)],
        monkeypatch,
        {"recent": 0, "total": 127, "sample_run_ids": []},
    )
    assert rc == 0

    parsed = _parse_summary(capsys.readouterr().out)
    assert parsed["status"] == "alert_regression"
    assert parsed["total"] == 127

    from datetime import date

    today_iso = date.today().isoformat()
    expected = (
        vault
        / "70-debugging"
        / f"compaction-audit-alert_regression-{today_iso}.md"
    )
    assert expected.exists()
    body = expected.read_text(encoding="utf-8")
    assert "status: alert_regression" in body
    assert "regression" in body
    # Dead-letter file must NOT exist for this variant.
    dead_letter_path = (
        vault
        / "70-debugging"
        / f"compaction-audit-alert_dead_letter-{today_iso}.md"
    )
    assert not dead_letter_path.exists()


def test_positive_recent_is_ok_no_writes_no_slack(
    tmp_path, monkeypatch, capsys
):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.setenv("SLACK_ALERT_WEBHOOK", "https://example/hook")

    with patch(
        "scripts.compaction_edge_auditor.urllib.request.urlopen",
        side_effect=AssertionError("ok path must not POST slack"),
    ):
        rc, _ = _run(
            ["--vault-path", str(vault)],
            monkeypatch,
            {"recent": 7, "total": 42, "sample_run_ids": ["abc", "def"]},
        )
    assert rc == 0

    parsed = _parse_summary(capsys.readouterr().out)
    assert parsed["status"] == "ok"
    assert parsed["recent"] == 7

    # No vault writes on ok.
    assert not any((vault / "70-debugging").iterdir())


# --------------------------------------------------------------------- #
# CLI flag behavior                                                     #
# --------------------------------------------------------------------- #


def test_dry_run_writes_nothing_but_still_emits_summary(
    tmp_path, monkeypatch, capsys
):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.setenv("SLACK_ALERT_WEBHOOK", "https://example/hook")

    with patch(
        "scripts.compaction_edge_auditor.urllib.request.urlopen",
        side_effect=AssertionError("dry-run must not POST slack"),
    ):
        rc, _ = _run(
            ["--vault-path", str(vault), "--dry-run"],
            monkeypatch,
            {"recent": 0, "total": 0, "sample_run_ids": []},
        )
    assert rc == 0

    out = capsys.readouterr().out
    assert _parse_summary(out)["status"] == "alert_dead_letter"
    # Dry-run printed the preview but wrote nothing.
    assert "[DRY-RUN]" in out
    assert not any((vault / "70-debugging").iterdir())


def test_fail_on_alert_exits_one_on_alert(tmp_path, monkeypatch, capsys):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)

    rc, _ = _run(
        ["--vault-path", str(vault), "--fail-on-alert"],
        monkeypatch,
        {"recent": 0, "total": 0, "sample_run_ids": []},
    )
    assert rc == 1


def test_fail_on_alert_exits_zero_on_ok(tmp_path, monkeypatch, capsys):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)

    rc, _ = _run(
        ["--vault-path", str(vault), "--fail-on-alert"],
        monkeypatch,
        {"recent": 5, "total": 100, "sample_run_ids": []},
    )
    assert rc == 0


def test_skip_neo4j_returns_ok_without_calling_fetch(
    tmp_path, monkeypatch, capsys
):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)

    # If the fetch is called, blow up — --skip-neo4j must not touch it.
    def _bang(since_iso):
        raise AssertionError("fetch_compaction_counts must not run")

    monkeypatch.setattr(aud, "fetch_compaction_counts", _bang)
    args = aud._parse_args(["--vault-path", str(vault), "--skip-neo4j"])
    rc = aud.run_audit(args)
    assert rc == 0
    parsed = _parse_summary(capsys.readouterr().out)
    assert parsed["status"] == "ok"
    # No vault writes.
    assert not any((vault / "70-debugging").iterdir())


# --------------------------------------------------------------------- #
# External-boundary fail-open behavior                                  #
# --------------------------------------------------------------------- #


def test_neo4j_connection_error_is_fail_open_as_ok(
    tmp_path, monkeypatch, capsys
):
    """fetch returning None (connection error) must NOT alert."""
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    rc, _ = _run(["--vault-path", str(vault)], monkeypatch, None)
    assert rc == 0
    parsed = _parse_summary(capsys.readouterr().out)
    assert parsed["status"] == "ok"
    assert not any((vault / "70-debugging").iterdir())


def test_vault_write_error_does_not_raise(tmp_path, monkeypatch, capsys):
    """A vault-write exception must be swallowed; auditor still exits 0."""
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)

    def _bang(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(aud, "write_vault_alert", _bang)
    # An inner exception must not propagate; the module-level try/except
    # in main() catches anything leaking from run_audit. We call main()
    # (not run_audit) here to cover the outer shield as well.
    argv = ["--vault-path", str(vault)]
    # Stub fetch to return a dead-letter triple.
    monkeypatch.setattr(
        aud,
        "fetch_compaction_counts",
        lambda since_iso: {
            "recent": 0,
            "total": 0,
            "sample_run_ids": [],
        },
    )
    rc = aud.main(argv)
    assert rc == 0  # fail-open
    out = capsys.readouterr().out
    # Summary line still emitted before the write attempt.
    assert "compaction_audit" in out


def test_slack_post_error_does_not_raise(tmp_path, monkeypatch, capsys):
    """Slack POST exceptions must be swallowed."""
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.setenv("SLACK_ALERT_WEBHOOK", "https://example/hook")

    with patch(
        "scripts.compaction_edge_auditor.urllib.request.urlopen",
        side_effect=RuntimeError("slack down"),
    ):
        rc, _ = _run(
            ["--vault-path", str(vault)],
            monkeypatch,
            {"recent": 0, "total": 0, "sample_run_ids": []},
        )
    assert rc == 0  # fail-open through slack boundary


# --------------------------------------------------------------------- #
# Idempotency                                                           #
# --------------------------------------------------------------------- #


def test_same_day_rerun_is_idempotent(tmp_path, monkeypatch, capsys):
    """The 2nd run the same day must not overwrite the 1st.

    The reused ``write_vault_alert`` helper gates overwrites on
    ``frontmatter type == "slo-alert"``; our note has
    ``type: compaction-audit``, so the 2nd run's overwrite attempt is
    refused (returns False) and the original body is preserved.
    """
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)

    rc1, _ = _run(
        ["--vault-path", str(vault)],
        monkeypatch,
        {"recent": 0, "total": 0, "sample_run_ids": []},
    )
    assert rc1 == 0
    from datetime import date

    today_iso = date.today().isoformat()
    target = (
        vault
        / "70-debugging"
        / f"compaction-audit-alert_dead_letter-{today_iso}.md"
    )
    assert target.exists()
    first_body = target.read_text(encoding="utf-8")

    # Second run same day with a mutated sample id; body would change
    # if the helper overwrote — it must not.
    rc2, _ = _run(
        ["--vault-path", str(vault)],
        monkeypatch,
        {
            "recent": 0,
            "total": 0,
            "sample_run_ids": ["second-run-sentinel"],
        },
    )
    assert rc2 == 0
    second_body = target.read_text(encoding="utf-8")
    assert first_body == second_body
    assert "second-run-sentinel" not in second_body


# --------------------------------------------------------------------- #
# Lookback window plumbing                                              #
# --------------------------------------------------------------------- #


def _parse_iso(s: str) -> float:
    """Parse an auditor-emitted ISO-Z timestamp into epoch seconds."""
    from datetime import datetime, timezone
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    ).timestamp()


def test_lookback_hours_shapes_since_iso(tmp_path, monkeypatch):
    """--lookback-hours=6 must pass through as ~now-6h ISO."""
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)

    before = time.time()
    rc, captured = _run(
        ["--vault-path", str(vault), "--lookback-hours", "6"],
        monkeypatch,
        {"recent": 1, "total": 1, "sample_run_ids": []},
    )
    after = time.time()
    assert rc == 0

    since = _parse_iso(captured["since_iso"])
    six_hours = 6 * 3600
    # since should be within a few seconds of (now - 6h).
    assert before - six_hours - 5 <= since <= after - six_hours + 5


def test_lookback_hours_default_is_24(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)

    before = time.time()
    rc, captured = _run(
        ["--vault-path", str(vault)],
        monkeypatch,
        {"recent": 1, "total": 1, "sample_run_ids": []},
    )
    after = time.time()
    assert rc == 0

    since = _parse_iso(captured["since_iso"])
    day = 24 * 3600
    assert before - day - 5 <= since <= after - day + 5


# --------------------------------------------------------------------- #
# Stdout summary line shape — always emitted                            #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "counts,expected_status",
    [
        ({"recent": 0, "total": 0, "sample_run_ids": []}, "alert_dead_letter"),
        (
            {"recent": 0, "total": 99, "sample_run_ids": []},
            "alert_regression",
        ),
        ({"recent": 3, "total": 99, "sample_run_ids": []}, "ok"),
        (None, "ok"),
    ],
)
def test_stdout_summary_line_always_emitted(
    tmp_path, monkeypatch, capsys, counts, expected_status
):
    vault = tmp_path / "vault"
    (vault / "70-debugging").mkdir(parents=True)
    monkeypatch.delenv("SLACK_ALERT_WEBHOOK", raising=False)

    rc, _ = _run(["--vault-path", str(vault)], monkeypatch, counts)
    assert rc == 0
    parsed = _parse_summary(capsys.readouterr().out)
    assert parsed["status"] == expected_status
    assert parsed["lookback_hours"] == 24
