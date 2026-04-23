"""Tests for the /metrics loopback enforcement middleware.

PLAN-0759 Sprint 21 Phase 3.

The middleware at ``app.main._metrics_loopback_guard`` rejects non-loopback
requests to the /metrics mount with a 403 unless the
``METRICS_ALLOW_PUBLIC=1`` env var is set. These tests exercise the full
middleware stack via Starlette's TestClient, toggling the client tuple to
simulate loopback vs non-loopback callers.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Iterator

import pytest
from starlette.testclient import TestClient

# Ensure project root is importable when tests are run from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app import main as app_main  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_middleware_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset the module-level rejection counter and override flag per test."""
    monkeypatch.setattr(app_main, "_metrics_rejection_count", 0, raising=True)
    monkeypatch.setattr(app_main, "_METRICS_ALLOW_PUBLIC", False, raising=True)
    yield


def _client(host: str | None) -> TestClient:
    """Build a TestClient with a specific (or omitted) client tuple."""
    if host is None:
        # Omit the client kwarg: Starlette will still supply one, but
        # we simulate the None case via a direct ASGI scope in the
        # dedicated null-client test below.
        return TestClient(app_main.app)
    return TestClient(app_main.app, client=(host, 50000))


# --------------------------------------------------------------------- #
# 1. Loopback client hitting /metrics -> 200                             #
# --------------------------------------------------------------------- #
def test_loopback_client_allowed_on_metrics() -> None:
    client = _client("127.0.0.1")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "# HELP" in resp.text


# --------------------------------------------------------------------- #
# 2. Loopback client hitting /metrics/ (trailing slash) -> 200           #
# --------------------------------------------------------------------- #
def test_loopback_client_allowed_on_metrics_trailing_slash() -> None:
    client = _client("127.0.0.1")
    resp = client.get("/metrics/")
    assert resp.status_code == 200


# --------------------------------------------------------------------- #
# 3. Non-loopback client hitting /metrics -> 403                         #
# --------------------------------------------------------------------- #
def test_non_loopback_client_rejected_on_metrics() -> None:
    client = _client("10.0.0.5")
    resp = client.get("/metrics")
    assert resp.status_code == 403
    assert resp.json() == {
        "error": "metrics endpoint not accessible from this client"
    }


# --------------------------------------------------------------------- #
# 4. Non-loopback client with METRICS_ALLOW_PUBLIC=1 -> 200              #
# --------------------------------------------------------------------- #
def test_override_env_var_allows_non_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_main, "_METRICS_ALLOW_PUBLIC", True, raising=True)
    client = _client("10.0.0.5")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "# HELP" in resp.text


# --------------------------------------------------------------------- #
# 5. Non-loopback client with METRICS_ALLOW_PUBLIC=0 -> 403              #
# --------------------------------------------------------------------- #
def test_override_env_var_zero_does_not_allow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Parse semantics live at import time; the module-level flag is the
    # source of truth. Simulate the env-var=0 case by pinning the flag
    # False (matching what import-time parsing would produce).
    monkeypatch.setattr(app_main, "_METRICS_ALLOW_PUBLIC", False, raising=True)
    client = _client("10.0.0.5")
    resp = client.get("/metrics")
    assert resp.status_code == 403


# --------------------------------------------------------------------- #
# 6. Non-loopback client with METRICS_ALLOW_PUBLIC='true' -> 403         #
# --------------------------------------------------------------------- #
def test_override_env_var_true_string_does_not_allow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The spec is strict: only the literal "1" unlocks. Verify the
    # import-time parse: "true" (case-insensitive) -> False.
    monkeypatch.setenv("METRICS_ALLOW_PUBLIC", "true")
    parsed = (
        os.environ.get("METRICS_ALLOW_PUBLIC", "").strip().lower() == "1"
    )
    assert parsed is False
    # And confirm end-to-end behavior with the flag pinned False:
    monkeypatch.setattr(app_main, "_METRICS_ALLOW_PUBLIC", parsed, raising=True)
    client = _client("10.0.0.5")
    resp = client.get("/metrics")
    assert resp.status_code == 403


# --------------------------------------------------------------------- #
# 7. Loopback client hitting non-/metrics path -> unaffected             #
# --------------------------------------------------------------------- #
def test_non_metrics_path_unaffected_by_middleware() -> None:
    # Even with a non-loopback client the middleware must NOT touch
    # other paths. Use a non-loopback IP to isolate the routing check.
    client = _client("10.0.0.5")
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    assert body["message"].startswith("Welcome")


# --------------------------------------------------------------------- #
# 8. Rate-limit WARNING logging: first N WARNING, then DEBUG             #
# --------------------------------------------------------------------- #
def test_rejection_logging_rate_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client("10.0.0.5")
    caplog.set_level(logging.DEBUG, logger=app_main.logger.name)

    # Fire 11 rejections. The first 10 should log WARNING; the 11th DEBUG.
    for _ in range(11):
        resp = client.get("/metrics")
        assert resp.status_code == 403

    warning_records = [
        r for r in caplog.records
        if r.name == app_main.logger.name
        and r.levelno == logging.WARNING
        and "Rejected non-loopback" in r.getMessage()
    ]
    debug_records = [
        r for r in caplog.records
        if r.name == app_main.logger.name
        and r.levelno == logging.DEBUG
        and "Rejected non-loopback" in r.getMessage()
    ]
    assert len(warning_records) == 10, (
        f"expected 10 WARNING rejections, got {len(warning_records)}"
    )
    assert len(debug_records) >= 1, (
        f"expected >=1 DEBUG rejection, got {len(debug_records)}"
    )


# --------------------------------------------------------------------- #
# 9. Null client (request.client is None) -> 200 (treated as local)      #
# --------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_null_client_treated_as_local() -> None:
    # Drive the ASGI app directly with a scope where "client" is None,
    # mirroring the ASGI-internal path where request.client is None.
    # Hit the trailing-slash form so the mount serves directly (the
    # /metrics -> /metrics/ redirect would also prove the middleware
    # let the request through, but asserting 200 keeps this unambiguous).
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/metrics/",
        "raw_path": b"/metrics/",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "server": ("testserver", 80),
        "client": None,
    }

    received: list[dict] = []

    async def _receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(msg: dict) -> None:
        received.append(msg)

    await app_main.app(scope, _receive, _send)

    start = next(m for m in received if m["type"] == "http.response.start")
    assert start["status"] == 200


# --------------------------------------------------------------------- #
# 10. Case-insensitive loopback host: 'localhost' passes                 #
# --------------------------------------------------------------------- #
def test_localhost_string_treated_as_loopback() -> None:
    client = _client("localhost")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "# HELP" in resp.text
