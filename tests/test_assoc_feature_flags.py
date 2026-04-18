"""Tests for Associative Linking feature flags (PLAN-0759 / ADR-0759).

Validates that all 8 ``ASSOC_*`` flags exist in ``app.config.Settings``
with their expected defaults. Write-path flags default to ``False``;
``ASSOC_GRAPH_RECALL_ENABLED`` was flipped to ``True`` after the Phase 6
gate eval (commit 14ecb3e, 2026-04-16).

The test deliberately:

- imports the ``Settings`` class directly (no global ``settings`` singleton);
- bypasses any local ``.env`` file via ``_env_file=None``;
- uses ``monkeypatch.delenv`` to strip any ``ASSOC_*`` keys from ``os.environ``
  before instantiation, so a polluted shell cannot smuggle values past the
  baseline;
- supplies the two required Pinecone fields inline so ``Settings()`` can
  instantiate in a clean test environment;
- stays completely self-contained: no Neo4j, no Pinecone, no OpenAI, no
  network, no fixtures beyond the built-in ``monkeypatch``.
"""

from __future__ import annotations

import pytest

from app.config import Settings


# All 8 ``ASSOC_*`` flags. Keep this list in sync with ``app/config.py``.
ASSOC_FLAG_NAMES: tuple[str, ...] = (
    "ASSOC_SIMILARITY_WRITE_ENABLED",
    "ASSOC_ENTITY_WRITE_ENABLED",
    "ASSOC_TEMPORAL_WRITE_ENABLED",
    "ASSOC_PROVENANCE_WRITE_ENABLED",
    "ASSOC_COOCCURRENCE_WRITE_ENABLED",
    "ASSOC_TASK_HEURISTIC_WRITE_ENABLED",
    "ASSOC_GRAPH_RECALL_ENABLED",
    "ASSOC_CROSS_PROJECT_ENABLED",
)

# Flags that have been shipped (flipped to True after passing their gate).
SHIPPED_TRUE: dict[str, bool] = {
    "ASSOC_GRAPH_RECALL_ENABLED": True,  # Phase 6, 2026-04-16
}


def test_all_assoc_flags_exist_with_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every PLAN-0759 ``ASSOC_*`` flag must exist with its expected default.

    Write-path flags default to False. Read-path flags that have cleared
    their gate eval default to True (tracked in ``SHIPPED_TRUE``).
    """
    for flag in ASSOC_FLAG_NAMES:
        monkeypatch.delenv(flag, raising=False)
        monkeypatch.delenv(flag.lower(), raising=False)

    settings = Settings(
        _env_file=None,
        PINECONE_API_KEY="test-key-not-used",
        PINECONE_ENV="test-env-not-used",
    )

    assert len(ASSOC_FLAG_NAMES) == 8, (
        f"ASSOC_FLAG_NAMES should list exactly 8 flags, found {len(ASSOC_FLAG_NAMES)}"
    )

    for flag in ASSOC_FLAG_NAMES:
        assert hasattr(settings, flag), (
            f"Settings is missing required PLAN-0759 flag: {flag}"
        )
        value = getattr(settings, flag)
        expected = SHIPPED_TRUE.get(flag, False)
        assert value is expected, (
            f"PLAN-0759 flag {flag} expected {expected!r}, got {value!r}"
        )
