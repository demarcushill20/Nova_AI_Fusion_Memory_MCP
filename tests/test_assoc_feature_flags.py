"""Tests for Associative Linking feature flags (PLAN-0759 / ADR-0759 — Sprint 1).

Sprint 1 of PLAN-0759 only declares 8 ``ASSOC_*`` feature flags in
``app.config.Settings``. No code reads them yet, so this module only needs to
prove they exist and default to ``False``.

The test deliberately:

- imports the ``Settings`` class directly (no global ``settings`` singleton);
- bypasses any local ``.env`` file via ``_env_file=None``;
- uses ``monkeypatch.delenv`` to strip any ``ASSOC_*`` keys from ``os.environ``
  before instantiation, so a polluted shell (e.g. ``ASSOC_GRAPH_RECALL_ENABLED=true``
  exported in a dev shell or CI job) cannot smuggle a ``True`` past the baseline —
  ``_env_file=None`` alone is not enough because Pydantic's ``case_sensitive=False``
  still reads from the process environment;
- supplies the two required Pinecone fields inline so ``Settings()`` can
  instantiate in a clean test environment;
- stays completely self-contained: no Neo4j, no Pinecone, no OpenAI, no
  network, no fixtures beyond the built-in ``monkeypatch``.

If a future PR flips any of these defaults to ``True`` without going through
the PLAN-0759 phase gate, this test will fail and force the conversation.
"""

from __future__ import annotations

import pytest

from app.config import Settings


# All 8 ``ASSOC_*`` flags introduced by Sprint 1. Keep this list in sync with
# the block in ``app/config.py`` — if a 9th flag is added, this list and the
# assertion below should both be updated.
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


def test_all_assoc_flags_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every PLAN-0759 ``ASSOC_*`` flag must default to ``False``.

    Sprint 1 contract: declarations only, zero behavior change. Any flag that
    silently flips to ``True`` would smuggle behavior into production ahead of
    the phase gate it belongs to, so this assertion is the safety net.
    """
    # Strip every ASSOC_* key (and its lowercase alias, since Pydantic's
    # case_sensitive=False will read either form) from os.environ, so the
    # assertion reflects declared defaults and not shell pollution.
    for flag in ASSOC_FLAG_NAMES:
        monkeypatch.delenv(flag, raising=False)
        monkeypatch.delenv(flag.lower(), raising=False)

    settings = Settings(
        _env_file=None,
        PINECONE_API_KEY="test-key-not-used",
        PINECONE_ENV="test-env-not-used",
    )

    # Sanity check: we actually got 8 flags, not a typo'd subset.
    assert len(ASSOC_FLAG_NAMES) == 8, (
        f"ASSOC_FLAG_NAMES should list exactly 8 flags, found {len(ASSOC_FLAG_NAMES)}"
    )

    for flag in ASSOC_FLAG_NAMES:
        assert hasattr(settings, flag), (
            f"Settings is missing required PLAN-0759 flag: {flag}"
        )
        value = getattr(settings, flag)
        assert value is False, (
            f"PLAN-0759 flag {flag} must default to False (Sprint 1 contract); "
            f"got {value!r}"
        )
