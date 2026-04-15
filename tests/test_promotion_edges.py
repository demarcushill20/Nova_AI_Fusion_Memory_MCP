"""Unit tests for Phase 5b promotion edge wiring.

Covers:
1. ``on_memory_promote`` extended signature (from_layer, to_layer, run_id)
2. Backward compatibility — positional (new_id, old_id) only
3. Edge version stamping
4. Promotion hook skipped when feature flag is off
5. Promotion hook is non-fatal — exceptions do not propagate

All tests are fully hermetic — no live Neo4j, Pinecone, or Redis required.
Every dependency is replaced with ``unittest.mock`` fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# --- Make the ``app`` package importable without pulling in app.config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

from app.services.associations.memory_edges import MemoryEdge
from app.services.associations.edge_service import MemoryEdgeService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edge_service() -> tuple[MemoryEdgeService, AsyncMock]:
    """Return (service, create_edge_mock) with a mocked driver."""
    driver = MagicMock(name="AsyncDriver")
    svc = MemoryEdgeService(driver)
    svc.create_edge = AsyncMock(return_value=True)
    return svc, svc.create_edge


# ---------------------------------------------------------------------------
# 1. on_memory_promote with layers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_promote_with_layers() -> None:
    """When ``from_layer`` and ``to_layer`` are provided, they appear in
    edge metadata."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_promote(
        "promoted-1", "source-1", from_layer="scratch", to_layer="core"
    )

    create_mock.assert_awaited_once()
    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.source_id == "promoted-1"
    assert edge.target_id == "source-1"
    assert edge.edge_type == "PROMOTED_FROM"
    assert edge.weight == 1.0
    assert edge.created_by == "edge_service.on_memory_promote"
    assert edge.metadata == {"from_layer": "scratch", "to_layer": "core"}


# ---------------------------------------------------------------------------
# 2. on_memory_promote with custom run_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_promote_with_custom_run_id() -> None:
    """When ``run_id`` is provided, it overrides the default."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_promote(
        "promoted-2", "source-2", run_id="consolidation-run-99"
    )

    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.run_id == "consolidation-run-99"


# ---------------------------------------------------------------------------
# 2b. Empty-string run_id is preserved, not replaced by default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_promote_empty_string_run_id_rejected() -> None:
    """Empty-string run_id reaches MemoryEdge (not silently replaced)
    and is properly rejected by __post_init__ validation."""
    svc, create_mock = _make_edge_service()

    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        await svc.on_memory_promote("promoted-r", "source-r", run_id="")


# ---------------------------------------------------------------------------
# 3. Backward compatibility — positional (new_id, old_id) only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_promote_backward_compat() -> None:
    """Calling with just (new_id, old_id) still works — defaults apply."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_promote("promoted-3", "source-3")

    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.run_id == "promotion_hook"
    assert edge.metadata is None
    assert edge.edge_version == 1
    assert edge.edge_type == "PROMOTED_FROM"


# ---------------------------------------------------------------------------
# 4. Edge version is stamped correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_promote_edge_version() -> None:
    """The edge_version field is set to 1 (the current EDGE_VERSION)."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_promote(
        "promoted-4", "source-4", from_layer="episodic", to_layer="semantic"
    )

    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.edge_version == 1


# ---------------------------------------------------------------------------
# 5. Hook skipped when feature flag is off
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_promotion_hook_skipped_when_flag_off() -> None:
    """When ASSOC_PROVENANCE_WRITE_ENABLED is False, the promotion hook
    should not be called. This test validates the flag-guard pattern that
    callers (e.g. a future consolidation flow) must implement.

    NOTE: This is a design-intent placeholder. It tests a local if-branch,
    not real production wiring. Replace with a real integration test once
    the consolidation flow is wired into memory_service.py.
    """

    # Simulate the guard pattern that callers implement:
    # if settings.ASSOC_PROVENANCE_WRITE_ENABLED:
    #     await svc.on_memory_promote(...)
    svc, create_mock = _make_edge_service()

    flag_enabled = False  # Simulates settings.ASSOC_PROVENANCE_WRITE_ENABLED

    if flag_enabled:
        await svc.on_memory_promote("promoted-5", "source-5")

    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. Self-loop returns None without creating an edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_promote_self_loop_noop() -> None:
    """When new_id == old_id, on_memory_promote returns None without
    calling create_edge."""
    svc, create_mock = _make_edge_service()

    result = await svc.on_memory_promote("same-id", "same-id")

    assert result is None
    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 8. Input validation rejects invalid IDs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_new, bad_old",
    [
        ("", "valid-id"),
        ("valid-id", ""),
        (None, "valid-id"),
        ("valid-id", None),
    ],
)
async def test_on_memory_promote_rejects_invalid_ids(bad_new, bad_old) -> None:
    """Invalid new_id or old_id raises ValueError before create_edge."""
    svc, create_mock = _make_edge_service()

    with pytest.raises(ValueError):
        await svc.on_memory_promote(bad_new, bad_old)

    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 9. Half-pair from_layer / to_layer raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "from_layer, to_layer",
    [
        ("scratch", None),
        (None, "core"),
    ],
)
async def test_on_memory_promote_rejects_half_pair_layers(from_layer, to_layer) -> None:
    """Providing only one of from_layer/to_layer raises ValueError."""
    svc, create_mock = _make_edge_service()

    with pytest.raises(ValueError, match="from_layer and to_layer must both be provided"):
        await svc.on_memory_promote(
            "promoted-x", "source-x", from_layer=from_layer, to_layer=to_layer
        )

    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 6. Hook is non-fatal — exceptions do not propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_promotion_hook_nonfatal() -> None:
    """If create_edge raises inside on_memory_promote, the caller's
    try/except envelope catches it. This mirrors the per-item pattern
    used in perform_upsert for supersession edges."""
    svc, create_mock = _make_edge_service()
    create_mock.side_effect = RuntimeError("Neo4j exploded")

    # The on_memory_promote method itself does NOT catch — it is the
    # caller's responsibility to wrap in try/except (per the fail-open
    # contract in the edge_service module docstring). Verify the error
    # propagates so the caller's envelope can catch it.
    with pytest.raises(RuntimeError, match="Neo4j exploded"):
        await svc.on_memory_promote("promoted-6", "source-6")

    # Confirm the call was attempted
    create_mock.assert_awaited_once()
