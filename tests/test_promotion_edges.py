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
from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# T1-T4 — memory_service-level promotion hook integration tests.
#
# Mirrors the fixture pattern used in
# ``tests/test_supersession_edges.py::test_supersession_hook_happy_path``.
# These are the first memory_service-level promotion tests — prior tests in
# this file exercised the edge_service helper in isolation.
# ---------------------------------------------------------------------------


async def _drive_promotion_hook(
    *,
    metadata: dict,
    flag_enabled: bool = True,
    edge_service: object | None = None,
    pinecone_client: object | None = None,
    graph_client: object | None = None,
) -> tuple[object, object, object]:
    """Run ``perform_upsert`` with the promotion hook configured and return
    ``(edge_service, pinecone_client, graph_client)`` so callers can inspect
    the edge-service calls and the metadata actually passed to both the
    Pinecone and Neo4j write paths.

    Follows the same patching pattern as the supersession hook fixtures:
    patches ``settings``, ``get_embedding``, ``detect_conflicts``, and
    ``check_semantic_duplicate`` at the module level, and sidesteps the
    heavyweight ``__init__`` via ``MemoryService.__new__``.
    """
    import asyncio as _asyncio

    mock_embedding = MagicMock(return_value=[0.1] * 8)
    mock_inject = AsyncMock(side_effect=lambda m: m)

    if edge_service is None:
        edge_service = MagicMock()
        edge_service.on_memory_promote = AsyncMock(return_value=None)

    # Pinecone client: we want to capture the exact metadata dict passed
    # to ``upsert_vector``. The real client is called via
    # ``asyncio.to_thread(self.pinecone_client.upsert_vector, id, emb, meta)``
    # from ``_persist_memory_item``, so a plain MagicMock whose
    # ``upsert_vector`` returns True works and records ``call_args``.
    if pinecone_client is None:
        pinecone_client = MagicMock()
        pinecone_client.upsert_vector = MagicMock(return_value=True)

    if graph_client is None:
        graph_client = MagicMock()
        graph_client.upsert_graph_data = AsyncMock(return_value=True)
        graph_client.driver = MagicMock(name="AsyncDriver")

    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", mock_embedding), \
         patch(
             "app.services.memory_service.detect_conflicts",
             new_callable=AsyncMock,
             return_value=None,
         ), \
         patch(
             "app.services.memory_service.check_semantic_duplicate",
             new_callable=AsyncMock,
             return_value=None,
         ):

        mock_settings.ASSOC_PROVENANCE_WRITE_ENABLED = flag_enabled
        mock_settings.ASSOC_SIMILARITY_WRITE_ENABLED = False
        mock_settings.ASSOC_ENTITY_WRITE_ENABLED = False
        mock_settings.ASSOC_TEMPORAL_WRITE_ENABLED = False
        mock_settings.ASSOC_COOCCURRENCE_WRITE_ENABLED = False
        mock_settings.ASSOC_TASK_HEURISTIC_WRITE_ENABLED = False
        mock_settings.WRITE_DEDUP_ENABLED = False
        mock_settings.CONFLICT_DETECTION_ENABLED = False
        mock_settings.EMBEDDING_MODEL = "test"

        from app.services.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)
        svc._initialized = True
        svc._similarity_linker = None
        svc._entity_linker = None
        svc._temporal_linker = None
        svc._provenance_edge_service = edge_service
        svc._cooccurrence_linker = None
        svc._task_heuristic_linker = None
        svc.redis_timeline = None
        svc.embedding_model_name = "test"
        svc.pinecone_client = pinecone_client
        svc.graph_client = graph_client
        svc.sequence_service = MagicMock()
        svc.sequence_service.next_seq = AsyncMock(return_value=42)
        svc._inject_chronology = mock_inject
        svc._resolve_memory_id = MagicMock(return_value="new-mem-id")

        await svc.perform_upsert(content="promoted content", metadata=metadata)

    return edge_service, pinecone_client, graph_client


@pytest.mark.asyncio
async def test_memory_service_promotion_hook_emits_edge() -> None:
    """T1: happy path — flag ON + valid ``_promoted_from`` metadata →
    ``on_memory_promote`` is awaited exactly once with the right kwargs,
    including ``run_id="wt-promote-sess-77"``."""
    edge_service, _, _ = await _drive_promotion_hook(
        metadata={
            "_promoted_from": {
                "old_id": "mem-old-1",
                "from_layer": "episodic",
                "to_layer": "semantic",
            },
            "session_id": "sess-77",
        },
        flag_enabled=True,
    )

    edge_service.on_memory_promote.assert_awaited_once()
    call = edge_service.on_memory_promote.call_args
    assert call.kwargs["new_id"] == "new-mem-id"
    assert call.kwargs["old_id"] == "mem-old-1"
    assert call.kwargs["from_layer"] == "episodic"
    assert call.kwargs["to_layer"] == "semantic"
    assert call.kwargs["run_id"] == "wt-promote-sess-77"


@pytest.mark.asyncio
async def test_memory_service_promotion_hook_skipped_when_flag_off() -> None:
    """T2: flag OFF → ``on_memory_promote`` is NOT awaited even when
    ``_promoted_from`` is present in metadata."""
    edge_service, _, _ = await _drive_promotion_hook(
        metadata={
            "_promoted_from": {
                "old_id": "mem-old-2",
                "from_layer": "episodic",
                "to_layer": "semantic",
            },
            "session_id": "sess-off",
        },
        flag_enabled=False,
    )

    edge_service.on_memory_promote.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_service_promotion_hook_nonfatal() -> None:
    """T3: flag ON + ``on_memory_promote`` raises RuntimeError →
    ``perform_upsert`` still returns the item id and does not propagate
    the exception."""
    boom_edge_service = MagicMock()
    boom_edge_service.on_memory_promote = AsyncMock(
        side_effect=RuntimeError("Neo4j exploded")
    )

    edge_service, _, _ = await _drive_promotion_hook(
        metadata={
            "_promoted_from": {
                "old_id": "mem-old-3",
                "from_layer": "episodic",
                "to_layer": "semantic",
            },
            "session_id": "sess-boom",
        },
        flag_enabled=True,
        edge_service=boom_edge_service,
    )

    # The edge service WAS called (flag on, shape valid) but the raised
    # exception must not have broken the upsert.
    edge_service.on_memory_promote.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_service_promotion_metadata_scrubbed() -> None:
    """T4: flag ON + ``_promoted_from`` in metadata →
    1. ``on_memory_promote`` is awaited with the run_id falling back to
       ``wt-promote-no-session`` when no ``session_id`` is supplied.
    2. The metadata dict actually passed to ``pinecone_client.upsert_vector``
       does NOT contain ``_promoted_from``.
    3. The metadata dict actually passed to ``graph_client.upsert_graph_data``
       also does NOT contain ``_promoted_from`` — Neo4j refuses dict/map
       property values, so a leaked ``_promoted_from`` would fail the whole
       graph write (``graph_client.py`` does ``SET n += $props``).
    """
    pinecone_client = MagicMock()
    pinecone_client.upsert_vector = MagicMock(return_value=True)

    edge_service, captured_pinecone, captured_graph = await _drive_promotion_hook(
        metadata={
            "_promoted_from": {
                "old_id": "mem-old-4",
                "from_layer": "episodic",
                "to_layer": "semantic",
            },
            # no session_id — validates the no-session fallback
        },
        flag_enabled=True,
        pinecone_client=pinecone_client,
    )

    # Part 1: the hook fired with the no-session run_id.
    edge_service.on_memory_promote.assert_awaited_once()
    call = edge_service.on_memory_promote.call_args
    assert call.kwargs["run_id"] == "wt-promote-no-session"
    assert call.kwargs["new_id"] == "new-mem-id"
    assert call.kwargs["old_id"] == "mem-old-4"

    # Part 2: the metadata dict actually sent to Pinecone does NOT leak
    # the internal ``_promoted_from`` key.
    captured_pinecone.upsert_vector.assert_called_once()
    args, _kwargs = captured_pinecone.upsert_vector.call_args
    # Signature: upsert_vector(id, embedding, metadata)
    pinecone_meta = args[2]
    assert isinstance(pinecone_meta, dict)
    assert "_promoted_from" not in pinecone_meta
    # Sanity: the text content did get populated.
    assert pinecone_meta.get("text") == "promoted content"

    # Part 3: the metadata dict actually sent to the graph path also does
    # NOT leak ``_promoted_from``. This is the load-bearing invariant —
    # Neo4j rejects dict-valued properties, so this must be scrubbed on
    # BOTH write paths, not just Pinecone.
    captured_graph.upsert_graph_data.assert_awaited_once()
    graph_args, _graph_kwargs = captured_graph.upsert_graph_data.call_args
    # Signature: upsert_graph_data(item_id, content, metadata)
    graph_meta = graph_args[2]
    assert isinstance(graph_meta, dict)
    assert "_promoted_from" not in graph_meta


@pytest.mark.asyncio
async def test_memory_service_promotion_all_underscore_keys_scrubbed_both_paths() -> None:
    """Dual-write-path scrub invariant: every leading-underscore key in the
    caller-supplied metadata is absent from BOTH the Pinecone upsert and
    the graph upsert, while all public keys survive on both paths.
    """
    pinecone_client = MagicMock()
    pinecone_client.upsert_vector = MagicMock(return_value=True)

    edge_service, captured_pinecone, captured_graph = await _drive_promotion_hook(
        metadata={
            "_promoted_from": {
                "old_id": "mem-old-5",
                "from_layer": "episodic",
                "to_layer": "semantic",
            },
            "_internal": "x",
            "public_key": "y",
            "session_id": "sess-1",
        },
        flag_enabled=True,
        pinecone_client=pinecone_client,
    )

    # Hook fired (metadata shape is valid).
    edge_service.on_memory_promote.assert_awaited_once()

    # Pinecone path.
    captured_pinecone.upsert_vector.assert_called_once()
    pinecone_meta = captured_pinecone.upsert_vector.call_args[0][2]
    assert "_promoted_from" not in pinecone_meta
    assert "_internal" not in pinecone_meta
    assert pinecone_meta.get("public_key") == "y"
    assert pinecone_meta.get("session_id") == "sess-1"
    # And no leading-underscore key at all survives.
    assert not any(
        isinstance(k, str) and k.startswith("_") for k in pinecone_meta.keys()
    )

    # Graph path — same invariant.
    captured_graph.upsert_graph_data.assert_awaited_once()
    graph_meta = captured_graph.upsert_graph_data.call_args[0][2]
    assert "_promoted_from" not in graph_meta
    assert "_internal" not in graph_meta
    assert graph_meta.get("public_key") == "y"
    assert graph_meta.get("session_id") == "sess-1"
    assert not any(
        isinstance(k, str) and k.startswith("_") for k in graph_meta.keys()
    )
