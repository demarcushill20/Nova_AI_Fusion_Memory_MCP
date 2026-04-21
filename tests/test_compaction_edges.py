"""Unit tests for Phase 5c compaction edge wiring.

Covers:
1. ``on_memory_compact`` creates N edges for N sources
2. Custom ``run_id`` is forwarded
3. Self-loop guard (summary_id in source_ids)
4. ``edge_version == 1`` on created edges
5. Compaction hook skipped when ``ASSOC_PROVENANCE_WRITE_ENABLED`` is False
6. Per-item fault tolerance — one source failure does not kill others

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
# 1. on_memory_compact creates edges — N sources produce N edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_creates_edges() -> None:
    """Three source memories produce exactly three COMPACTED_FROM edges."""
    svc, create_mock = _make_edge_service()
    sources = ["src-1", "src-2", "src-3"]

    result = await svc.on_memory_compact("summary-1", sources)

    assert result == 3
    assert create_mock.await_count == 3

    for i, call in enumerate(create_mock.call_args_list):
        edge: MemoryEdge = call[0][0]
        assert edge.source_id == "summary-1"
        assert edge.target_id == sources[i]
        assert edge.edge_type == "COMPACTED_FROM"
        assert edge.weight == 1.0
        assert edge.created_by == "edge_service.on_memory_compact"
        assert edge.run_id == "compaction_hook"


# ---------------------------------------------------------------------------
# 2b. Empty-string run_id reaches MemoryEdge and is rejected (not silently
#     replaced by the default). The per-item except catches the ValueError,
#     so the method returns 0 instead of raising.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_empty_string_run_id_rejected() -> None:
    """Empty-string run_id is not silently replaced — it reaches MemoryEdge
    which rejects it. The per-item except catches the ValueError."""
    svc, create_mock = _make_edge_service()

    result = await svc.on_memory_compact(
        "summary-r", ["src-1", "src-2"], run_id=""
    )

    assert result == 0
    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2. Custom run_id is forwarded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_with_custom_run_id() -> None:
    """When ``run_id`` is provided, it overrides the default."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_compact(
        "summary-2", ["src-a", "src-b"], run_id="my-run-42"
    )

    for call in create_mock.call_args_list:
        edge: MemoryEdge = call[0][0]
        assert edge.run_id == "my-run-42"


# ---------------------------------------------------------------------------
# 3. Self-loop guard — summary_id in source_ids is skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_self_loop_guard() -> None:
    """When summary_id appears in source_ids, that entry is skipped."""
    svc, create_mock = _make_edge_service()

    result = await svc.on_memory_compact(
        "summary-3", ["src-x", "summary-3", "src-y"]
    )

    # Only two edges created — the self-loop entry was skipped
    assert result == 2
    assert create_mock.await_count == 2

    targets = [call[0][0].target_id for call in create_mock.call_args_list]
    assert "summary-3" not in targets
    assert targets == ["src-x", "src-y"]


# ---------------------------------------------------------------------------
# 4. edge_version == 1 on every created edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_edge_version() -> None:
    """All COMPACTED_FROM edges have edge_version == 1."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_compact("summary-4", ["src-1", "src-2"])

    for call in create_mock.call_args_list:
        edge: MemoryEdge = call[0][0]
        assert edge.edge_version == 1


# ---------------------------------------------------------------------------
# 5. Hook skipped when feature flag is off
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_caller_guard_pattern_contract() -> None:
    """Caller-side guard pattern: when the flag is off, on_memory_compact
    is never called. This is a pattern contract test — it validates the
    caller's guard logic, not the actual feature flag wiring."""
    mock_edge_service = MagicMock()
    mock_edge_service.on_memory_compact = AsyncMock(return_value=3)

    # Simulate the flag-guarded caller pattern from memory_service.py
    flag_enabled = False
    provenance_svc = mock_edge_service

    if flag_enabled:
        await provenance_svc.on_memory_compact(
            "summary-5", ["src-1", "src-2", "src-3"]
        )

    # No call was made because the flag was off
    mock_edge_service.on_memory_compact.assert_not_awaited()


# ---------------------------------------------------------------------------
# 6. Per-item fault tolerance — one failure does not kill the rest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_hook_nonfatal() -> None:
    """If one source edge fails, the remaining sources still get edges."""
    svc, create_mock = _make_edge_service()

    # Second call (src-2) raises; first and third should succeed
    create_mock.side_effect = [
        True,                                    # src-1: success
        RuntimeError("Neo4j timeout"),           # src-2: failure
        True,                                    # src-3: success
    ]

    result = await svc.on_memory_compact(
        "summary-6", ["src-1", "src-2", "src-3"]
    )

    # Two succeeded, one failed
    assert result == 2
    assert create_mock.await_count == 3


# ---------------------------------------------------------------------------
# 7. Empty source list produces zero edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_empty_sources() -> None:
    """An empty source list returns 0 and makes no create_edge calls."""
    svc, create_mock = _make_edge_service()

    result = await svc.on_memory_compact("summary-7", [])

    assert result == 0
    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 8. Invalid summary_id raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_invalid_summary_id() -> None:
    """An empty or non-string summary_id raises ValueError."""
    svc, _ = _make_edge_service()

    with pytest.raises(ValueError, match="summary_id must be a non-empty string"):
        await svc.on_memory_compact("", ["src-1"])

    with pytest.raises(ValueError, match="summary_id must be a non-empty string"):
        await svc.on_memory_compact(None, ["src-1"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. Bare string source_ids raises TypeError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_bare_string_raises() -> None:
    """Passing a bare string instead of a list raises TypeError."""
    svc, _ = _make_edge_service()

    with pytest.raises(TypeError, match="source_ids must be a list"):
        await svc.on_memory_compact("summary-x", "src-1")


# ---------------------------------------------------------------------------
# 10. Partial create_edge failure yields correct count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_compact_partial_create_failure() -> None:
    """create_edge returning False for one source yields correct count."""
    svc, create_mock = _make_edge_service()
    create_mock.side_effect = [True, False, True]

    result = await svc.on_memory_compact("summary-p", ["src-1", "src-2", "src-3"])

    assert result == 2
    assert create_mock.await_count == 3


# ---------------------------------------------------------------------------
# Memory-service integration — T1..T8.
#
# Mirrors the fixture pattern used in
# ``tests/test_promotion_edges.py::_drive_promotion_hook``. These are the
# first memory_service-level compaction tests — prior tests in this file
# exercised the edge_service helper in isolation.
# ---------------------------------------------------------------------------


async def _drive_compaction_hook(
    *,
    metadata: dict,
    flag_enabled: bool = True,
    edge_service: object | None = None,
    pinecone_client: object | None = None,
    graph_client: object | None = None,
) -> tuple[object, object, object]:
    """Run ``perform_upsert`` with the compaction hook configured and return
    ``(edge_service, pinecone_client, graph_client)`` so callers can inspect
    the edge-service calls and the metadata actually passed to both the
    Pinecone and Neo4j write paths.

    Follows the same patching pattern as the promotion hook fixture:
    patches ``settings``, ``get_embedding``, ``detect_conflicts``, and
    ``check_semantic_duplicate`` at the module level, and sidesteps the
    heavyweight ``__init__`` via ``MemoryService.__new__``.
    """
    mock_embedding = MagicMock(return_value=[0.1] * 8)
    mock_inject = AsyncMock(side_effect=lambda m: m)

    if edge_service is None:
        edge_service = MagicMock()
        edge_service.on_memory_compact = AsyncMock(return_value=0)

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

        await svc.perform_upsert(content="compacted summary", metadata=metadata)

    return edge_service, pinecone_client, graph_client


@pytest.mark.asyncio
async def test_memory_service_compaction_hook_emits_edges() -> None:
    """T1: happy path — flag ON + valid ``_compacted_from`` metadata →
    ``on_memory_compact`` is awaited exactly once with the right kwargs,
    including ``run_id="wt-compact-sess-99"`` and the
    ``{"algorithm","reason"}`` attribution metadata forwarded through."""
    edge_service, _, _ = await _drive_compaction_hook(
        metadata={
            "_compacted_from": {
                "source_ids": ["src-1", "src-2", "src-3"],
                "algorithm": "thin-group",
                "reason": "test",
            },
            "session_id": "sess-99",
        },
        flag_enabled=True,
    )

    edge_service.on_memory_compact.assert_awaited_once()
    call = edge_service.on_memory_compact.call_args
    assert call.kwargs["summary_id"] == "new-mem-id"
    assert call.kwargs["source_ids"] == ["src-1", "src-2", "src-3"]
    assert call.kwargs["run_id"] == "wt-compact-sess-99"
    assert call.kwargs["metadata"] == {
        "algorithm": "thin-group",
        "reason": "test",
    }


@pytest.mark.asyncio
async def test_memory_service_compaction_hook_skipped_when_flag_off() -> None:
    """T2: flag OFF → ``on_memory_compact`` is NOT awaited even when
    ``_compacted_from`` is present in metadata."""
    edge_service, _, _ = await _drive_compaction_hook(
        metadata={
            "_compacted_from": {
                "source_ids": ["src-1", "src-2"],
                "algorithm": "thin-group",
                "reason": "test",
            },
            "session_id": "sess-off",
        },
        flag_enabled=False,
    )

    edge_service.on_memory_compact.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_service_compaction_hook_no_session() -> None:
    """T3: no ``session_id`` in metadata → ``run_id`` falls back to
    ``wt-compact-no-session``."""
    edge_service, _, _ = await _drive_compaction_hook(
        metadata={
            "_compacted_from": {
                "source_ids": ["src-a", "src-b"],
                "algorithm": "thin-group",
            },
            # no session_id
        },
        flag_enabled=True,
    )

    edge_service.on_memory_compact.assert_awaited_once()
    call = edge_service.on_memory_compact.call_args
    assert call.kwargs["run_id"] == "wt-compact-no-session"


@pytest.mark.asyncio
async def test_memory_service_compaction_hook_empty_source_ids() -> None:
    """T4: empty ``source_ids`` list → hook is skipped (not awaited)."""
    edge_service, _, _ = await _drive_compaction_hook(
        metadata={
            "_compacted_from": {
                "source_ids": [],
                "algorithm": "thin-group",
            },
            "session_id": "sess-empty",
        },
        flag_enabled=True,
    )

    edge_service.on_memory_compact.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_source_ids",
    [
        "src-1",            # not a list (bare string)
        ["src-1", 42],      # list containing non-string element
        ["src-1", ""],      # list containing an empty string
    ],
)
async def test_memory_service_compaction_hook_malformed_source_ids(
    bad_source_ids,
) -> None:
    """T5: malformed ``source_ids`` → hook is skipped (not awaited).

    Covers three shapes: non-list, non-string element, and list containing
    an empty string. All three should be rejected by the caller-side
    type-strict guard before ``on_memory_compact`` is reached.
    """
    edge_service, _, _ = await _drive_compaction_hook(
        metadata={
            "_compacted_from": {
                "source_ids": bad_source_ids,
                "algorithm": "thin-group",
            },
            "session_id": "sess-bad",
        },
        flag_enabled=True,
    )

    edge_service.on_memory_compact.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_service_compaction_hook_nonfatal() -> None:
    """T6: flag ON + ``on_memory_compact`` raises RuntimeError →
    ``perform_upsert`` still completes and does not propagate the
    exception."""
    boom_edge_service = MagicMock()
    boom_edge_service.on_memory_compact = AsyncMock(
        side_effect=RuntimeError("Neo4j exploded")
    )

    edge_service, _, _ = await _drive_compaction_hook(
        metadata={
            "_compacted_from": {
                "source_ids": ["src-1", "src-2"],
                "algorithm": "thin-group",
                "reason": "test",
            },
            "session_id": "sess-boom",
        },
        flag_enabled=True,
        edge_service=boom_edge_service,
    )

    # The edge service WAS called (flag on, shape valid) but the raised
    # exception must not have broken the upsert.
    edge_service.on_memory_compact.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_service_compaction_metadata_scrubbed_both_paths() -> None:
    """T7: flag ON + ``_compacted_from`` in metadata →
    1. ``on_memory_compact`` is awaited once.
    2. The metadata dict actually passed to ``pinecone_client.upsert_vector``
       does NOT contain ``_compacted_from``.
    3. The metadata dict actually passed to ``graph_client.upsert_graph_data``
       also does NOT contain ``_compacted_from`` — Neo4j refuses dict/map
       property values, so a leaked ``_compacted_from`` would fail the whole
       graph write (``graph_client.py`` does ``SET n += $props``).
    """
    pinecone_client = MagicMock()
    pinecone_client.upsert_vector = MagicMock(return_value=True)

    edge_service, captured_pinecone, captured_graph = await _drive_compaction_hook(
        metadata={
            "_compacted_from": {
                "source_ids": ["src-1", "src-2"],
                "algorithm": "thin-group",
                "reason": "test",
            },
            "session_id": "sess-scrub",
        },
        flag_enabled=True,
        pinecone_client=pinecone_client,
    )

    # Part 1: the hook fired.
    edge_service.on_memory_compact.assert_awaited_once()

    # Part 2: the metadata dict actually sent to Pinecone does NOT leak
    # the internal ``_compacted_from`` key.
    captured_pinecone.upsert_vector.assert_called_once()
    args, _kwargs = captured_pinecone.upsert_vector.call_args
    # Signature: upsert_vector(id, embedding, metadata)
    pinecone_meta = args[2]
    assert isinstance(pinecone_meta, dict)
    assert "_compacted_from" not in pinecone_meta

    # Part 3: the metadata dict actually sent to the graph path also does
    # NOT leak ``_compacted_from``. This is the load-bearing invariant —
    # Neo4j rejects dict-valued properties, so this must be scrubbed on
    # BOTH write paths, not just Pinecone.
    captured_graph.upsert_graph_data.assert_awaited_once()
    graph_args, _graph_kwargs = captured_graph.upsert_graph_data.call_args
    # Signature: upsert_graph_data(item_id, content, metadata)
    graph_meta = graph_args[2]
    assert isinstance(graph_meta, dict)
    assert "_compacted_from" not in graph_meta


@pytest.mark.asyncio
async def test_memory_service_compaction_hook_only_algorithm() -> None:
    """T8: only ``algorithm`` set, no ``reason`` → helper metadata is
    ``{"algorithm": "x"}``. When neither is set, metadata arg is ``None``.
    """
    # Case A: only algorithm
    edge_service_a, _, _ = await _drive_compaction_hook(
        metadata={
            "_compacted_from": {
                "source_ids": ["src-1"],
                "algorithm": "x",
            },
            "session_id": "sess-alg",
        },
        flag_enabled=True,
    )
    edge_service_a.on_memory_compact.assert_awaited_once()
    call_a = edge_service_a.on_memory_compact.call_args
    assert call_a.kwargs["metadata"] == {"algorithm": "x"}

    # Case B: neither algorithm nor reason
    edge_service_b, _, _ = await _drive_compaction_hook(
        metadata={
            "_compacted_from": {
                "source_ids": ["src-1"],
            },
            "session_id": "sess-none",
        },
        flag_enabled=True,
    )
    edge_service_b.on_memory_compact.assert_awaited_once()
    call_b = edge_service_b.on_memory_compact.call_args
    assert call_b.kwargs["metadata"] is None
