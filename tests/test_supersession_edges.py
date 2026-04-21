"""Unit tests for Phase 5a supersession edge wiring.

Covers:
1. ``on_memory_supersede`` extended signature (reason, run_id, backward compat)
2. Supersession hook integration in ``perform_upsert`` (flag guard, non-fatal)

All tests are fully hermetic — no live Neo4j, Pinecone, or Redis required.
Every dependency is replaced with ``unittest.mock`` fakes.
"""

from __future__ import annotations

import asyncio
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
# 1. on_memory_supersede with reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_with_reason() -> None:
    """When ``reason`` is provided, it appears in edge metadata.

    Also pins the full edge-metadata stamping contract (G3): edge_version=1,
    created_by identifier, weight=1.0, metadata shape.
    """
    svc, create_mock = _make_edge_service()

    await svc.on_memory_supersede("new-1", "old-1", reason="decision changed")

    create_mock.assert_awaited_once()
    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.source_id == "new-1"
    assert edge.target_id == "old-1"
    assert edge.edge_type == "SUPERSEDES"
    assert edge.created_by == "edge_service.on_memory_supersede"
    assert edge.metadata == {"reason": "decision changed"}
    # G3 — explicit stamping assertions
    assert edge.edge_version == 1
    assert edge.weight == 1.0


# ---------------------------------------------------------------------------
# 2. on_memory_supersede with custom run_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_with_custom_run_id() -> None:
    """When ``run_id`` is provided, it overrides the default."""
    svc, create_mock = _make_edge_service()

    await svc.on_memory_supersede("new-2", "old-2", run_id="my-custom-run")

    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.run_id == "my-custom-run"


# ---------------------------------------------------------------------------
# 2b. Empty-string run_id is preserved, not replaced by default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_empty_string_run_id_rejected() -> None:
    """Empty-string run_id reaches MemoryEdge (not silently replaced)
    and is properly rejected by __post_init__ validation."""
    svc, create_mock = _make_edge_service()

    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        await svc.on_memory_supersede("new-r", "old-r", run_id="")


# ---------------------------------------------------------------------------
# 3. Backward compatibility — positional (new_id, old_id) only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_backward_compat() -> None:
    """Calling with just (new_id, old_id) still works — defaults apply.

    The default ``run_id`` is ``"wt-supersede-direct"`` so non-hook callers
    still land under the ``wt-supersede-*`` linker prefix and remain
    addressable by rollback-by-prefix tooling.
    """
    svc, create_mock = _make_edge_service()

    await svc.on_memory_supersede("new-3", "old-3")

    edge: MemoryEdge = create_mock.call_args[0][0]
    assert edge.run_id == "wt-supersede-direct"
    assert edge.metadata is None
    assert edge.edge_version == 1


# ---------------------------------------------------------------------------
# 3b. Self-loop returns None without creating an edge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_self_loop_returns_none() -> None:
    """Self-loop (same-id for both endpoints) returns None without creating an edge."""
    svc, create_mock = _make_edge_service()

    result = await svc.on_memory_supersede("same-id", "same-id")

    assert result is None
    create_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Hook skipped when feature flag is off
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersession_hook_skipped_when_flag_off() -> None:
    """When ASSOC_PROVENANCE_WRITE_ENABLED is False, no edge service
    is constructed and no supersession edges are created."""

    mock_persist = AsyncMock(return_value=True)
    mock_embedding = MagicMock(return_value=[0.1] * 8)
    mock_inject = AsyncMock(side_effect=lambda m: m)

    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", mock_embedding), \
         patch("app.services.memory_service.detect_conflicts", new_callable=AsyncMock) as mock_detect, \
         patch("app.services.memory_service.check_semantic_duplicate", new_callable=AsyncMock, return_value=None):

        mock_settings.ASSOC_PROVENANCE_WRITE_ENABLED = False
        mock_settings.ASSOC_SIMILARITY_WRITE_ENABLED = False
        mock_settings.ASSOC_ENTITY_WRITE_ENABLED = False
        mock_settings.ASSOC_TEMPORAL_WRITE_ENABLED = False
        mock_settings.WRITE_DEDUP_ENABLED = False
        mock_settings.CONFLICT_DETECTION_ENABLED = True
        mock_settings.EMBEDDING_MODEL = "test"

        from app.services.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)
        svc._initialized = True
        svc._similarity_linker = None
        svc._entity_linker = None
        svc._temporal_linker = None
        svc._provenance_edge_service = None
        svc.redis_timeline = None
        svc.embedding_model_name = "test"
        svc.pinecone_client = MagicMock()
        svc.graph_client = MagicMock()
        svc.sequence_service = MagicMock()
        svc.sequence_service.next_seq = AsyncMock(return_value=42)
        svc._persist_memory_item = mock_persist
        svc._inject_chronology = mock_inject
        svc._resolve_memory_id = MagicMock(return_value="test-id")

        # conflict_info would be populated but flag is off
        mock_detect.return_value = [
            {"id": "conflict-1", "similarity": 0.85, "text": "old decision"}
        ]

        result = await svc.perform_upsert(
            content="new decision",
            metadata={"category": "decision"},
        )

        assert result == "test-id"
        # The provenance edge service should never have been constructed
        assert svc._provenance_edge_service is None


# ---------------------------------------------------------------------------
# 5. Hook is non-fatal — exceptions do not break perform_upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersession_hook_nonfatal() -> None:
    """If on_memory_supersede raises, perform_upsert still returns the
    item ID successfully — the hook is fail-open."""

    mock_persist = AsyncMock(return_value=True)
    mock_embedding = MagicMock(return_value=[0.1] * 8)
    mock_inject = AsyncMock(side_effect=lambda m: m)

    boom_edge_service = MagicMock()
    boom_edge_service.on_memory_supersede = AsyncMock(
        side_effect=RuntimeError("Neo4j exploded")
    )

    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", mock_embedding), \
         patch("app.services.memory_service.detect_conflicts", new_callable=AsyncMock) as mock_detect, \
         patch("app.services.memory_service.check_semantic_duplicate", new_callable=AsyncMock, return_value=None):

        mock_settings.ASSOC_PROVENANCE_WRITE_ENABLED = True
        mock_settings.ASSOC_SIMILARITY_WRITE_ENABLED = False
        mock_settings.ASSOC_ENTITY_WRITE_ENABLED = False
        mock_settings.ASSOC_TEMPORAL_WRITE_ENABLED = False
        mock_settings.WRITE_DEDUP_ENABLED = False
        mock_settings.CONFLICT_DETECTION_ENABLED = True
        mock_settings.EMBEDDING_MODEL = "test"

        from app.services.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)
        svc._initialized = True
        svc._similarity_linker = None
        svc._entity_linker = None
        svc._temporal_linker = None
        svc._provenance_edge_service = boom_edge_service
        svc.redis_timeline = None
        svc.embedding_model_name = "test"
        svc.pinecone_client = MagicMock()
        svc.graph_client = MagicMock()
        svc.sequence_service = MagicMock()
        svc.sequence_service.next_seq = AsyncMock(return_value=42)
        svc._persist_memory_item = mock_persist
        svc._inject_chronology = mock_inject
        svc._resolve_memory_id = MagicMock(return_value="test-id")

        mock_detect.return_value = [
            {"id": "conflict-1", "similarity": 0.85, "text": "old decision"}
        ]

        result = await svc.perform_upsert(
            content="new decision",
            metadata={"category": "decision"},
        )

        # Despite the exception, upsert succeeds
        assert result == "test-id"
        # The edge service WAS called (flag was on, conflicts existed)
        boom_edge_service.on_memory_supersede.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. Happy-path integration — flag ON, two conflicts, edges created
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersession_hook_happy_path() -> None:
    """When the flag is ON and detect_conflicts returns two conflicts,
    on_memory_supersede is called exactly twice with correct arguments."""

    mock_persist = AsyncMock(return_value=True)
    mock_embedding = MagicMock(return_value=[0.1] * 8)
    mock_inject = AsyncMock(side_effect=lambda m: m)

    happy_edge_service = MagicMock()
    happy_edge_service.on_memory_supersede = AsyncMock(return_value=None)

    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", mock_embedding), \
         patch("app.services.memory_service.detect_conflicts", new_callable=AsyncMock) as mock_detect, \
         patch("app.services.memory_service.check_semantic_duplicate", new_callable=AsyncMock, return_value=None):

        mock_settings.ASSOC_PROVENANCE_WRITE_ENABLED = True
        mock_settings.ASSOC_SIMILARITY_WRITE_ENABLED = False
        mock_settings.ASSOC_ENTITY_WRITE_ENABLED = False
        mock_settings.ASSOC_TEMPORAL_WRITE_ENABLED = False
        mock_settings.WRITE_DEDUP_ENABLED = False
        mock_settings.CONFLICT_DETECTION_ENABLED = True
        mock_settings.EMBEDDING_MODEL = "test"

        from app.services.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)
        svc._initialized = True
        svc._similarity_linker = None
        svc._entity_linker = None
        svc._temporal_linker = None
        svc._provenance_edge_service = happy_edge_service
        svc.redis_timeline = None
        svc.embedding_model_name = "test"
        svc.pinecone_client = MagicMock()
        svc.graph_client = MagicMock()
        svc.sequence_service = MagicMock()
        svc.sequence_service.next_seq = AsyncMock(return_value=42)
        svc._persist_memory_item = mock_persist
        svc._inject_chronology = mock_inject
        svc._resolve_memory_id = MagicMock(return_value="test-id")

        mock_detect.return_value = [
            {"id": "conflict-A", "similarity": 0.91, "text": "old decision A"},
            {"id": "conflict-B", "similarity": 0.87, "text": "old decision B"},
        ]

        result = await svc.perform_upsert(
            content="new decision",
            metadata={"category": "decision", "session_id": "sess-42"},
        )

        assert result == "test-id"

        # Exactly two calls — one per conflict
        assert happy_edge_service.on_memory_supersede.await_count == 2

        calls = happy_edge_service.on_memory_supersede.call_args_list
        # First conflict
        assert calls[0].kwargs["new_id"] == "test-id"
        assert calls[0].kwargs["old_id"] == "conflict-A"
        assert "0.910" in calls[0].kwargs["reason"]
        assert calls[0].kwargs["run_id"] == "wt-supersede-sess-42"
        # Second conflict
        assert calls[1].kwargs["new_id"] == "test-id"
        assert calls[1].kwargs["old_id"] == "conflict-B"
        assert "0.870" in calls[1].kwargs["reason"]
        assert calls[1].kwargs["run_id"] == "wt-supersede-sess-42"


# ---------------------------------------------------------------------------
# G1 — MERGE idempotency (client-side contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_emits_structurally_identical_edges() -> None:
    """Client-side half of the MERGE idempotency contract.

    Two calls to ``on_memory_supersede`` with identical arguments must
    produce two structurally identical ``MemoryEdge`` objects so that
    Neo4j's ``MERGE`` keys match on the second write. This test does NOT
    prove at-most-one semantics in the graph — that is the server-side
    half, delegated to Neo4j ``MERGE`` inside ``build_merge_edge_cypher``
    and pinned by ``tests/test_edge_cypher.py``. What it pins here is that
    the Python service emits equal edges and does not dedupe client-side.
    """
    svc, create_mock = _make_edge_service()

    kwargs = dict(
        new_id="idem-new",
        old_id="idem-old",
        reason="idempotency check",
        run_id="wt-supersede-idem-run",
    )

    await svc.on_memory_supersede(**kwargs)
    await svc.on_memory_supersede(**kwargs)

    # Both calls reach create_edge — client-side does NOT dedupe.
    assert create_mock.await_count == 2

    edge_a: MemoryEdge = create_mock.call_args_list[0][0][0]
    edge_b: MemoryEdge = create_mock.call_args_list[1][0][0]

    # Structural equality on the fields that matter for MERGE and rollback.
    assert edge_a.source_id == edge_b.source_id == "idem-new"
    assert edge_a.target_id == edge_b.target_id == "idem-old"
    assert edge_a.edge_type == edge_b.edge_type == "SUPERSEDES"
    assert edge_a.run_id == edge_b.run_id == "wt-supersede-idem-run"
    assert edge_a.metadata == edge_b.metadata == {"reason": "idempotency check"}
    assert edge_a.created_by == edge_b.created_by == "edge_service.on_memory_supersede"
    assert edge_a.weight == edge_b.weight == 1.0
    assert edge_a.edge_version == edge_b.edge_version == 1


# ---------------------------------------------------------------------------
# G2 — Rollback round-trip (hermetic / mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_memory_supersede_rollback_round_trip() -> None:
    """End-to-end rollback-by-run_id feasibility, fully hermetic.

    1. Emit N=3 SUPERSEDES edges tagged with a unique test run_id,
       capturing every emitted ``MemoryEdge`` into ``captured_edges``.
    2. Assert all captured edges carry that run_id (prerequisite for
       rollback-by-run_id).
    3. Drive ``scripts.assoc_rollback.assoc_rollback`` against a mock
       neo4j driver in dry-run mode whose count response is sourced
       dynamically from ``len(captured_edges)``. If steps 1-2 had written
       zero edges, the mock would report zero — so the final assertion
       ``report.total == len(captured_edges) == 3`` actually couples
       the write side to the rollback-plan side.

    Live rollback is covered by ``tests/test_assoc_rollback.py`` against a
    real Neo4j; here we only prove the dry-run plan is reachable and is
    coupled to the writes emitted earlier in the same test.
    """
    import uuid

    svc, create_mock = _make_edge_service()

    # Capture every edge actually passed to create_edge so the rollback
    # mock can answer from captured writes, not a hard-coded number.
    captured_edges: list[MemoryEdge] = []

    async def _capture(edge: MemoryEdge) -> None:
        captured_edges.append(edge)
        return None

    create_mock.side_effect = _capture

    run_id = f"test-5a-rollback-{uuid.uuid4()}"

    for i in range(3):
        await svc.on_memory_supersede(
            new_id=f"new-{i}",
            old_id=f"old-{i}",
            reason=f"round-trip {i}",
            run_id=run_id,
        )

    # Step 1/2 — every emitted edge (as actually captured) carries our run_id.
    assert create_mock.await_count == len(captured_edges)
    assert len(captured_edges) == 3
    assert all(e.run_id == run_id for e in captured_edges)
    assert all(e.run_id.startswith("test-5a-rollback-") for e in captured_edges)

    # Step 3 — wire a mock neo4j driver into assoc_rollback.dry_run and
    # assert the planned delete count equals the number of edges we
    # actually emitted (coupling step 1-2 writes to step 3 plan).
    from scripts.assoc_rollback import assoc_rollback

    class _FakeRecord(dict):
        def __getitem__(self, key):
            return dict.__getitem__(self, key)

    # The count the mock returns is derived from captured_edges length,
    # not a hard-coded 3. If the writes above had been zero, the plan
    # would report zero and the final assertion would fail.
    fake_count_records = [
        _FakeRecord({"rel_type": "SUPERSEDES", "c": len(captured_edges)})
    ]

    fake_session = MagicMock()
    fake_session.__enter__ = MagicMock(return_value=fake_session)
    fake_session.__exit__ = MagicMock(return_value=False)
    fake_session.run = MagicMock(return_value=iter(fake_count_records))

    fake_driver = MagicMock()
    fake_driver.session = MagicMock(return_value=fake_session)

    report = assoc_rollback(
        run_id=run_id,
        dry_run=True,
        driver=fake_driver,
    )

    assert report.dry_run is True
    # Real coupling: plan total equals the number of edges emitted above.
    assert report.total == len(captured_edges)
    assert report.total == 3
    assert report.deleted_by_type == {"SUPERSEDES": len(captured_edges)}
    # The driver was queried with our run_id (rollback-by-run_id contract).
    called_args, called_kwargs = fake_session.run.call_args
    assert called_kwargs == {"run_id": run_id} or called_args[1] == {"run_id": run_id}


# ---------------------------------------------------------------------------
# G4 — run_id defaulting behavior after Change 1
# ---------------------------------------------------------------------------


async def _drive_supersession_hook(
    metadata: dict,
) -> MagicMock:
    """Run ``perform_upsert`` with a single captured conflict and return the
    mocked edge service so the caller can inspect the ``run_id`` passed to
    ``on_memory_supersede``.

    Mirrors the fixture pattern used by ``test_supersession_hook_happy_path``.
    """
    mock_persist = AsyncMock(return_value=True)
    mock_embedding = MagicMock(return_value=[0.1] * 8)
    mock_inject = AsyncMock(side_effect=lambda m: m)

    edge_service = MagicMock()
    edge_service.on_memory_supersede = AsyncMock(return_value=None)

    with patch("app.services.memory_service.settings") as mock_settings, \
         patch("app.services.memory_service.get_embedding", mock_embedding), \
         patch("app.services.memory_service.detect_conflicts", new_callable=AsyncMock) as mock_detect, \
         patch("app.services.memory_service.check_semantic_duplicate", new_callable=AsyncMock, return_value=None):

        mock_settings.ASSOC_PROVENANCE_WRITE_ENABLED = True
        mock_settings.ASSOC_SIMILARITY_WRITE_ENABLED = False
        mock_settings.ASSOC_ENTITY_WRITE_ENABLED = False
        mock_settings.ASSOC_TEMPORAL_WRITE_ENABLED = False
        mock_settings.WRITE_DEDUP_ENABLED = False
        mock_settings.CONFLICT_DETECTION_ENABLED = True
        mock_settings.EMBEDDING_MODEL = "test"

        from app.services.memory_service import MemoryService

        svc = MemoryService.__new__(MemoryService)
        svc._initialized = True
        svc._similarity_linker = None
        svc._entity_linker = None
        svc._temporal_linker = None
        svc._provenance_edge_service = edge_service
        svc.redis_timeline = None
        svc.embedding_model_name = "test"
        svc.pinecone_client = MagicMock()
        svc.graph_client = MagicMock()
        svc.sequence_service = MagicMock()
        svc.sequence_service.next_seq = AsyncMock(return_value=42)
        svc._persist_memory_item = mock_persist
        svc._inject_chronology = mock_inject
        svc._resolve_memory_id = MagicMock(return_value="test-id")

        mock_detect.return_value = [
            {"id": "conflict-X", "similarity": 0.9, "text": "old"}
        ]

        await svc.perform_upsert(content="new", metadata=metadata)

    return edge_service


@pytest.mark.asyncio
async def test_supersession_hook_run_id_defaults_no_session() -> None:
    """When metadata lacks ``session_id``, the hook uses the
    ``wt-supersede-no-session`` run_id so the SUPERSEDES edge is still
    rollback-addressable via its linker prefix."""
    edge_service = await _drive_supersession_hook(
        metadata={"category": "decision"}
    )
    edge_service.on_memory_supersede.assert_awaited_once()
    call = edge_service.on_memory_supersede.call_args
    assert call.kwargs["run_id"] == "wt-supersede-no-session"


@pytest.mark.asyncio
async def test_supersession_hook_run_id_defaults_with_session() -> None:
    """When metadata carries ``session_id='sess-42'``, the hook namespaces it
    under the ``wt-supersede-`` linker prefix (not a raw session id)."""
    edge_service = await _drive_supersession_hook(
        metadata={"category": "decision", "session_id": "sess-42"}
    )
    edge_service.on_memory_supersede.assert_awaited_once()
    call = edge_service.on_memory_supersede.call_args
    assert call.kwargs["run_id"] == "wt-supersede-sess-42"
