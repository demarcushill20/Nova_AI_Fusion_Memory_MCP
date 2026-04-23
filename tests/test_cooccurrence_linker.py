"""Unit tests for the CooccurrenceLinker (PLAN-0759 Phase 7a).

These tests use mocked EntityLinker and MemoryEdgeService to verify the
co-occurrence linking logic without requiring a live Neo4j instance.

Test inventory
--------------
1.  test_cooccurrence_creates_edges_for_shared_entities — 2+ shared → edge
2.  test_cooccurrence_skips_single_shared_entity — 1 shared → no edge
3.  test_cooccurrence_hub_suppression — mention_count > 50 → skipped
4.  test_cooccurrence_self_loop_guard — memory does not co-occur with itself
5.  test_cooccurrence_weight_calculation — IDF-style weight in (0, 1]
6.  test_cooccurrence_edge_type_and_metadata — CO_OCCURS type, cooccurrence_linker creator
7.  test_cooccurrence_nonfatal_per_edge — one edge failure does not kill others
8.  test_cooccurrence_flag_default_is_false — flag defaults to False
9.  test_cooccurrence_no_entities — no entities → no edges, no crash
10. test_cooccurrence_no_project_skips — None project → skip
11. test_cooccurrence_timeout — wait_for timeout → fail-open, 0 returned
12. test_cooccurrence_semaphore_saturation — all slots full → dropped
13. test_cooccurrence_secondary_hub_suppression — actual degree > HUB_THRESHOLD → skipped
14. test_cooccurrence_max_edges_per_entity_cap — capped at MAX_EDGES_PER_ENTITY
15. test_cooccurrence_weight_floor — minimum weight is 0.01
16. test_cooccurrence_weight_ceiling — maximum weight is 1.0
17. test_cooccurrence_malformed_entity_skipped — non-dict entity → skipped gracefully
18. test_cooccurrence_mention_count_none — mention_count: None → treated as 0
19. test_cooccurrence_entity_fetch_failure — per-entity fetch failure → skip + continue
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# --- Make the ``app`` package importable ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---

from app.services.associations.cooccurrence_linker import CooccurrenceLinker
from app.services.associations.memory_edges import MemoryEdge


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _make_entity(name: str, mention_count: int = 5) -> dict:
    """Build an entity dict matching EntityLinker.get_entities_for_memory output."""
    return {"project": "test-project", "name": name, "mention_count": mention_count}


def _make_memory_ref(memory_id: str) -> dict:
    """Build a memory-ref dict matching EntityLinker.get_memories_for_entity output."""
    return {
        "memory_id": memory_id,
        "created_at": "2026-04-13T00:00:00+00:00",
        "last_seen_at": "2026-04-13T00:00:00+00:00",
    }


def _build_linker(
    entities_for_memory: list[dict] | None = None,
    memories_for_entity: dict[str, list[dict]] | None = None,
    create_edge_return: bool = True,
    create_edge_side_effect: Exception | None = None,
) -> tuple[CooccurrenceLinker, AsyncMock, AsyncMock]:
    """Construct a CooccurrenceLinker with mocked dependencies.

    Returns (linker, mock_edge_service, mock_entity_linker).
    """
    mock_edge_service = AsyncMock()
    if create_edge_side_effect:
        mock_edge_service.create_edge.side_effect = create_edge_side_effect
    else:
        mock_edge_service.create_edge.return_value = create_edge_return

    mock_entity_linker = AsyncMock()
    mock_entity_linker.get_entities_for_memory.return_value = (
        entities_for_memory or []
    )

    if memories_for_entity is not None:
        async def _get_memories(project, entity_name, limit=20):
            return memories_for_entity.get(entity_name, [])
        mock_entity_linker.get_memories_for_entity.side_effect = _get_memories
    else:
        mock_entity_linker.get_memories_for_entity.return_value = []

    linker = CooccurrenceLinker(
        edge_service=mock_edge_service,
        entity_linker=mock_entity_linker,
    )
    return linker, mock_edge_service, mock_entity_linker


async def _drain(linker: CooccurrenceLinker) -> None:
    """Wait for all in-flight tasks to complete."""
    if linker._inflight:
        await asyncio.gather(*linker._inflight, return_exceptions=True)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_cooccurrence_creates_edges_for_shared_entities():
    """Two memories sharing 2+ entities should produce a CO_OCCURS edge."""
    entities = [
        _make_entity("python", 5),
        _make_entity("neo4j", 3),
        _make_entity("redis", 2),
    ]
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "redis": [_make_memory_ref("memory-A")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count == 1
    edge_arg: MemoryEdge = mock_edge_svc.create_edge.call_args[0][0]
    assert edge_arg.edge_type == "CO_OCCURS"
    assert edge_arg.created_by == "cooccurrence_linker"
    assert 0.0 < edge_arg.weight <= 1.0


@pytest.mark.asyncio
async def test_cooccurrence_skips_single_shared_entity():
    """A pair sharing only 1 entity should NOT produce an edge."""
    entities = [
        _make_entity("python", 5),
        _make_entity("neo4j", 3),
    ]
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "neo4j": [_make_memory_ref("memory-A")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count == 0


@pytest.mark.asyncio
async def test_cooccurrence_hub_suppression():
    """Entities with mention_count > HUB_THRESHOLD should be skipped."""
    entities = [
        _make_entity("python", 60),  # hub: mention_count > 50
        _make_entity("neo4j", 3),
    ]
    memories_for_entity = {
        "python": [_make_memory_ref(f"mem-{i}") for i in range(60)],
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
    }

    linker, mock_edge_svc, mock_entity_linker = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count == 0

    entity_calls = mock_entity_linker.get_memories_for_entity.call_args_list
    entity_names_queried = [c.kwargs.get("entity_name") for c in entity_calls]
    assert "python" not in entity_names_queried


@pytest.mark.asyncio
async def test_cooccurrence_self_loop_guard():
    """A memory should not create a CO_OCCURS edge with itself."""
    entities = [
        _make_entity("python", 5),
        _make_entity("neo4j", 3),
    ]
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A")],
        "neo4j": [_make_memory_ref("memory-A")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count == 0


@pytest.mark.asyncio
async def test_cooccurrence_weight_calculation():
    """IDF-style weight should be positive and <= 1.0."""
    entities = [
        _make_entity("python", 10),
        _make_entity("neo4j", 2),
        _make_entity("redis", 3),
    ]
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")] +
                  [_make_memory_ref(f"other-{i}") for i in range(8)],
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "redis": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B"),
                  _make_memory_ref("other-x")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count >= 1
    edge_arg: MemoryEdge = mock_edge_svc.create_edge.call_args[0][0]
    assert 0.0 < edge_arg.weight <= 1.0
    assert edge_arg.weight >= 0.01


@pytest.mark.asyncio
async def test_cooccurrence_edge_type_and_metadata():
    """The edge should have type CO_OCCURS and created_by cooccurrence_linker."""
    entities = [
        _make_entity("python", 5),
        _make_entity("neo4j", 3),
    ]
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count == 1
    edge_arg: MemoryEdge = mock_edge_svc.create_edge.call_args[0][0]
    assert edge_arg.edge_type == "CO_OCCURS"
    assert edge_arg.created_by == "cooccurrence_linker"
    assert edge_arg.run_id.startswith("wt-cooccur-")
    assert edge_arg.metadata is None
    assert {edge_arg.source_id, edge_arg.target_id} == {"memory-A", "memory-B"}


@pytest.mark.asyncio
async def test_cooccurrence_nonfatal_per_edge():
    """One edge creation failure should not prevent other edges from being created."""
    entities = [
        _make_entity("python", 5),
        _make_entity("neo4j", 3),
    ]
    memories_for_entity = {
        "python": [
            _make_memory_ref("memory-A"),
            _make_memory_ref("memory-B"),
            _make_memory_ref("memory-C"),
        ],
        "neo4j": [
            _make_memory_ref("memory-A"),
            _make_memory_ref("memory-B"),
            _make_memory_ref("memory-C"),
        ],
    }

    mock_edge_service = AsyncMock()
    mock_edge_service.create_edge.side_effect = [
        RuntimeError("Neo4j down"),
        True,
    ]

    mock_entity_linker = AsyncMock()
    mock_entity_linker.get_entities_for_memory.return_value = entities

    async def _get_memories(project, entity_name, limit=20):
        return memories_for_entity.get(entity_name, [])
    mock_entity_linker.get_memories_for_entity.side_effect = _get_memories

    linker = CooccurrenceLinker(
        edge_service=mock_edge_service,
        entity_linker=mock_entity_linker,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_service.create_edge.call_count == 2


@pytest.mark.asyncio
async def test_cooccurrence_flag_default_is_true():
    """The ASSOC_COOCCURRENCE_WRITE_ENABLED flag defaults to True.

    Flipped 2026-04-23 Sprint 21 after backfill produced 4,734 CO_OCCURS
    edges cleanly with hub-suppression and per-entity degree cap enforced.
    The write-time linker's hot path is identical to the backfill scan,
    so flipping is a low-risk density win.
    """
    from app.config import Settings

    settings = Settings(
        _env_file=None,
        PINECONE_API_KEY="test-key",
        PINECONE_ENV="test-env",
    )
    assert settings.ASSOC_COOCCURRENCE_WRITE_ENABLED is True


@pytest.mark.asyncio
async def test_cooccurrence_no_entities():
    """A memory with no entities should produce no edges and no crash."""
    linker, mock_edge_svc, mock_entity_linker = _build_linker(
        entities_for_memory=[],
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count == 0
    assert mock_entity_linker.get_entities_for_memory.call_count == 1
    assert mock_entity_linker.get_memories_for_entity.call_count == 0


@pytest.mark.asyncio
async def test_cooccurrence_no_project_skips():
    """If project is None, enqueue_link should return immediately with no work."""
    linker, mock_edge_svc, mock_entity_linker = _build_linker()

    await linker.enqueue_link(memory_id="memory-A", project=None)
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count == 0
    assert mock_entity_linker.get_entities_for_memory.call_count == 0


# --------------------------------------------------------------------------- #
# F4 / F11-F14 new tests                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_cooccurrence_timeout():
    """When _link_one exceeds BACKGROUND_TIMEOUT, fail-open returns 0."""
    entities = [_make_entity("python", 5), _make_entity("neo4j", 3)]
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )
    linker.BACKGROUND_TIMEOUT = 0.001

    # Make entity lookup slow enough to trigger timeout
    original_get = linker._entity_linker.get_entities_for_memory

    async def _slow_get(*a, **kw):
        await asyncio.sleep(0.1)
        return await original_get(*a, **kw)

    linker._entity_linker.get_entities_for_memory = _slow_get

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    # Timeout should have prevented edge creation
    assert mock_edge_svc.create_edge.call_count == 0


@pytest.mark.asyncio
async def test_cooccurrence_semaphore_saturation():
    """When all semaphore slots are full, new requests are dropped."""
    linker, mock_edge_svc, mock_entity_linker = _build_linker()
    # Set semaphore to 1 slot
    linker._semaphore = asyncio.Semaphore(1)
    linker.BACKGROUND_MAX_IN_FLIGHT = 1

    # Block the semaphore
    await linker._semaphore.acquire()

    # Now try to enqueue — should be dropped
    await linker.enqueue_link(memory_id="memory-A", project="test-project")

    # Release semaphore
    linker._semaphore.release()

    # No task should have been created
    assert mock_entity_linker.get_entities_for_memory.call_count == 0


@pytest.mark.asyncio
async def test_cooccurrence_secondary_hub_suppression():
    """When actual degree > HUB_THRESHOLD, entity is skipped even if mention_count is low."""
    # Entity has low mention_count (passes primary check) but actual
    # get_memories_for_entity returns > HUB_THRESHOLD results
    entities = [
        _make_entity("python", 5),
        _make_entity("neo4j", 3),
    ]

    memories_for_entity = {
        # 55 co-mentioning memories — exceeds HUB_THRESHOLD (50)
        "python": [_make_memory_ref(f"mem-{i}") for i in range(55)],
        # Only neo4j shared, which is 1 entity — below MIN_SHARED_ENTITIES
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    # "python" should be hub-skipped via secondary check, leaving only
    # "neo4j" shared — 1 entity below MIN_SHARED_ENTITIES → no edge
    assert mock_edge_svc.create_edge.call_count == 0


@pytest.mark.asyncio
async def test_cooccurrence_max_edges_per_entity_cap():
    """Co-mentioning memories are capped at MAX_EDGES_PER_ENTITY."""
    entities = [
        _make_entity("python", 5),
        _make_entity("neo4j", 3),
    ]

    # 40 memories share both entities with memory-A
    all_mems = [_make_memory_ref("memory-A")] + [
        _make_memory_ref(f"mem-{i}") for i in range(40)
    ]
    memories_for_entity = {
        "python": all_mems,
        "neo4j": all_mems,
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    # Should be capped at MAX_EDGES_PER_ENTITY (30) minus self = 29 max
    assert mock_edge_svc.create_edge.call_count <= 30


@pytest.mark.asyncio
async def test_cooccurrence_weight_floor():
    """Minimum weight should be 0.01 even for entities with equal degree."""
    entities = [
        _make_entity("python", 5),
        _make_entity("neo4j", 5),
    ]
    # Both entities have the same degree → IDF = log(1) = 0 for both
    # But weight floor ensures minimum 0.01
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B"),
                   _make_memory_ref("c"), _make_memory_ref("d"), _make_memory_ref("e")],
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B"),
                  _make_memory_ref("c"), _make_memory_ref("d"), _make_memory_ref("e")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count >= 1
    edge_arg: MemoryEdge = mock_edge_svc.create_edge.call_args[0][0]
    assert edge_arg.weight >= 0.01


@pytest.mark.asyncio
async def test_cooccurrence_weight_ceiling():
    """Maximum weight should be capped at 1.0."""
    entities = [
        _make_entity("python", 2),
        _make_entity("neo4j", 2),
        _make_entity("redis", 2),
        _make_entity("fastapi", 2),
        _make_entity("pydantic", 2),
    ]
    # One memory with many rare shared entities to push IDF sum high
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "redis": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "fastapi": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "pydantic": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    assert mock_edge_svc.create_edge.call_count == 1
    edge_arg: MemoryEdge = mock_edge_svc.create_edge.call_args[0][0]
    assert edge_arg.weight <= 1.0


@pytest.mark.asyncio
async def test_cooccurrence_malformed_entity_skipped():
    """Non-dict entities should be skipped without crashing."""
    entities = [
        "not-a-dict",
        42,
        None,
        _make_entity("python", 5),
        _make_entity("neo4j", 3),
    ]
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    # The valid entities should still produce an edge
    assert mock_edge_svc.create_edge.call_count == 1
    edge_arg: MemoryEdge = mock_edge_svc.create_edge.call_args[0][0]
    assert edge_arg.edge_type == "CO_OCCURS"


@pytest.mark.asyncio
async def test_cooccurrence_mention_count_none():
    """mention_count: None should be treated as 0, not cause TypeError."""
    entities = [
        {"project": "test-project", "name": "python", "mention_count": None},
        {"project": "test-project", "name": "neo4j", "mention_count": None},
    ]
    memories_for_entity = {
        "python": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
        "neo4j": [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")],
    }

    linker, mock_edge_svc, _ = _build_linker(
        entities_for_memory=entities,
        memories_for_entity=memories_for_entity,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    # Should not crash and should produce an edge
    assert mock_edge_svc.create_edge.call_count == 1


@pytest.mark.asyncio
async def test_cooccurrence_entity_fetch_failure():
    """Per-entity fetch failure should skip that entity and continue."""
    entities = [
        _make_entity("python", 5),
        _make_entity("neo4j", 3),
        _make_entity("redis", 2),
    ]

    call_count = 0

    async def _get_memories(project, entity_name, limit=20):
        nonlocal call_count
        call_count += 1
        if entity_name == "python":
            raise RuntimeError("Neo4j connection lost")
        # neo4j and redis both co-mention memory-B
        return [_make_memory_ref("memory-A"), _make_memory_ref("memory-B")]

    mock_edge_service = AsyncMock()
    mock_edge_service.create_edge.return_value = True

    mock_entity_linker = AsyncMock()
    mock_entity_linker.get_entities_for_memory.return_value = entities
    mock_entity_linker.get_memories_for_entity.side_effect = _get_memories

    linker = CooccurrenceLinker(
        edge_service=mock_edge_service,
        entity_linker=mock_entity_linker,
    )

    await linker.enqueue_link(memory_id="memory-A", project="test-project")
    await _drain(linker)

    # python fetch failed, but neo4j + redis both succeeded →
    # memory-B shares 2 entities → edge created
    assert mock_edge_service.create_edge.call_count == 1
    assert call_count == 3  # all 3 entities were attempted
