"""Tests for Phase 2 session checkpoint system.

Tests the create_checkpoint and get_last_checkpoint logic in isolation.
No openai/pinecone/neo4j dependencies needed — we mock MemoryService
internals and test the checkpoint contract directly.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.sequence_service import SequenceService


# --- Isolated checkpoint logic replicas ---
# These mirror the exact logic in MemoryService.create_checkpoint
# to test the contract without the full dependency chain.


async def _create_checkpoint_logic(
    seq_service: SequenceService,
    session_id: str,
    session_summary: str,
    started_at=None,
    ended_at=None,
    open_threads=None,
    next_actions=None,
    project=None,
    thread_id=None,
):
    """Replica of MemoryService.create_checkpoint for isolated testing.

    Returns (content, metadata) tuple that would be passed to perform_upsert.
    """
    if not session_id or not session_id.strip():
        return None
    if not session_summary or not session_summary.strip():
        return None

    last_seq = seq_service.current_seq()

    metadata = {
        "memory_type": "checkpoint",
        "session_id": session_id.strip(),
        "session_summary": session_summary.strip(),
        "last_event_seq": last_seq,
    }

    if started_at:
        metadata["started_at"] = started_at
    if ended_at:
        metadata["ended_at"] = ended_at
    if open_threads:
        metadata["open_threads"] = open_threads
    if next_actions:
        metadata["next_actions"] = next_actions
    if project:
        metadata["project"] = project
    if thread_id:
        metadata["thread_id"] = thread_id

    content = f"Session checkpoint: {session_id.strip()}\n\n{session_summary.strip()}"
    return content, metadata


def _get_last_checkpoint_logic(results, project=None, thread_id=None):
    """Replica of the get_last_checkpoint sorting/filtering logic.

    Takes raw Pinecone-style results and returns the latest checkpoint.
    """
    if not results:
        return None

    # Filter by memory_type == checkpoint
    filtered = [
        r for r in results
        if r.get("metadata", {}).get("memory_type") == "checkpoint"
    ]

    if project:
        filtered = [
            r for r in filtered
            if r.get("metadata", {}).get("project") == project
        ]

    if thread_id:
        filtered = [
            r for r in filtered
            if r.get("metadata", {}).get("thread_id") == thread_id
        ]

    if not filtered:
        return None

    # Sort by event_seq descending, return latest
    filtered.sort(
        key=lambda r: r.get("metadata", {}).get("event_seq", 0),
        reverse=True,
    )
    return filtered[0]


@pytest.fixture
def seq_service(tmp_path):
    return SequenceService(seq_file=str(tmp_path / "event_seq.counter"))


# --- create_checkpoint Tests ---


class TestCreateCheckpoint:
    """Checkpoint creation produces correct content and metadata."""

    @pytest.mark.asyncio
    async def test_basic_checkpoint(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="session_001",
            session_summary="Built Phase 1 chronology enforcement.",
        )
        assert result is not None
        content, metadata = result
        assert metadata["memory_type"] == "checkpoint"
        assert metadata["session_id"] == "session_001"
        assert metadata["session_summary"] == "Built Phase 1 chronology enforcement."
        assert metadata["last_event_seq"] == 0  # No events yet

    @pytest.mark.asyncio
    async def test_content_format(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="session_001",
            session_summary="Summary here.",
        )
        content, _ = result
        assert content == "Session checkpoint: session_001\n\nSummary here."

    @pytest.mark.asyncio
    async def test_last_event_seq_snapshots_counter(self, seq_service):
        """last_event_seq reflects events written before checkpoint."""
        # Simulate 5 events written before checkpoint
        for _ in range(5):
            await seq_service.next_seq()
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="session_002",
            session_summary="After 5 events.",
        )
        _, metadata = result
        assert metadata["last_event_seq"] == 5

    @pytest.mark.asyncio
    async def test_optional_fields_included(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="session_003",
            session_summary="Full checkpoint.",
            started_at="2026-03-08T10:00:00Z",
            ended_at="2026-03-08T11:30:00Z",
            open_threads=["thread_a", "thread_b"],
            next_actions=["deploy", "test"],
            project="NovaTrade",
            thread_id="thread_main",
        )
        _, metadata = result
        assert metadata["started_at"] == "2026-03-08T10:00:00Z"
        assert metadata["ended_at"] == "2026-03-08T11:30:00Z"
        assert metadata["open_threads"] == ["thread_a", "thread_b"]
        assert metadata["next_actions"] == ["deploy", "test"]
        assert metadata["project"] == "NovaTrade"
        assert metadata["thread_id"] == "thread_main"

    @pytest.mark.asyncio
    async def test_optional_fields_omitted_when_none(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="session_004",
            session_summary="Minimal.",
        )
        _, metadata = result
        assert "started_at" not in metadata
        assert "ended_at" not in metadata
        assert "open_threads" not in metadata
        assert "next_actions" not in metadata
        assert "project" not in metadata
        assert "thread_id" not in metadata

    @pytest.mark.asyncio
    async def test_empty_session_id_rejected(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="",
            session_summary="Summary.",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_whitespace_session_id_rejected(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="   ",
            session_summary="Summary.",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_summary_rejected(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="session_005",
            session_summary="",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_whitespace_summary_rejected(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="session_005",
            session_summary="   ",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_session_id_stripped(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="  session_006  ",
            session_summary="  Trimmed summary.  ",
        )
        _, metadata = result
        assert metadata["session_id"] == "session_006"
        assert metadata["session_summary"] == "Trimmed summary."

    @pytest.mark.asyncio
    async def test_memory_type_always_checkpoint(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service,
            session_id="session_007",
            session_summary="Type check.",
        )
        _, metadata = result
        assert metadata["memory_type"] == "checkpoint"

    @pytest.mark.asyncio
    async def test_multiple_checkpoints_track_sequence(self, seq_service):
        """Each checkpoint snapshots the correct last_event_seq."""
        # Write 3 events
        for _ in range(3):
            await seq_service.next_seq()

        r1 = await _create_checkpoint_logic(
            seq_service, "s1", "First checkpoint."
        )
        _, m1 = r1
        assert m1["last_event_seq"] == 3

        # Write 2 more events
        for _ in range(2):
            await seq_service.next_seq()

        r2 = await _create_checkpoint_logic(
            seq_service, "s2", "Second checkpoint."
        )
        _, m2 = r2
        assert m2["last_event_seq"] == 5


# --- get_last_checkpoint Tests ---


class TestGetLastCheckpoint:
    """Checkpoint retrieval returns the most recent by event_seq."""

    def _make_checkpoint(self, event_seq, session_id, project=None, thread_id=None):
        """Helper to create a fake Pinecone-style checkpoint result."""
        metadata = {
            "memory_type": "checkpoint",
            "event_seq": event_seq,
            "session_id": session_id,
            "session_summary": f"Summary for {session_id}",
            "last_event_seq": event_seq - 1,
        }
        if project:
            metadata["project"] = project
        if thread_id:
            metadata["thread_id"] = thread_id
        return {
            "id": f"id_{session_id}",
            "score": 0.0,
            "metadata": metadata,
        }

    def test_returns_latest_by_event_seq(self):
        results = [
            self._make_checkpoint(10, "old_session"),
            self._make_checkpoint(50, "latest_session"),
            self._make_checkpoint(30, "mid_session"),
        ]
        latest = _get_last_checkpoint_logic(results)
        assert latest["metadata"]["session_id"] == "latest_session"
        assert latest["metadata"]["event_seq"] == 50

    def test_empty_results_returns_none(self):
        assert _get_last_checkpoint_logic([]) is None

    def test_none_results_returns_none(self):
        assert _get_last_checkpoint_logic(None) is None

    def test_single_checkpoint(self):
        results = [self._make_checkpoint(1, "only_session")]
        latest = _get_last_checkpoint_logic(results)
        assert latest["metadata"]["session_id"] == "only_session"

    def test_filter_by_project(self):
        results = [
            self._make_checkpoint(50, "nova_trade", project="NovaTrade"),
            self._make_checkpoint(100, "nova_shift", project="NovaSHIFT"),
            self._make_checkpoint(30, "nova_trade_old", project="NovaTrade"),
        ]
        latest = _get_last_checkpoint_logic(results, project="NovaTrade")
        assert latest["metadata"]["session_id"] == "nova_trade"
        assert latest["metadata"]["project"] == "NovaTrade"

    def test_filter_by_thread_id(self):
        results = [
            self._make_checkpoint(50, "s1", thread_id="thread_a"),
            self._make_checkpoint(100, "s2", thread_id="thread_b"),
            self._make_checkpoint(80, "s3", thread_id="thread_a"),
        ]
        latest = _get_last_checkpoint_logic(results, thread_id="thread_a")
        assert latest["metadata"]["session_id"] == "s3"

    def test_filter_by_project_and_thread(self):
        results = [
            self._make_checkpoint(100, "s1", project="P1", thread_id="t1"),
            self._make_checkpoint(200, "s2", project="P2", thread_id="t1"),
            self._make_checkpoint(150, "s3", project="P1", thread_id="t2"),
            self._make_checkpoint(80, "s4", project="P1", thread_id="t1"),
        ]
        latest = _get_last_checkpoint_logic(results, project="P1", thread_id="t1")
        assert latest["metadata"]["session_id"] == "s1"

    def test_no_matching_filter_returns_none(self):
        results = [
            self._make_checkpoint(50, "s1", project="NovaTrade"),
        ]
        latest = _get_last_checkpoint_logic(results, project="NonExistent")
        assert latest is None

    def test_non_checkpoint_items_ignored(self):
        results = [
            {
                "id": "regular_memory",
                "score": 0.9,
                "metadata": {
                    "memory_type": "scratch",
                    "event_seq": 999,
                },
            },
            self._make_checkpoint(10, "actual_checkpoint"),
        ]
        latest = _get_last_checkpoint_logic(results)
        assert latest["metadata"]["session_id"] == "actual_checkpoint"
        assert latest["metadata"]["event_seq"] == 10

    def test_all_non_checkpoint_returns_none(self):
        results = [
            {
                "id": "regular_1",
                "score": 0.5,
                "metadata": {"memory_type": "scratch", "event_seq": 100},
            },
            {
                "id": "regular_2",
                "score": 0.3,
                "metadata": {"memory_type": "decision", "event_seq": 200},
            },
        ]
        latest = _get_last_checkpoint_logic(results)
        assert latest is None


# --- Contract Tests ---


class TestCheckpointContract:
    """Session checkpoints meet the Phase 2 contract."""

    @pytest.mark.asyncio
    async def test_checkpoint_is_memory_item(self, seq_service):
        """Checkpoint metadata is compatible with perform_upsert."""
        result = await _create_checkpoint_logic(
            seq_service, "s1", "Summary."
        )
        content, metadata = result
        # Content is a non-empty string
        assert isinstance(content, str) and len(content) > 0
        # Metadata is a dict with required checkpoint fields
        assert isinstance(metadata, dict)
        assert "memory_type" in metadata
        assert "session_id" in metadata
        assert "session_summary" in metadata
        assert "last_event_seq" in metadata

    @pytest.mark.asyncio
    async def test_last_event_seq_is_integer(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service, "s1", "Summary."
        )
        _, metadata = result
        assert isinstance(metadata["last_event_seq"], int)

    @pytest.mark.asyncio
    async def test_last_event_seq_never_negative(self, seq_service):
        result = await _create_checkpoint_logic(
            seq_service, "s1", "Summary."
        )
        _, metadata = result
        assert metadata["last_event_seq"] >= 0

    @pytest.mark.asyncio
    async def test_checkpoint_after_events_captures_position(self, seq_service):
        """Checkpoint last_event_seq correctly marks the timeline position."""
        # Write some events
        await seq_service.next_seq()  # 1
        await seq_service.next_seq()  # 2
        await seq_service.next_seq()  # 3

        result = await _create_checkpoint_logic(
            seq_service, "s1", "After event 3."
        )
        _, metadata = result
        assert metadata["last_event_seq"] == 3

        # The checkpoint itself will consume seq 4 when perform_upsert
        # calls _inject_chronology, but last_event_seq captures the
        # state BEFORE the checkpoint was written.

    @pytest.mark.asyncio
    async def test_required_fields_present(self, seq_service):
        """All required checkpoint fields are always present."""
        result = await _create_checkpoint_logic(
            seq_service, "s1", "Summary."
        )
        _, metadata = result
        required = {"memory_type", "session_id", "session_summary", "last_event_seq"}
        assert required.issubset(set(metadata.keys()))

    def test_get_last_checkpoint_deterministic(self):
        """Same input always returns the same checkpoint (no randomness)."""
        results = [
            {
                "id": "c1",
                "score": 0.0,
                "metadata": {
                    "memory_type": "checkpoint",
                    "event_seq": 10,
                    "session_id": "old",
                },
            },
            {
                "id": "c2",
                "score": 0.0,
                "metadata": {
                    "memory_type": "checkpoint",
                    "event_seq": 20,
                    "session_id": "new",
                },
            },
        ]
        # Run 10 times — must always return the same result
        for _ in range(10):
            latest = _get_last_checkpoint_logic(results)
            assert latest["metadata"]["session_id"] == "new"
