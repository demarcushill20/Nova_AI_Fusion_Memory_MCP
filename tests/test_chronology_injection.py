"""Tests for Phase 1 chronology injection.

Tests the _inject_chronology logic and bulk sequence allocation in
isolation — no openai/pinecone/neo4j dependencies needed.

The SequenceService is tested thoroughly in test_sequence_service.py.
Here we test the injection contract: every write gets event_seq,
event_time, and memory_type, regardless of what the caller provides.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from app.services.sequence_service import SequenceService


# --- Test the injection logic directly (mirrors MemoryService._inject_chronology) ---


async def _inject_chronology(metadata: dict, seq_service: SequenceService) -> dict:
    """Replica of MemoryService._inject_chronology for isolated testing.

    This is the exact logic added to MemoryService.perform_upsert.
    Testing it here avoids importing the full dependency chain.
    """
    metadata["event_seq"] = await seq_service.next_seq()
    if "event_time" not in metadata or not metadata["event_time"]:
        metadata["event_time"] = datetime.now(timezone.utc).isoformat()
    if "memory_type" not in metadata or not metadata["memory_type"]:
        metadata["memory_type"] = "scratch"
    return metadata


async def _inject_chronology_bulk(items: list, seq_service: SequenceService) -> list:
    """Replica of the bulk injection logic from perform_bulk_upsert."""
    now_iso = datetime.now(timezone.utc).isoformat()
    seq_batch = await seq_service.next_batch(len(items))
    for i, item in enumerate(items):
        meta = item.setdefault("metadata", {})
        meta["event_seq"] = seq_batch[i]
        if "event_time" not in meta or not meta["event_time"]:
            meta["event_time"] = now_iso
        if "memory_type" not in meta or not meta["memory_type"]:
            meta["memory_type"] = "scratch"
    return items


@pytest.fixture
def seq_service(tmp_path):
    return SequenceService(seq_file=str(tmp_path / "event_seq.counter"))


# --- Single Upsert Injection ---


class TestUpsertChronologyInjection:
    """_inject_chronology auto-injects event_seq + event_time."""

    @pytest.mark.asyncio
    async def test_event_seq_injected(self, seq_service):
        result = await _inject_chronology({}, seq_service)
        assert result["event_seq"] == 1

    @pytest.mark.asyncio
    async def test_event_time_injected(self, seq_service):
        result = await _inject_chronology({}, seq_service)
        assert "event_time" in result
        dt = datetime.fromisoformat(result["event_time"])
        assert dt.tzinfo is not None

    @pytest.mark.asyncio
    async def test_memory_type_defaults_to_scratch(self, seq_service):
        result = await _inject_chronology({}, seq_service)
        assert result["memory_type"] == "scratch"

    @pytest.mark.asyncio
    async def test_caller_event_time_preserved(self, seq_service):
        caller_time = "2026-01-15T10:00:00+00:00"
        result = await _inject_chronology({"event_time": caller_time}, seq_service)
        assert result["event_time"] == caller_time

    @pytest.mark.asyncio
    async def test_event_seq_always_system_assigned(self, seq_service):
        """Even if caller sets event_seq, system overwrites it."""
        result = await _inject_chronology({"event_seq": 9999}, seq_service)
        assert result["event_seq"] == 1  # System-assigned, not caller's 9999

    @pytest.mark.asyncio
    async def test_caller_memory_type_preserved(self, seq_service):
        result = await _inject_chronology({"memory_type": "decision"}, seq_service)
        assert result["memory_type"] == "decision"

    @pytest.mark.asyncio
    async def test_sequential_calls_get_increasing_seq(self, seq_service):
        seqs = []
        for _ in range(5):
            r = await _inject_chronology({}, seq_service)
            seqs.append(r["event_seq"])
        assert seqs == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_empty_event_time_gets_overwritten(self, seq_service):
        """Empty string event_time treated as missing."""
        result = await _inject_chronology({"event_time": ""}, seq_service)
        assert result["event_time"] != ""
        dt = datetime.fromisoformat(result["event_time"])
        assert dt.tzinfo is not None

    @pytest.mark.asyncio
    async def test_empty_memory_type_gets_overwritten(self, seq_service):
        """Empty string memory_type treated as missing."""
        result = await _inject_chronology({"memory_type": ""}, seq_service)
        assert result["memory_type"] == "scratch"

    @pytest.mark.asyncio
    async def test_existing_metadata_preserved(self, seq_service):
        """Injection doesn't clobber other metadata fields."""
        result = await _inject_chronology(
            {"category": "research", "tags": ["ai"], "agent": "test"},
            seq_service,
        )
        assert result["category"] == "research"
        assert result["tags"] == ["ai"]
        assert result["agent"] == "test"
        assert "event_seq" in result
        assert "event_time" in result


# --- Bulk Upsert Injection ---


class TestBulkUpsertChronologyInjection:
    """Bulk injection assigns consecutive event_seq values."""

    @pytest.mark.asyncio
    async def test_all_items_get_event_seq(self, seq_service):
        items = [{"content": f"item {i}"} for i in range(3)]
        result = await _inject_chronology_bulk(items, seq_service)
        seqs = [item["metadata"]["event_seq"] for item in result]
        assert seqs == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_all_items_get_event_time(self, seq_service):
        items = [{"content": f"item {i}"} for i in range(3)]
        result = await _inject_chronology_bulk(items, seq_service)
        for item in result:
            assert "event_time" in item["metadata"]
            dt = datetime.fromisoformat(item["metadata"]["event_time"])
            assert dt.tzinfo is not None

    @pytest.mark.asyncio
    async def test_bulk_seqs_are_consecutive(self, seq_service):
        items = [{"content": f"item {i}"} for i in range(5)]
        result = await _inject_chronology_bulk(items, seq_service)
        seqs = [item["metadata"]["event_seq"] for item in result]
        assert seqs == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_bulk_after_single_continues_sequence(self, seq_service):
        """Sequence continues from single upserts into bulk."""
        await _inject_chronology({}, seq_service)  # seq 1
        await _inject_chronology({}, seq_service)  # seq 2
        items = [{"content": f"item {i}"} for i in range(3)]
        result = await _inject_chronology_bulk(items, seq_service)
        seqs = [item["metadata"]["event_seq"] for item in result]
        assert seqs == [3, 4, 5]

    @pytest.mark.asyncio
    async def test_bulk_preserves_caller_event_time(self, seq_service):
        caller_time = "2026-06-01T12:00:00+00:00"
        items = [
            {"content": "with time", "metadata": {"event_time": caller_time}},
            {"content": "without time"},
        ]
        result = await _inject_chronology_bulk(items, seq_service)
        assert result[0]["metadata"]["event_time"] == caller_time
        assert result[1]["metadata"]["event_time"] != caller_time

    @pytest.mark.asyncio
    async def test_bulk_preserves_caller_memory_type(self, seq_service):
        items = [
            {"content": "decision", "metadata": {"memory_type": "decision"}},
            {"content": "default"},
        ]
        result = await _inject_chronology_bulk(items, seq_service)
        assert result[0]["metadata"]["memory_type"] == "decision"
        assert result[1]["metadata"]["memory_type"] == "scratch"

    @pytest.mark.asyncio
    async def test_bulk_items_share_same_event_time(self, seq_service):
        """All items in a bulk batch get the same event_time (snapshot)."""
        items = [{"content": f"item {i}"} for i in range(5)]
        result = await _inject_chronology_bulk(items, seq_service)
        times = {item["metadata"]["event_time"] for item in result}
        assert len(times) == 1  # All same timestamp

    @pytest.mark.asyncio
    async def test_empty_bulk_is_noop(self, seq_service):
        result = await _inject_chronology_bulk([], seq_service)
        assert result == []
        assert seq_service.current_seq() == 0  # No sequences consumed


# --- Contract: System-Enforced, Not Agent-Dependent ---


class TestChronologyContract:
    """The chronology fields are system-enforced guarantees."""

    @pytest.mark.asyncio
    async def test_event_seq_is_integer(self, seq_service):
        result = await _inject_chronology({}, seq_service)
        assert isinstance(result["event_seq"], int)

    @pytest.mark.asyncio
    async def test_event_time_is_iso8601(self, seq_service):
        result = await _inject_chronology({}, seq_service)
        # Must parse as ISO 8601
        dt = datetime.fromisoformat(result["event_time"])
        assert dt.year >= 2026

    @pytest.mark.asyncio
    async def test_event_seq_never_zero(self, seq_service):
        """First event_seq is 1, not 0."""
        result = await _inject_chronology({}, seq_service)
        assert result["event_seq"] >= 1

    @pytest.mark.asyncio
    async def test_event_seq_never_negative(self, seq_service):
        for _ in range(10):
            r = await _inject_chronology({}, seq_service)
            assert r["event_seq"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_injection_no_duplicates(self, seq_service):
        """Concurrent calls produce unique event_seq values."""
        tasks = [_inject_chronology({}, seq_service) for _ in range(50)]
        results = await asyncio.gather(*tasks)
        seqs = [r["event_seq"] for r in results]
        assert len(seqs) == len(set(seqs))
        assert sorted(seqs) == list(range(1, 51))
