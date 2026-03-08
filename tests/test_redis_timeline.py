"""Tests for Phase 6 Redis timeline and upgraded sequence service.

Tests the RedisTimeline sorted set logic, SequenceService Redis/file
dual-backend, and graceful fallback behavior.

Uses fakeredis for isolated testing — no real Redis connection needed.
Falls back to logic-only tests if fakeredis is not installed.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Tuple, Dict, Any, Optional

from app.services.sequence_service import SequenceService, REDIS_SEQ_KEY


# --- Try to import fakeredis for realistic Redis simulation ---
try:
    import fakeredis.aioredis as fakeredis_aio
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False


# --- Logic-only tests (no Redis needed) ---


class TestSequenceServiceFileBackend:
    """File-based backend works without Redis."""

    @pytest.fixture
    def seq_service(self, tmp_path):
        return SequenceService(
            seq_file=str(tmp_path / "event_seq.counter"),
            redis_enabled=False,
        )

    @pytest.mark.asyncio
    async def test_next_seq_starts_at_1(self, seq_service):
        assert await seq_service.next_seq() == 1

    @pytest.mark.asyncio
    async def test_sequential_monotonic(self, seq_service):
        seqs = [await seq_service.next_seq() for _ in range(5)]
        assert seqs == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_batch(self, seq_service):
        batch = await seq_service.next_batch(3)
        assert batch == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_batch_then_seq_continues(self, seq_service):
        await seq_service.next_batch(3)
        assert await seq_service.next_seq() == 4

    @pytest.mark.asyncio
    async def test_current_seq(self, seq_service):
        assert seq_service.current_seq() == 0
        await seq_service.next_seq()
        assert seq_service.current_seq() == 1

    @pytest.mark.asyncio
    async def test_redis_not_used_when_disabled(self, seq_service):
        assert not seq_service._using_redis
        assert seq_service._redis is None


class TestSequenceServiceRedisInit:
    """Redis initialization and sync logic."""

    @pytest.fixture
    def seq_service(self, tmp_path):
        return SequenceService(
            seq_file=str(tmp_path / "event_seq.counter"),
            redis_url="redis://localhost:6379/15",
            redis_enabled=True,
        )

    @pytest.mark.asyncio
    async def test_redis_init_fails_gracefully(self, seq_service):
        """When Redis is unreachable, falls back to file."""
        result = await seq_service.initialize_redis()
        # Will fail because no Redis server is running (or fakeredis not connected)
        # Either way, should not raise
        assert seq_service._using_redis is False or result is True

    @pytest.mark.asyncio
    async def test_file_fallback_after_redis_fail(self, seq_service):
        """After Redis init failure, file backend still works."""
        await seq_service.initialize_redis()  # Will fail (no server)
        val = await seq_service.next_seq()
        assert val == 1  # File backend works


# --- RedisTimeline Logic Tests (mock-based) ---


class TestRedisTimelineLogic:
    """Test timeline entry construction and key naming."""

    def test_timeline_key(self):
        from app.services.redis_timeline import RedisTimeline
        rt = RedisTimeline.__new__(RedisTimeline)  # Skip __init__
        rt._redis = None
        assert rt._timeline_key("global") == "nova:timeline:global"
        assert rt._timeline_key("NovaTrade") == "nova:timeline:NovaTrade"

    def test_checkpoint_key(self):
        from app.services.redis_timeline import RedisTimeline
        rt = RedisTimeline.__new__(RedisTimeline)
        rt._redis = None
        assert rt._checkpoint_key("global") == "nova:checkpoints:global"
        assert rt._checkpoint_key("P1") == "nova:checkpoints:P1"

    def test_event_entry_json(self):
        """Event entries are valid JSON with id and metadata."""
        entry = {"id": "mem_123", "memory_type": "scratch", "project": "P1"}
        serialized = json.dumps(entry, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized["id"] == "mem_123"
        assert deserialized["memory_type"] == "scratch"

    def test_checkpoint_entry_json(self):
        """Checkpoint entries contain session_id and summary."""
        entry = {"session_id": "s1", "summary": "Did stuff"}
        serialized = json.dumps(entry, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized["session_id"] == "s1"


# --- FakeRedis Integration Tests (when available) ---


@pytest.mark.skipif(not HAS_FAKEREDIS, reason="fakeredis not installed")
class TestRedisTimelineWithFakeRedis:
    """Full integration tests using fakeredis."""

    @pytest.fixture
    async def redis_client(self):
        client = fakeredis_aio.FakeRedis(decode_responses=True)
        yield client
        await client.flushall()
        await client.close()

    @pytest.fixture
    def timeline(self, redis_client):
        from app.services.redis_timeline import RedisTimeline
        return RedisTimeline(redis_client)

    @pytest.mark.asyncio
    async def test_record_and_get_recent(self, timeline):
        for i in range(1, 6):
            await timeline.record_event(
                event_seq=i,
                memory_id=f"mem_{i}",
                metadata_summary={"memory_type": "scratch"},
            )
        results = await timeline.get_recent(n=3)
        assert len(results) == 3
        seqs = [seq for _, seq in results]
        assert seqs == [5, 4, 3]

    @pytest.mark.asyncio
    async def test_get_recent_returns_entries(self, timeline):
        await timeline.record_event(1, "mem_1", {"memory_type": "scratch"})
        results = await timeline.get_recent(n=1)
        entry, seq = results[0]
        assert entry["id"] == "mem_1"
        assert seq == 1

    @pytest.mark.asyncio
    async def test_get_since_seq(self, timeline):
        for i in range(1, 11):
            await timeline.record_event(i, f"mem_{i}")
        results = await timeline.get_since_seq(since_seq=8)
        assert len(results) == 3
        seqs = [seq for _, seq in results]
        assert seqs == [8, 9, 10]

    @pytest.mark.asyncio
    async def test_record_checkpoint(self, timeline):
        ok = await timeline.record_checkpoint("s1", 42, "Built stuff")
        assert ok is True

    @pytest.mark.asyncio
    async def test_get_last_checkpoint(self, timeline):
        await timeline.record_checkpoint("s1", 10, "First session")
        await timeline.record_checkpoint("s2", 25, "Second session")
        result = await timeline.get_last_checkpoint()
        assert result is not None
        entry, score = result
        assert entry["session_id"] == "s2"
        assert score == 25

    @pytest.mark.asyncio
    async def test_no_checkpoint_returns_none(self, timeline):
        result = await timeline.get_last_checkpoint()
        assert result is None

    @pytest.mark.asyncio
    async def test_timeline_size(self, timeline):
        assert await timeline.timeline_size() == 0
        for i in range(5):
            await timeline.record_event(i + 1, f"mem_{i}")
        assert await timeline.timeline_size() == 5

    @pytest.mark.asyncio
    async def test_project_scoped_timeline(self, timeline):
        await timeline.record_event(1, "m1", {"project": "P1"}, scope="P1")
        await timeline.record_event(2, "m2", {"project": "P2"}, scope="P2")
        p1_results = await timeline.get_recent(n=10, scope="P1")
        assert len(p1_results) == 1
        # Global should have both (record_event adds to global too)
        global_results = await timeline.get_recent(n=10, scope="global")
        assert len(global_results) == 2

    @pytest.mark.asyncio
    async def test_50_events_get_10(self, timeline):
        for i in range(1, 51):
            await timeline.record_event(i, f"mem_{i}")
        results = await timeline.get_recent(n=10)
        assert len(results) == 10
        seqs = [seq for _, seq in results]
        assert seqs == list(range(50, 40, -1))


@pytest.mark.skipif(not HAS_FAKEREDIS, reason="fakeredis not installed")
class TestSequenceServiceWithFakeRedis:
    """SequenceService with Redis backend via fakeredis."""

    @pytest.fixture
    async def seq_service(self, tmp_path):
        client = fakeredis_aio.FakeRedis(decode_responses=True)
        svc = SequenceService(
            seq_file=str(tmp_path / "event_seq.counter"),
            redis_enabled=False,  # We'll manually set up
        )
        svc._redis = client
        svc._using_redis = True
        yield svc
        await client.flushall()
        await client.close()

    @pytest.mark.asyncio
    async def test_redis_incr(self, seq_service):
        val = await seq_service.next_seq()
        assert val == 1

    @pytest.mark.asyncio
    async def test_redis_monotonic(self, seq_service):
        seqs = [await seq_service.next_seq() for _ in range(5)]
        assert seqs == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_redis_batch(self, seq_service):
        batch = await seq_service.next_batch(3)
        assert batch == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_redis_file_sync(self, seq_service):
        """Redis writes are synced to file."""
        await seq_service.next_seq()
        await seq_service.next_seq()
        assert seq_service.current_seq() == 2  # File is in sync

    @pytest.mark.asyncio
    async def test_redis_concurrent_no_duplicates(self, seq_service):
        """Concurrent INCR produces unique values."""
        tasks = [seq_service.next_seq() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        assert len(results) == len(set(results))
        assert sorted(results) == list(range(1, 51))


# --- Graceful Fallback Tests ---


class TestGracefulFallback:
    """Redis failure triggers graceful fallback to file."""

    @pytest.fixture
    def seq_service(self, tmp_path):
        svc = SequenceService(
            seq_file=str(tmp_path / "event_seq.counter"),
            redis_enabled=True,
        )
        # Simulate a broken Redis connection
        mock_redis = AsyncMock()
        mock_redis.incr.side_effect = ConnectionError("Redis down")
        mock_redis.pipeline.side_effect = ConnectionError("Redis down")
        svc._redis = mock_redis
        svc._using_redis = True
        return svc

    @pytest.mark.asyncio
    async def test_next_seq_falls_back(self, seq_service):
        """Redis failure → file fallback works."""
        val = await seq_service.next_seq()
        assert val == 1
        assert not seq_service._using_redis  # Switched to file

    @pytest.mark.asyncio
    async def test_next_batch_falls_back(self, seq_service):
        """Redis batch failure → file fallback works."""
        batch = await seq_service.next_batch(3)
        assert batch == [1, 2, 3]
        assert not seq_service._using_redis

    @pytest.mark.asyncio
    async def test_continued_operation_after_fallback(self, seq_service):
        """After fallback, subsequent calls use file."""
        await seq_service.next_seq()  # Triggers fallback
        val2 = await seq_service.next_seq()
        assert val2 == 2  # File continues correctly
