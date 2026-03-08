"""Tests for the SequenceService — monotonic event ordering."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from app.services.sequence_service import SequenceService


@pytest.fixture
def seq_file(tmp_path):
    """Provide a temporary counter file path."""
    return str(tmp_path / "event_seq.counter")


@pytest.fixture
def seq_service(seq_file):
    """Provide a fresh SequenceService instance."""
    return SequenceService(seq_file=seq_file)


# --- Basic Functionality ---


class TestNextSeq:
    """next_seq() returns strictly increasing integers starting at 1."""

    @pytest.mark.asyncio
    async def test_first_call_returns_1(self, seq_service):
        assert await seq_service.next_seq() == 1

    @pytest.mark.asyncio
    async def test_sequential_calls_are_monotonic(self, seq_service):
        results = []
        for _ in range(10):
            results.append(await seq_service.next_seq())
        assert results == list(range(1, 11))

    @pytest.mark.asyncio
    async def test_no_duplicates(self, seq_service):
        results = [await seq_service.next_seq() for _ in range(100)]
        assert len(results) == len(set(results))

    @pytest.mark.asyncio
    async def test_strictly_increasing(self, seq_service):
        results = [await seq_service.next_seq() for _ in range(50)]
        for i in range(1, len(results)):
            assert results[i] > results[i - 1]


class TestNextBatch:
    """next_batch() allocates consecutive sequence blocks."""

    @pytest.mark.asyncio
    async def test_batch_returns_correct_count(self, seq_service):
        batch = await seq_service.next_batch(5)
        assert len(batch) == 5

    @pytest.mark.asyncio
    async def test_batch_is_consecutive(self, seq_service):
        batch = await seq_service.next_batch(5)
        assert batch == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_batch_continues_from_previous(self, seq_service):
        await seq_service.next_seq()  # 1
        await seq_service.next_seq()  # 2
        batch = await seq_service.next_batch(3)
        assert batch == [3, 4, 5]

    @pytest.mark.asyncio
    async def test_after_batch_next_seq_continues(self, seq_service):
        await seq_service.next_batch(5)  # 1-5
        assert await seq_service.next_seq() == 6

    @pytest.mark.asyncio
    async def test_batch_zero_returns_empty(self, seq_service):
        batch = await seq_service.next_batch(0)
        assert batch == []

    @pytest.mark.asyncio
    async def test_batch_negative_returns_empty(self, seq_service):
        batch = await seq_service.next_batch(-3)
        assert batch == []

    @pytest.mark.asyncio
    async def test_large_batch(self, seq_service):
        batch = await seq_service.next_batch(500)
        assert len(batch) == 500
        assert batch[0] == 1
        assert batch[-1] == 500


class TestCurrentSeq:
    """current_seq() reads without incrementing."""

    @pytest.mark.asyncio
    async def test_initial_value_is_zero(self, seq_service):
        assert seq_service.current_seq() == 0

    @pytest.mark.asyncio
    async def test_reflects_last_write(self, seq_service):
        await seq_service.next_seq()  # 1
        await seq_service.next_seq()  # 2
        assert seq_service.current_seq() == 2

    @pytest.mark.asyncio
    async def test_does_not_increment(self, seq_service):
        await seq_service.next_seq()  # 1
        _ = seq_service.current_seq()
        _ = seq_service.current_seq()
        assert await seq_service.next_seq() == 2  # not 3 or 4


# --- Persistence ---


class TestPersistence:
    """Counter survives service restart (file-backed)."""

    @pytest.mark.asyncio
    async def test_survives_restart(self, seq_file):
        svc1 = SequenceService(seq_file=seq_file)
        await svc1.next_seq()  # 1
        await svc1.next_seq()  # 2
        await svc1.next_seq()  # 3

        # Simulate restart — new instance, same file
        svc2 = SequenceService(seq_file=seq_file)
        assert svc2.current_seq() == 3
        assert await svc2.next_seq() == 4

    @pytest.mark.asyncio
    async def test_batch_survives_restart(self, seq_file):
        svc1 = SequenceService(seq_file=seq_file)
        await svc1.next_batch(10)  # 1-10

        svc2 = SequenceService(seq_file=seq_file)
        assert svc2.current_seq() == 10
        assert await svc2.next_seq() == 11

    def test_creates_file_if_missing(self, tmp_path):
        path = str(tmp_path / "subdir" / "deep" / "counter.seq")
        svc = SequenceService(seq_file=path)
        assert Path(path).exists()
        assert svc.current_seq() == 0

    @pytest.mark.asyncio
    async def test_handles_corrupted_file(self, seq_file):
        # Write garbage to the counter file
        Path(seq_file).write_text("not_a_number")
        svc = SequenceService(seq_file=seq_file)
        # Should recover gracefully, treating corrupted value as 0
        assert svc.current_seq() == 0
        assert await svc.next_seq() == 1

    @pytest.mark.asyncio
    async def test_handles_empty_file(self, seq_file):
        Path(seq_file).write_text("")
        svc = SequenceService(seq_file=seq_file)
        assert svc.current_seq() == 0
        assert await svc.next_seq() == 1


# --- Concurrency ---


class TestConcurrency:
    """Concurrent async callers get unique, ordered sequences."""

    @pytest.mark.asyncio
    async def test_concurrent_next_seq_no_duplicates(self, seq_service):
        """Many concurrent next_seq() calls produce unique values."""
        tasks = [seq_service.next_seq() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        assert len(results) == len(set(results)), "Duplicate sequence values detected"

    @pytest.mark.asyncio
    async def test_concurrent_next_seq_all_present(self, seq_service):
        """All values 1..N are present after N concurrent calls."""
        tasks = [seq_service.next_seq() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        assert sorted(results) == list(range(1, 51))

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, seq_service):
        """Mix of next_seq and next_batch under concurrency."""
        async def do_singles():
            return [await seq_service.next_seq() for _ in range(5)]

        async def do_batch():
            return await seq_service.next_batch(5)

        r1, r2, r3 = await asyncio.gather(do_singles(), do_batch(), do_singles())
        all_values = r1 + r2 + r3
        assert len(all_values) == 15
        assert len(all_values) == len(set(all_values)), "Duplicate values in mixed ops"
        assert sorted(all_values) == list(range(1, 16))


# --- Atomic Write Safety ---


class TestAtomicWrite:
    """Counter file write uses temp-then-rename for crash safety."""

    @pytest.mark.asyncio
    async def test_no_temp_file_left_behind(self, seq_file):
        svc = SequenceService(seq_file=seq_file)
        await svc.next_seq()
        tmp_path = Path(seq_file).with_suffix(".tmp")
        assert not tmp_path.exists(), "Temporary file was not cleaned up"

    @pytest.mark.asyncio
    async def test_counter_file_contains_integer(self, seq_file):
        svc = SequenceService(seq_file=seq_file)
        await svc.next_seq()
        await svc.next_seq()
        await svc.next_seq()
        content = Path(seq_file).read_text().strip()
        assert content == "3"
