"""
Monotonic sequence counter for chronological memory ordering.

Phase 6: Dual-backend — Redis INCR (primary) with file-based fallback.

Redis INCR is truly atomic even under multi-process concurrency and
delivers sub-ms latency. The file-based backend remains as a fallback
when Redis is unavailable or disabled.

Every memory write gets a unique event_seq. This is the canonical
ordering key — not timestamps (clocks drift, multi-agent writes
interleave). event_seq guarantees strict "what happened after what."
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default counter file location (Docker volume-persistent)
DEFAULT_SEQ_FILE = "/data/event_seq.counter"

# Redis key prefix
REDIS_SEQ_KEY = "nova:event_seq:global"


class SequenceService:
    """Atomic monotonic sequence counter with Redis primary and file fallback.

    When Redis is available, uses INCR for true atomic counters that work
    across multiple processes. Falls back to file-based counter with
    asyncio.Lock when Redis is unavailable.
    """

    def __init__(
        self,
        seq_file: str = DEFAULT_SEQ_FILE,
        redis_url: Optional[str] = None,
        redis_enabled: bool = False,
    ):
        self._seq_file = Path(seq_file)
        self._lock = asyncio.Lock()
        self._redis = None
        self._redis_url = redis_url
        self._redis_enabled = redis_enabled
        self._using_redis = False
        self._ensure_file_exists()
        logger.info(
            f"SequenceService initialized (file: {self._seq_file}, "
            f"redis_enabled: {redis_enabled}, current: {self._read_counter()})"
        )

    async def initialize_redis(self) -> bool:
        """Connect to Redis if enabled. Call during service startup.

        Returns:
            True if Redis is connected and ready, False otherwise.
        """
        if not self._redis_enabled or not self._redis_url:
            logger.info("Redis disabled or no URL configured. Using file-based counter.")
            return False

        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
            # Verify connection
            await self._redis.ping()
            self._using_redis = True

            # Sync: if file counter is ahead (e.g. ran without Redis), catch up
            file_val = self._read_counter()
            redis_val = await self._redis.get(REDIS_SEQ_KEY)
            redis_int = int(redis_val) if redis_val else 0
            if file_val > redis_int:
                await self._redis.set(REDIS_SEQ_KEY, file_val)
                logger.info(
                    f"Synced Redis counter from file: {redis_int} → {file_val}"
                )
            elif redis_int > file_val:
                self._write_counter(redis_int)
                logger.info(
                    f"Synced file counter from Redis: {file_val} → {redis_int}"
                )

            logger.info(f"Redis sequence backend connected at {self._redis_url}")
            return True
        except ImportError:
            logger.warning("redis package not installed. Using file-based counter.")
            return False
        except Exception as e:
            logger.warning(f"Redis connection failed, using file fallback: {e}")
            self._redis = None
            self._using_redis = False
            return False

    def _ensure_file_exists(self) -> None:
        """Create the counter file and parent dirs if they don't exist."""
        try:
            self._seq_file.parent.mkdir(parents=True, exist_ok=True)
            if not self._seq_file.exists():
                self._seq_file.write_text("0")
                logger.info(f"Created new sequence counter file at {self._seq_file}")
        except OSError as e:
            logger.error(f"Failed to create sequence file {self._seq_file}: {e}")
            raise

    def _read_counter(self) -> int:
        """Read the current counter value from file."""
        try:
            text = self._seq_file.read_text().strip()
            return int(text) if text else 0
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to read counter file, resetting to 0: {e}")
            return 0

    def _write_counter(self, value: int) -> None:
        """Write the counter value to file atomically.

        Uses write-to-temp-then-rename for crash safety on Linux.
        """
        tmp_path = self._seq_file.with_suffix(".tmp")
        try:
            tmp_path.write_text(str(value))
            tmp_path.rename(self._seq_file)
        except OSError as e:
            logger.error(f"Failed to write counter file: {e}")
            raise

    async def _guard_redis_regression(self, redis_val: int, count: int) -> int:
        """Prevent the sequence counter from regressing below the durable
        on-disk high-water mark after a Redis reset.

        The file counter survives a Redis flush (e.g. a stray ``FLUSHALL`` on
        a shared Redis instance); Redis does not. After a flush, ``INCR`` on
        the now-missing key restarts at 1 — far below the true sequence
        position already persisted in Pinecone. Trusting that value would
        rewind the timeline and mint event_seq values that collide with
        historical records.

        If the lowest value in this allocation (``redis_val - count + 1``) is
        at or below the file watermark, reseed Redis to ``watermark + count``
        and return that, so the sequence stays strictly monotonic across a
        flush. Returns the (possibly corrected) top-of-range value.
        """
        watermark = self._read_counter()
        lowest = redis_val - count + 1
        if lowest <= watermark:
            corrected = watermark + count
            try:
                await self._redis.set(REDIS_SEQ_KEY, corrected)
            except Exception as e:  # pragma: no cover - best-effort reseed
                logger.warning(f"Failed to reseed Redis counter after regression: {e}")
            logger.warning(
                "Redis sequence counter regressed (allocated low=%s ≤ file "
                "watermark %s) — Redis was likely flushed. Reseeded to %s to "
                "preserve monotonicity and avoid event_seq collisions.",
                lowest, watermark, corrected,
            )
            return corrected
        return redis_val

    async def next_seq(self) -> int:
        """Return the next monotonic sequence number.

        Uses Redis INCR when available, file-based counter otherwise.

        Returns:
            The next integer in the sequence (starts at 1).
        """
        if self._using_redis and self._redis:
            try:
                val = await self._redis.incr(REDIS_SEQ_KEY)
                val = await self._guard_redis_regression(val, count=1)
                # Keep file in sync for fallback resilience
                self._write_counter(val)
                return val
            except Exception as e:
                logger.warning(f"Redis INCR failed, falling back to file: {e}")
                self._using_redis = False

        # File-based fallback
        async with self._lock:
            current = self._read_counter()
            next_val = current + 1
            self._write_counter(next_val)
            return next_val

    async def next_batch(self, count: int) -> List[int]:
        """Return ``count`` consecutive sequence numbers.

        Uses Redis pipeline when available for atomic batch allocation.

        Args:
            count: Number of sequence numbers to allocate.

        Returns:
            List of ``count`` integers in ascending order.
        """
        if count <= 0:
            return []

        if self._using_redis and self._redis:
            try:
                pipe = self._redis.pipeline()
                for _ in range(count):
                    pipe.incr(REDIS_SEQ_KEY)
                results = await pipe.execute()
                # Guard against a Redis reset mid-batch: if the lowest value
                # regressed below the durable file watermark, reissue the whole
                # batch above it.
                if results and results[0] <= self._read_counter():
                    top = await self._guard_redis_regression(results[-1], count=count)
                    results = list(range(top - count + 1, top + 1))
                # Keep file in sync
                self._write_counter(results[-1])
                return results
            except Exception as e:
                logger.warning(f"Redis batch failed, falling back to file: {e}")
                self._using_redis = False

        # File-based fallback
        async with self._lock:
            current = self._read_counter()
            batch = list(range(current + 1, current + 1 + count))
            self._write_counter(current + count)
            return batch

    def current_seq(self) -> int:
        """Return the current counter value without incrementing.

        Reads from file (always kept in sync with Redis).
        """
        return self._read_counter()

    async def close(self) -> None:
        """Close Redis connection if open."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._using_redis = False
            logger.info("Redis connection closed.")
