"""
Monotonic sequence counter for chronological memory ordering.

Provides atomic, strictly increasing event_seq values that survive
server restarts. File-backed by default (zero external dependencies),
upgradeable to Redis in Phase 6.

Every memory write gets a unique event_seq. This is the canonical
ordering key — not timestamps (clocks drift, multi-agent writes
interleave). event_seq guarantees strict "what happened after what."
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Default counter file location (Docker volume-persistent)
DEFAULT_SEQ_FILE = "/data/event_seq.counter"


class SequenceService:
    """Atomic monotonic sequence counter backed by a local file.

    Thread-safe via asyncio.Lock. File is created on first use if it
    doesn't exist. Counter starts at 0 and increments by 1 per call.

    The file contains a single integer — the current counter value.
    """

    def __init__(self, seq_file: str = DEFAULT_SEQ_FILE):
        self._seq_file = Path(seq_file)
        self._lock = asyncio.Lock()
        self._ensure_file_exists()
        logger.info(
            f"SequenceService initialized with counter file: {self._seq_file} "
            f"(current value: {self._read_counter()})"
        )

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

    async def next_seq(self) -> int:
        """Return the next monotonic sequence number.

        Atomic — only one caller gets each value, even under concurrent
        async calls within the same process.

        Returns:
            The next integer in the sequence (starts at 1).
        """
        async with self._lock:
            current = self._read_counter()
            next_val = current + 1
            self._write_counter(next_val)
            return next_val

    async def next_batch(self, count: int) -> List[int]:
        """Return ``count`` consecutive sequence numbers.

        Used by bulk_upsert to assign ordered sequences to all items
        in a single atomic operation.

        Args:
            count: Number of sequence numbers to allocate.

        Returns:
            List of ``count`` integers in ascending order.
        """
        if count <= 0:
            return []
        async with self._lock:
            current = self._read_counter()
            batch = list(range(current + 1, current + 1 + count))
            self._write_counter(current + count)
            return batch

    def current_seq(self) -> int:
        """Return the current counter value without incrementing.

        Useful for recording ``last_event_seq`` in session checkpoints.
        """
        return self._read_counter()
