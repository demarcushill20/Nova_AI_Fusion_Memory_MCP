"""
Redis-backed timeline index for O(log N) chronological queries.

Uses Redis sorted sets with event_seq as score for fast recency lookups.
Pinecone remains the semantic store; Redis is the chronological source of truth.

Two sorted sets per scope:
- nova:timeline:{scope}    — all events, score = event_seq
- nova:checkpoints:{scope} — checkpoint events only, score = last_event_seq
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Key prefixes
TIMELINE_KEY = "nova:timeline"
CHECKPOINT_KEY = "nova:checkpoints"
DEFAULT_SCOPE = "global"


class RedisTimeline:
    """Append-only event timeline in Redis sorted sets.

    Provides O(log N) recency queries and O(1) latest-checkpoint lookups,
    replacing Pinecone dummy-vector queries for chronological retrieval.
    """

    def __init__(self, redis_client):
        """Initialize with an existing async Redis client.

        Args:
            redis_client: An initialized redis.asyncio client instance.
        """
        self._redis = redis_client
        logger.info("RedisTimeline initialized.")

    def _timeline_key(self, scope: str = DEFAULT_SCOPE) -> str:
        return f"{TIMELINE_KEY}:{scope}"

    def _checkpoint_key(self, scope: str = DEFAULT_SCOPE) -> str:
        return f"{CHECKPOINT_KEY}:{scope}"

    async def record_event(
        self,
        event_seq: int,
        memory_id: str,
        metadata_summary: Optional[Dict[str, Any]] = None,
        scope: str = DEFAULT_SCOPE,
    ) -> bool:
        """Add event to the timeline sorted set (score = event_seq).

        Args:
            event_seq: The monotonic sequence number (used as sort score).
            memory_id: The unique memory item ID.
            metadata_summary: Compact metadata for fast retrieval without
                hitting Pinecone (e.g. memory_type, project).
            scope: Timeline scope (default "global", can be project name).

        Returns:
            True if recorded successfully.
        """
        try:
            entry = {"id": memory_id}
            if metadata_summary:
                entry.update(metadata_summary)
            member = json.dumps(entry, sort_keys=True)
            await self._redis.zadd(self._timeline_key(scope), {member: event_seq})
            # Also add to global if scope is project-specific
            if scope != DEFAULT_SCOPE:
                await self._redis.zadd(
                    self._timeline_key(DEFAULT_SCOPE), {member: event_seq}
                )
            return True
        except Exception as e:
            logger.error(f"Failed to record event {memory_id} to timeline: {e}")
            return False

    async def get_recent(
        self, n: int = 20, scope: str = DEFAULT_SCOPE
    ) -> List[Tuple[Dict[str, Any], int]]:
        """Get N most recent events by event_seq (O(log N) + O(K)).

        Args:
            n: Number of events to return.
            scope: Timeline scope to query.

        Returns:
            List of (parsed_entry, event_seq) tuples, most recent first.
        """
        try:
            raw = await self._redis.zrevrange(
                self._timeline_key(scope), 0, n - 1, withscores=True
            )
            results = []
            for member, score in raw:
                try:
                    entry = json.loads(member)
                except (json.JSONDecodeError, TypeError):
                    entry = {"raw": member}
                results.append((entry, int(score)))
            return results
        except Exception as e:
            logger.error(f"Failed to get recent events from timeline: {e}")
            return []

    async def get_since_seq(
        self, since_seq: int, scope: str = DEFAULT_SCOPE
    ) -> List[Tuple[Dict[str, Any], int]]:
        """Get all events with event_seq >= since_seq.

        Args:
            since_seq: Minimum event_seq (inclusive).
            scope: Timeline scope to query.

        Returns:
            List of (parsed_entry, event_seq) tuples, ordered by event_seq ascending.
        """
        try:
            raw = await self._redis.zrangebyscore(
                self._timeline_key(scope),
                min=since_seq,
                max="+inf",
                withscores=True,
            )
            results = []
            for member, score in raw:
                try:
                    entry = json.loads(member)
                except (json.JSONDecodeError, TypeError):
                    entry = {"raw": member}
                results.append((entry, int(score)))
            return results
        except Exception as e:
            logger.error(f"Failed to get events since seq {since_seq}: {e}")
            return []

    async def record_checkpoint(
        self,
        session_id: str,
        last_event_seq: int,
        summary: str,
        scope: str = DEFAULT_SCOPE,
    ) -> bool:
        """Store checkpoint in a dedicated sorted set.

        Args:
            session_id: The session identifier.
            last_event_seq: Sequence number snapshot (used as sort score).
            summary: Brief session summary.
            scope: Timeline scope.

        Returns:
            True if recorded successfully.
        """
        try:
            entry = json.dumps(
                {"session_id": session_id, "summary": summary},
                sort_keys=True,
            )
            await self._redis.zadd(
                self._checkpoint_key(scope), {entry: last_event_seq}
            )
            # Also add to global if scope-specific
            if scope != DEFAULT_SCOPE:
                await self._redis.zadd(
                    self._checkpoint_key(DEFAULT_SCOPE), {entry: last_event_seq}
                )
            return True
        except Exception as e:
            logger.error(f"Failed to record checkpoint {session_id}: {e}")
            return False

    async def get_last_checkpoint(
        self, scope: str = DEFAULT_SCOPE
    ) -> Optional[Tuple[Dict[str, Any], int]]:
        """Get the most recent checkpoint (O(1) via ZREVRANGE 0 0).

        Returns:
            (parsed_entry, last_event_seq) tuple or None.
        """
        try:
            raw = await self._redis.zrevrange(
                self._checkpoint_key(scope), 0, 0, withscores=True
            )
            if not raw:
                return None
            member, score = raw[0]
            try:
                entry = json.loads(member)
            except (json.JSONDecodeError, TypeError):
                entry = {"raw": member}
            return (entry, int(score))
        except Exception as e:
            logger.error(f"Failed to get last checkpoint: {e}")
            return None

    async def timeline_size(self, scope: str = DEFAULT_SCOPE) -> int:
        """Return the number of events in the timeline."""
        try:
            return await self._redis.zcard(self._timeline_key(scope))
        except Exception as e:
            logger.error(f"Failed to get timeline size: {e}")
            return 0
