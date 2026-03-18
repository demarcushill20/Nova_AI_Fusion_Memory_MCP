"""Write-time semantic deduplication and conflict detection (P9A.5).

Prevents duplicate memories at write time by checking semantic similarity
against existing entries. For decision-category memories, detects potentially
conflicting decisions that may supersede each other.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Similarity thresholds
DUPLICATE_THRESHOLD = 0.92  # Above this = near-duplicate
CONFLICT_THRESHOLD_LOW = 0.70  # Similar but not duplicate = potential conflict
CONFLICT_THRESHOLD_HIGH = 0.92  # At or above = duplicate, not conflict


async def check_semantic_duplicate(
    pinecone_client: Any,
    content: str,
    embedding: List[float],
    threshold: float = DUPLICATE_THRESHOLD,
    exclude_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check if a semantically similar memory already exists.

    Args:
        pinecone_client: Initialized PineconeClient instance.
        content: The new content to check.
        embedding: Pre-computed embedding for the new content.
        threshold: Cosine similarity threshold for duplicate detection.
        exclude_id: ID to exclude from results (for updates).

    Returns:
        The matching result dict if a near-duplicate is found, else None.
    """
    try:
        results = await _query_similar(pinecone_client, embedding, top_k=3)

        for r in results:
            score = float(r.get("score", 0))
            r_id = r.get("id", "")

            # Skip self-match when updating
            if exclude_id and r_id == exclude_id:
                continue

            if score >= threshold:
                return {
                    "id": r_id,
                    "score": score,
                    "text": r.get("metadata", {}).get("text", ""),
                    "event_time": r.get("metadata", {}).get("event_time"),
                    "memory_type": r.get("metadata", {}).get("memory_type"),
                }
    except Exception as e:
        logger.warning(f"Semantic duplicate check failed: {e}")

    return None


async def detect_conflicts(
    pinecone_client: Any,
    new_content: str,
    new_embedding: List[float],
    category: str = "decision",
) -> List[Dict[str, Any]]:
    """Find potentially conflicting memories for decision-category content.

    A conflict is defined as a memory that is semantically similar (0.70-0.92)
    to the new content — similar enough to be about the same topic, but
    different enough to not be a duplicate. This often indicates a superseded
    or contradicting decision.

    Args:
        pinecone_client: Initialized PineconeClient instance.
        new_content: The new decision content.
        new_embedding: Pre-computed embedding for the new content.
        category: Memory category to filter (default: "decision").

    Returns:
        List of potential conflict dicts with id, text, similarity, event_time.
    """
    try:
        results = await _query_similar(
            pinecone_client,
            new_embedding,
            top_k=5,
            filter_dict={"memory_type": {"$eq": category}},
        )

        conflicts = []
        for r in results:
            score = float(r.get("score", 0))
            # Similar but not duplicate = potential conflict
            if CONFLICT_THRESHOLD_LOW <= score < CONFLICT_THRESHOLD_HIGH:
                conflicts.append({
                    "id": r.get("id", ""),
                    "text": r.get("metadata", {}).get("text", ""),
                    "similarity": score,
                    "event_time": r.get("metadata", {}).get("event_time"),
                })

        return conflicts
    except Exception as e:
        logger.warning(f"Conflict detection failed: {e}")
        return []


def resolve_duplicate_action(
    duplicate: Optional[Dict[str, Any]],
    on_duplicate: str = "auto",
    category: Optional[str] = None,
) -> str:
    """Determine what to do when a duplicate is found.

    Args:
        duplicate: The duplicate result from check_semantic_duplicate, or None.
        on_duplicate: User-specified behavior:
            - "skip": Do not upsert if duplicate found (default for debug/scratch).
            - "update": Overwrite the existing memory (default for decisions).
            - "append": Keep both entries (default for research/context).
            - "conflict": Flag for manual review.
        category: Memory category — used for default behavior selection.

    Returns:
        Action string: "skip", "update", "append", or "conflict".
    """
    if duplicate is None:
        return "append"  # No duplicate → proceed normally

    # If caller specified explicit behavior, use it
    if on_duplicate != "auto":
        return on_duplicate

    # Auto-resolve based on category
    category = (category or "").lower()
    if category in ("debug", "scratch"):
        return "skip"
    if category == "decision":
        return "update"
    if category in ("research", "context", "pattern"):
        return "append"

    return "skip"  # Default: don't create duplicates


async def _query_similar(
    pinecone_client: Any,
    embedding: List[float],
    top_k: int = 3,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Query Pinecone for similar vectors.

    Wraps the synchronous query_vector call in asyncio.to_thread.
    """
    import asyncio

    results = await asyncio.to_thread(
        pinecone_client.query_vector,
        query_vector=embedding,
        top_k=top_k,
        filter=filter_dict,
    )
    return results or []
