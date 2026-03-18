"""Temporal decay scoring for memory retrieval (Phase P9A.2).

Applies exponential decay to memory scores based on age and category,
so that recent memories are weighted higher than stale ones.
"""

import math
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Category-specific half-lives (in days)
HALF_LIVES: Dict[str, float] = {
    "debug": 7,        # Debug context expires fast
    "context": 14,     # Operational context: 2 weeks
    "research": 30,    # Research findings: 1 month
    "decision": 90,    # Decisions persist longer
    "pattern": 180,    # Patterns are long-lived
    "checkpoint": 14,  # Session checkpoints: 2 weeks
    "default": 30,     # Everything else: 1 month
}

# ln(2) ≈ 0.693147
_LN2 = math.log(2)


def load_half_lives(json_override: Optional[str] = None) -> Dict[str, float]:
    """Return the half-life table, optionally overridden by a JSON string.

    The JSON string should be a dict mapping category names to half-life
    values in days, e.g. '{"debug": 3, "decision": 120}'.  Unknown keys
    are added; known keys are overwritten.
    """
    hl = dict(HALF_LIVES)
    if json_override:
        try:
            overrides = json.loads(json_override)
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    hl[str(k)] = float(v)
                logger.info(f"Temporal decay half-lives overridden: {overrides}")
            else:
                logger.warning(
                    "TEMPORAL_DECAY_HALF_LIVES is not a JSON object; ignoring."
                )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to parse TEMPORAL_DECAY_HALF_LIVES: {exc}")
    return hl


def temporal_decay_score(
    event_time: str,
    category: Optional[str] = None,
    now: Optional[datetime] = None,
    half_lives: Optional[Dict[str, float]] = None,
) -> float:
    """Return an exponential-decay factor in [0.01, 1.0] based on age and category.

    Args:
        event_time: ISO-8601 timestamp string of the memory event.
        category: Memory category (e.g. "debug", "decision"). Falls back to
            "default" if *None* or not found in the half-life table.
        now: Reference "current" time. Defaults to ``datetime.now(UTC)``.
        half_lives: Optional custom half-life table. Defaults to the module-
            level ``HALF_LIVES`` constant.

    Returns:
        A float in [0.01, 1.0].  Fresh memories score ~1.0; memories older
        than several half-lives approach the floor of 0.01.
    """
    now = now or datetime.now(timezone.utc)
    hl = half_lives or HALF_LIVES

    # Parse event_time
    try:
        t = datetime.fromisoformat(event_time)
        # If the parsed datetime is naive, assume UTC
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0.5  # Unknown age → neutral score

    age_days = (now - t).total_seconds() / 86400.0

    # Future timestamps get full score (no penalty)
    if age_days < 0:
        return 1.0

    half_life = hl.get(category or "default", hl.get("default", 30))

    # Guard against non-positive half-lives
    if half_life <= 0:
        half_life = 30.0

    decay = math.exp(-_LN2 * age_days / half_life)

    # Floor at 1% to never fully zero out
    return max(0.01, decay)
