import logging
import os
import re
from enum import Enum, auto
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RoutingMode(Enum):
    """Enum to represent the chosen retrieval strategy."""
    VECTOR = auto()            # Pure semantic similarity
    GRAPH = auto()             # Graph traversal (relationships)
    HYBRID = auto()            # Both vector + graph (default)
    TEMPORAL = auto()          # Pure recency (event_seq DESC, no embeddings)
    TEMPORAL_SEMANTIC = auto() # Temporal window + semantic refinement
    DECISION = auto()          # Decision recall: graph-first + temporal
    PATTERN = auto()           # Pattern recall: high-weight graph
    SESSION = auto()           # Session replay: timeline-first


# --- Intent Pattern Definitions ---
# Each mode maps to a list of regex patterns. A query is scored by how many
# patterns match for each mode. The highest-scoring mode wins; ties are broken
# by declaration order (dict ordering in Python 3.7+).

INTENT_PATTERNS: Dict[RoutingMode, List[str]] = {
    RoutingMode.TEMPORAL: [
        r"\b(last|recent|latest|today|yesterday|this week)\b",
        r"what (did|have) (we|I) (do|done|work)",
        r"\b(just did|earlier today|most recent|recently)\b",
        r"\b(what happened|current state|catch up)\b",
    ],
    RoutingMode.DECISION: [
        r"\b(decid\w*|decision\w*|chose|chosen|rationale)\b",
        r"why did (we|I|you) (choose|pick|go with|decide|select)",
        r"what (was|is) the (plan|approach|strategy|reasoning)",
        r"\b(trade-?offs?|pros and cons|justification)\b",
    ],
    RoutingMode.PATTERN: [
        r"\b(how do we|pattern|workflow|best practice|standard)\b",
        r"\b(usual|typical|convention|how should)\b",
        r"\b(common approach|established|guideline|procedure)\b",
    ],
    RoutingMode.SESSION: [
        r"\b(session|checkpoint|resume|pick up where)\b",
        r"\b(last time|previous session|continue from)\b",
        r"\b(where were we|where did we leave|left off)\b",
    ],
    RoutingMode.GRAPH: [
        r"\b(relat|connect|link|between|depend|upstream|downstream)\b",
        r"\b(associated with|interact with|network of)\b",
        r"\b(compare|contrast)\b",
    ],
    RoutingMode.VECTOR: [
        r"\b(define|explain|summarize|describe|what is|who is)\b",
        r"\b(list all|tell me about|overview of)\b",
    ],
}

# Priority order for breaking ties when two modes have the same match count.
# Lower index = higher priority.
_MODE_PRIORITY: List[RoutingMode] = [
    RoutingMode.DECISION,
    RoutingMode.SESSION,
    RoutingMode.PATTERN,
    RoutingMode.TEMPORAL,
    RoutingMode.GRAPH,
    RoutingMode.VECTOR,
    RoutingMode.HYBRID,
]

# LLM classification prompt template (for future Tier 2 routing)
_LLM_CLASSIFY_PROMPT = """\
Classify the following memory query into exactly one retrieval mode.

Modes:
- VECTOR: pure semantic similarity search
- GRAPH: relationship/connection traversal
- HYBRID: combined vector + graph
- TEMPORAL: pure recency (recent events, no semantic)
- TEMPORAL_SEMANTIC: recency + semantic refinement
- DECISION: recall a past decision and its rationale
- PATTERN: recall an established workflow or best practice
- SESSION: replay or resume from a session/checkpoint

Query: {query}

Reply with ONLY the mode name (e.g., DECISION). No explanation."""


class QueryRouter:
    """
    Classifies user queries to determine the optimal retrieval strategy.

    Phase P9A.4: Uses regex-based intent patterns instead of simple keyword
    substring matching. Falls back to HYBRID for unclassifiable queries.
    Optionally delegates ambiguous queries to an LLM classifier (disabled by
    default).
    """

    def __init__(self):
        self._llm_cache: Dict[str, RoutingMode] = {}
        logger.info("QueryRouter initialized with pattern-based intent classification.")

    def route(self, query_text: str) -> RoutingMode:
        """
        Determines the routing mode based on regex intent patterns.

        Args:
            query_text: The user's input query.

        Returns:
            The determined RoutingMode.
        """
        query_lower = query_text.lower().strip()
        if not query_lower:
            return RoutingMode.HYBRID

        # Score each mode by counting how many of its patterns match
        scores: Dict[RoutingMode, int] = {}
        for mode, patterns in INTENT_PATTERNS.items():
            match_count = sum(1 for p in patterns if re.search(p, query_lower))
            if match_count > 0:
                scores[mode] = match_count

        if not scores:
            logger.debug(f"No intent patterns matched for query: '{query_text[:80]}'. Defaulting to HYBRID.")
            return RoutingMode.HYBRID

        # If temporal + another non-temporal mode matched, use TEMPORAL_SEMANTIC
        if RoutingMode.TEMPORAL in scores and len(scores) > 1:
            non_temporal = {k: v for k, v in scores.items() if k != RoutingMode.TEMPORAL}
            if non_temporal:
                logger.debug(
                    f"Temporal + {list(non_temporal.keys())} matched -> TEMPORAL_SEMANTIC "
                    f"for query: '{query_text[:80]}'"
                )
                return RoutingMode.TEMPORAL_SEMANTIC

        # If session + temporal both matched (but no other), prefer SESSION
        if RoutingMode.SESSION in scores and RoutingMode.TEMPORAL in scores and len(scores) == 2:
            return RoutingMode.SESSION

        # Check for ambiguous case (multiple modes with equal top score)
        max_score = max(scores.values())
        top_modes = [m for m, s in scores.items() if s == max_score]

        if len(top_modes) == 1:
            chosen = top_modes[0]
        else:
            # Tie-break by priority order
            chosen = self._break_tie(top_modes)

            # If LLM routing is enabled, try LLM for ambiguous cases
            llm_result = self.route_with_llm(query_text)
            if llm_result is not None:
                chosen = llm_result

        logger.info(
            f"QUERY_ROUTE query={query_text[:50]!r} mode={chosen.name}"
        )
        return chosen

    def _break_tie(self, tied_modes: List[RoutingMode]) -> RoutingMode:
        """Break a tie between modes using priority ordering."""
        for priority_mode in _MODE_PRIORITY:
            if priority_mode in tied_modes:
                return priority_mode
        # Fallback (should not happen since HYBRID is in priority list)
        return RoutingMode.HYBRID

    def route_with_llm(self, query_text: str) -> Optional[RoutingMode]:
        """
        Optional LLM-based classification for ambiguous queries.

        Tier 2 classifier using Claude Haiku. Only active when
        QUERY_ROUTER_LLM_ENABLED=true. Results are cached in-memory
        for repeated similar queries.

        Args:
            query_text: The user's input query.

        Returns:
            A RoutingMode if LLM classification succeeded, None otherwise
            (indicating the rule-based result should be used).
        """
        # Check if LLM routing is enabled via env or config
        llm_enabled = os.environ.get("QUERY_ROUTER_LLM_ENABLED", "false").lower() in (
            "true", "1", "yes",
        )
        if not llm_enabled:
            return None

        # Check cache
        cache_key = query_text.lower().strip()
        if cache_key in self._llm_cache:
            logger.debug(f"LLM route cache hit for: '{query_text[:50]}'")
            return self._llm_cache[cache_key]

        # TODO: Wire actual LLM call (Claude Haiku) here.
        # For now, return None to fall back to rule-based routing.
        # The prompt template is available at _LLM_CLASSIFY_PROMPT.
        logger.debug(
            "LLM routing enabled but not yet wired. Falling back to rule-based."
        )
        return None


if __name__ == '__main__':
    # Example Usage
    router = QueryRouter()
    queries = [
        "What is the capital of France?",
        "Explain the concept of photosynthesis.",
        "How is John related to Mary in the project?",
        "Show me the connections between AI and machine learning.",
        "Tell me about the twin-charging system and its relationship to performance.",
        "Who is the CEO of OpenAI?",
        "Compare vector databases and graph databases.",
        "Just chat about the weather.",
        "What did we decide about auth?",
        "Why did we choose Pinecone over Weaviate?",
        "How do we handle error recovery?",
        "Resume from last checkpoint",
        "What did we do last session?",
        "What are the recent decisions about the API design?",
    ]

    print("Testing Query Router v2 (Pattern-Based):")
    for q in queries:
        mode = router.route(q)
        print(f"  Query: '{q}' -> Mode: {mode.name}")
