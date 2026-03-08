import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

class RoutingMode(Enum):
    """Enum to represent the chosen retrieval strategy."""
    VECTOR = auto()
    GRAPH = auto()
    HYBRID = auto() # Default: Query both vector and graph stores
    TEMPORAL = auto()          # Recency-first: pure temporal retrieval
    TEMPORAL_SEMANTIC = auto() # Temporal window + semantic refinement

class QueryRouter:
    """
    Classifies user queries to determine the optimal retrieval strategy.
    Starts with a simple rule-based approach.
    """
    def __init__(self):
        # Define keywords that suggest a specific retrieval mode
        # These are examples and should be refined based on testing
        self.vector_keywords = [
            "what is", "who is", "define", "explain", "list", "summarize",
            "when did", "where is", "how to", "steps to", "procedure for"
        ]
        self.graph_keywords = [
            "relationship", "related to", "connection between", "link between",
            "how does", "why does", "compare", "contrast", "network of",
            "associated with", "interact with"
        ]
        self.temporal_keywords = [
            "last", "latest", "most recent", "recently", "just did",
            "before we ended", "previous session", "what did we do",
            "last time", "earlier today", "yesterday", "last session",
            "what happened", "last thing", "final", "end of session",
            "current state", "where were we", "pick up where",
            "continuation", "resume", "catch up",
        ]
        logger.info("QueryRouter initialized with rule-based keyword matching.")

    def route(self, query_text: str) -> RoutingMode:
        """
        Determines the routing mode based on keywords in the query.

        Args:
            query_text: The user's input query.

        Returns:
            The determined RoutingMode (VECTOR, GRAPH, or HYBRID).
        """
        query_lower = query_text.lower()
        is_vector_query = any(keyword in query_lower for keyword in self.vector_keywords)
        is_graph_query = any(keyword in query_lower for keyword in self.graph_keywords)
        is_temporal_query = any(keyword in query_lower for keyword in self.temporal_keywords)

        routing_decision: RoutingMode

        # Temporal detection takes priority — recency intent is distinct from
        # semantic or graph intent and should never be downgraded to HYBRID.
        if is_temporal_query and (is_vector_query or is_graph_query):
            # Temporal + semantic/graph keywords → two-stage retrieval
            routing_decision = RoutingMode.TEMPORAL_SEMANTIC
            logger.debug(f"Routing query (Temporal-Semantic): '{query_text}'")
        elif is_temporal_query:
            # Pure temporal query — no semantic similarity needed
            routing_decision = RoutingMode.TEMPORAL
            logger.debug(f"Routing query (Temporal): '{query_text}'")
        elif is_vector_query and is_graph_query:
            routing_decision = RoutingMode.HYBRID
            logger.debug(f"Routing query (Hybrid - keywords for both): '{query_text}'")
        elif is_vector_query:
            routing_decision = RoutingMode.VECTOR
            logger.debug(f"Routing query (Vector): '{query_text}'")
        elif is_graph_query:
            routing_decision = RoutingMode.GRAPH
            logger.debug(f"Routing query (Graph): '{query_text}'")
        else:
            routing_decision = RoutingMode.HYBRID
            logger.debug(f"Routing query (Hybrid - default): '{query_text}'")

        return routing_decision

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
        "Just chat about the weather."
    ]

    print("Testing Query Router:")
    for q in queries:
        mode = router.route(q)
        print(f"Query: '{q}' -> Mode: {mode.name}")