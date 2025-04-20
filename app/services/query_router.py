import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

class RoutingMode(Enum):
    """Enum to represent the chosen retrieval strategy."""
    VECTOR = auto()
    GRAPH = auto()
    HYBRID = auto() # Default: Query both vector and graph stores

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
        # --- DEBUG PRINT ---
        if "connections between ai and machine learning" in query_lower:
            print(f"[DEBUG] Query: '{query_text}'")
            print(f"[DEBUG] is_vector_query: {is_vector_query}")
            print(f"[DEBUG] is_graph_query: {is_graph_query}")
        # --- END DEBUG PRINT ---

        routing_decision: RoutingMode

        if is_vector_query and is_graph_query:
            # Keywords for both types found, default to hybrid
            routing_decision = RoutingMode.HYBRID
            logger.debug(f"Routing query (Hybrid - keywords for both): '{query_text}'")
        elif is_vector_query:
            # Only vector keywords found
            routing_decision = RoutingMode.VECTOR
            logger.debug(f"Routing query (Vector): '{query_text}'")
        elif is_graph_query:
            # Only graph keywords found
            routing_decision = RoutingMode.GRAPH
            logger.debug(f"Routing query (Graph): '{query_text}'")
        else:
            # No specific keywords found, default to hybrid
            routing_decision = RoutingMode.HYBRID
            logger.debug(f"Routing query (Hybrid - default): '{query_text}'")

        # TODO: Future enhancement - use LLM for more nuanced classification
        # e.g., call OpenAI API with a prompt to classify the query intent.

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