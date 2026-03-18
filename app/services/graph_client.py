import logging
import asyncio
from neo4j import AsyncGraphDatabase, exceptions as neo4j_exceptions, AsyncDriver, Result # Import async components
from typing import List, Dict, Any, Optional

# Import settings from the config module
try:
    from ..config import settings
except ImportError:
    print("Error: Could not import settings from app.config. Ensure the file exists and is configured.")
    # Fallback or raise error - Raising for clarity during development
    raise

# Import embedding service (needed if we decide to query by embedding later)
# from .embedding_service import get_embedding, batch_get_embeddings

logger = logging.getLogger(__name__)

# Define the node label consistent with Nova_AI.py usage
NEO4J_NODE_LABEL = "base" # Using lowercase as seen in Nova_AI.py graph_task

class GraphClient:
    """
    Manages interactions with the Neo4j graph database.
    Handles initialization, connection, and CRUD operations for memory nodes.
    Uses the neo4j async driver.
    """
    def __init__(self):
        """Initializes the GraphClient, deferring connection."""
        self.driver: Optional[AsyncDriver] = None
        self._DATABASE = settings.NEO4J_DATABASE
        logger.info("GraphClient initialized. Connection deferred.")

    async def initialize(self) -> bool:
        """
        Initializes the asynchronous connection to the Neo4j database.

        Returns:
            True if initialization is successful, False otherwise.
        """
        if self.driver:
            logger.info("Neo4j driver already initialized.")
            return True

        uri = settings.NEO4J_URI
        user = settings.NEO4J_USER
        password = settings.NEO4J_PASSWORD

        try:
            logger.info(f"Initializing Neo4j async driver for URI: {uri}, User: {user}...")
            auth_kwargs: Dict[str, Any] = {}
            if password:
                masked_password = f"{password[:1]}...{password[-1:]}" if len(password) > 1 else "****"
                logger.info(f"Attempting authenticated driver connection with User: '{user}', Password: '{masked_password}'")
                auth_kwargs["auth"] = (user, password)
            else:
                logger.info("Attempting driver connection without authentication (NEO4J_PASSWORD not set).")

            # Brief delay to allow Neo4j container to finish startup if just launched
            await asyncio.sleep(1)
            logger.info("Attempting Neo4j driver creation...")
            # Use AsyncGraphDatabase for the async driver
            self.driver = AsyncGraphDatabase.driver(uri, **auth_kwargs)
            # Verify connectivity during initialization
            await self.check_connection()
            logger.info(f"Successfully connected to Neo4j database: {self._DATABASE} at {uri}")
            # Ensure constraints/indexes if needed (run once)
            await self._ensure_constraints()
            return True
        except neo4j_exceptions.AuthError as auth_err:
             logger.error(f"❌ Neo4j authentication failed for user '{user}': {auth_err}", exc_info=True)
             self.driver = None
             return False
        except neo4j_exceptions.ServiceUnavailable as su_err:
             logger.error(f"❌ Neo4j service unavailable at {uri}: {su_err}", exc_info=True)
             self.driver = None
             return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j async driver: {e}", exc_info=True)
            self.driver = None
            return False

    async def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            logger.info("Closing Neo4j driver connection...")
            await self.driver.close()
            self.driver = None
            logger.info("Neo4j driver connection closed.")

    async def _ensure_constraints(self):
        """Ensure necessary constraints/indexes exist (e.g., unique ID)."""
        if not self.driver:
            logger.error("Cannot ensure constraints: Neo4j driver not initialized.")
            return
        # Using 'entity_id' as the property name for uniqueness, matching Nova_AI.py's graph ingestion logic
        constraint_query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{NEO4J_NODE_LABEL}) REQUIRE n.entity_id IS UNIQUE"
        # Phase 1: Index on event_seq for temporal ordering queries
        event_seq_index_query = f"CREATE INDEX IF NOT EXISTS FOR (n:{NEO4J_NODE_LABEL}) ON (n.event_seq)"
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                logger.info(f"Ensuring unique constraint on :{NEO4J_NODE_LABEL}(entity_id)...")
                await session.run(constraint_query)
                logger.info(f"Constraint on :{NEO4J_NODE_LABEL}(entity_id) ensured.")
                logger.info(f"Ensuring index on :{NEO4J_NODE_LABEL}(event_seq)...")
                await session.run(event_seq_index_query)
                logger.info(f"Index on :{NEO4J_NODE_LABEL}(event_seq) ensured.")
                # Phase 5: Session node constraint
                session_constraint = "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE"
                session_seq_index = "CREATE INDEX IF NOT EXISTS FOR (s:Session) ON (s.last_event_seq)"
                logger.info("Ensuring unique constraint on :Session(session_id)...")
                await session.run(session_constraint)
                await session.run(session_seq_index)
                logger.info("Session constraints ensured.")
        except Exception as e:
            logger.error(f"❌ Failed to ensure Neo4j constraint: {e}", exc_info=True)
            # Continue execution, but log the error

    async def check_connection(self) -> bool:
        """
        Checks the connection to the Neo4j database by running a simple query.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        if not self.driver:
            logger.warning("Cannot check connection: Neo4j driver not initialized.")
            return False
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result: Result = await session.run("RETURN 1")
                summary = await result.consume() # Consume the result to check for errors
                logger.info(f"Neo4j connection check successful. Query counters: {summary.counters}")
                return True
        except Exception as e:
            logger.error(f"❌ Neo4j connection check failed: {e}", exc_info=True)
            return False

    async def upsert_graph_data(self, node_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Upserts a single node representing a memory item into Neo4j.
        Creates a node with the label ':base' (matching Nova_AI.py) and sets properties.

        Args:
            node_id: The unique ID for the node (should match Pinecone ID, e.g., MD5 hash).
            content: The text content of the memory item.
            metadata: Optional dictionary of metadata to store as properties.

        Returns:
            True if upsert is successful, False otherwise.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot upsert graph data.")
            return False

        # Prepare properties, ensuring metadata is handled safely
        properties = {
            "text": content,
            # Store metadata fields directly if possible, avoid nesting complex objects if not needed
            **(metadata or {}) # Unpack metadata dict into properties
        }
        # Ensure required 'entity_id' property is set for the constraint
        properties['entity_id'] = node_id

        # Cypher query to MERGE the node based on 'entity_id' and set/update properties
        # Using SET n += $properties handles both creation and update cleanly.
        # Adding the specific label :base dynamically.
        cypher = f"""
        MERGE (n:{NEO4J_NODE_LABEL} {{entity_id: $node_id}})
        SET n += $props
        RETURN n.entity_id AS id
        """
        params = {"node_id": node_id, "props": properties}

        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result = await session.run(cypher, params)
                record = await result.single()
                summary = await result.consume() # Important to consume results

                if record and record["id"] == node_id:
                     # Check summary counters to see if properties were set or node was created
                     if summary.counters.properties_set > 0 or summary.counters.nodes_created > 0:
                          logger.info(f"Successfully upserted graph node ID: {node_id} (Nodes created: {summary.counters.nodes_created}, Properties set: {summary.counters.properties_set})")
                          return True
                     else:
                          # Node existed but no properties were updated (unlikely with SET n += $props unless props were identical)
                          logger.warning(f"Graph node ID {node_id} merged but no properties were updated.")
                          return True # Still considered success as the node exists with the ID
                else:
                     # This case should ideally not happen with MERGE if the query ran successfully
                     logger.error(f"Failed to verify upsert for graph node ID: {node_id}. Result record: {record}")
                     return False

        except neo4j_exceptions.ConstraintError as ce:
             logger.error(f"❌ Constraint error during graph upsert for ID {node_id} (potential duplicate?): {ce}", exc_info=True)
             return False # Indicates an issue with the unique ID constraint logic
        except Exception as e:
            logger.error(f"❌ Failed to upsert graph data for ID {node_id}: {e}", exc_info=True)
            return False

    async def query_graph(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries the Neo4j graph for relevant memory nodes.
        Currently retrieves recent ':base' nodes as a simple strategy,
        mirroring the broad retrieval approach potentially used in Nova_AI.py
        before RRF/reranking.

        Args:
            query_text: The user's query text (currently unused in this simple strategy).
            top_k: The maximum number of nodes to return.

        Returns:
            A list of dictionaries representing graph nodes, or an empty list on error.
            Each dict should ideally contain 'id', 'text', 'metadata', and a placeholder 'score'.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot query graph.")
            return []

        # Simple Strategy: Retrieve top_k most recently added/updated nodes?
        # Requires a timestamp property. Let's assume we retrieve nodes and rely on RRF/reranker.
        # We retrieve nodes with the label ':base'.
        # TODO: Implement a more sophisticated query (e.g., full-text search, vector similarity) if needed.
        cypher = f"""
        MATCH (n:{NEO4J_NODE_LABEL})
        RETURN n.entity_id AS id, n.text AS text, n AS node_properties
        LIMIT $limit
        """
        params = {"limit": top_k}

        results = []
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result_cursor: Result = await session.run(cypher, params)
                records = await result_cursor.data() # Fetch all records

                for record in records:
                    node_props = dict(record.get("node_properties", {})) # Get all properties
                    # Construct the result dictionary expected by the merger
                    results.append({
                        "id": record.get("id"),
                        "text": record.get("text"),
                        "source": "graph",
                        "score": 0.0, # Placeholder score - RRF uses rank, not score here
                        "metadata": {k: v for k, v in node_props.items() if k not in ['entity_id', 'text']} # Store other props in metadata
                    })
            logger.info(f"Neo4j graph query returned {len(results)} nodes (limit {top_k}).")
            return results
        except Exception as e:
            logger.error(f"❌ Failed to query graph: {e}", exc_info=True)
            return []

    async def query_recent_events(
        self, n: int = 20, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve N most recent nodes ordered by event_seq DESC.

        Uses Neo4j's native ORDER BY for proper server-side sorting.

        Args:
            n: Number of events to return.
            filters: Optional dict with keys: project, thread_id, memory_type, since_seq.

        Returns:
            List of memory item dicts sorted by event_seq descending.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot query recent events.")
            return []

        where_clauses = []
        params: Dict[str, Any] = {"limit": n}

        if filters:
            if filters.get("project"):
                where_clauses.append("n.project = $project")
                params["project"] = filters["project"]
            if filters.get("thread_id"):
                where_clauses.append("n.thread_id = $thread_id")
                params["thread_id"] = filters["thread_id"]
            if filters.get("memory_type"):
                where_clauses.append("n.memory_type = $memory_type")
                params["memory_type"] = filters["memory_type"]
            if filters.get("since_seq") is not None:
                where_clauses.append("n.event_seq >= $since_seq")
                params["since_seq"] = filters["since_seq"]

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        cypher = f"""
        MATCH (n:{NEO4J_NODE_LABEL})
        {where_str}
        WHERE n.event_seq IS NOT NULL
        RETURN n.entity_id AS id, n.text AS text, n.event_seq AS event_seq,
               n.event_time AS event_time, n.memory_type AS memory_type, n AS node_properties
        ORDER BY n.event_seq DESC
        LIMIT $limit
        """

        # Fix double WHERE if we already have where_clauses
        if where_clauses:
            cypher = f"""
        MATCH (n:{NEO4J_NODE_LABEL})
        WHERE {' AND '.join(where_clauses)} AND n.event_seq IS NOT NULL
        RETURN n.entity_id AS id, n.text AS text, n.event_seq AS event_seq,
               n.event_time AS event_time, n.memory_type AS memory_type, n AS node_properties
        ORDER BY n.event_seq DESC
        LIMIT $limit
        """

        results = []
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result_cursor = await session.run(cypher, params)
                records = await result_cursor.data()

                for record in records:
                    node_props = dict(record.get("node_properties", {}))
                    results.append({
                        "id": record.get("id"),
                        "text": record.get("text"),
                        "source": "graph",
                        "score": 0.0,
                        "metadata": {
                            k: v for k, v in node_props.items()
                            if k not in ["entity_id", "text"]
                        },
                    })
            logger.info(f"Neo4j query_recent_events returned {len(results)} nodes (limit {n}).")
            return results
        except Exception as e:
            logger.error(f"Failed to query recent events from graph: {e}", exc_info=True)
            return []

    # --- Phase 5: Session Graph Model ---

    async def create_session_node(
        self, session_id: str, started_at: Optional[str] = None, **kwargs
    ) -> bool:
        """Create or merge a Session node in Neo4j.

        Args:
            session_id: Unique session identifier.
            started_at: ISO 8601 start time.
            **kwargs: Additional properties (ended_at, last_event_seq, summary, project, thread_id).

        Returns:
            True if successful, False otherwise.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot create session node.")
            return False

        props: Dict[str, Any] = {"session_id": session_id}
        if started_at:
            props["started_at"] = started_at
        for key in ("ended_at", "last_event_seq", "summary", "project", "thread_id"):
            if key in kwargs and kwargs[key] is not None:
                props[key] = kwargs[key]

        cypher = """
        MERGE (s:Session {session_id: $session_id})
        SET s += $props
        RETURN s.session_id AS id
        """
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result = await session.run(cypher, {"session_id": session_id, "props": props})
                record = await result.single()
                await result.consume()
                if record:
                    logger.info(f"Session node created/merged: {session_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to create session node {session_id}: {e}", exc_info=True)
            return False

    async def link_event_to_session(self, event_id: str, session_id: str) -> bool:
        """Create INCLUDES edge from Session to event node.

        Args:
            event_id: The entity_id of the :base event node.
            session_id: The session_id of the :Session node.

        Returns:
            True if edge was created, False otherwise.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot link event to session.")
            return False

        cypher = f"""
        MERGE (s:Session {{session_id: $session_id}})
        WITH s
        MATCH (e:{NEO4J_NODE_LABEL} {{entity_id: $event_id}})
        MERGE (s)-[:INCLUDES]->(e)
        RETURN s.session_id AS sid, e.entity_id AS eid
        """
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result = await session.run(
                    cypher, {"session_id": session_id, "event_id": event_id}
                )
                record = await result.single()
                await result.consume()
                if record:
                    logger.info(f"Linked event {event_id} to session {session_id}")
                    return True
                logger.warning(
                    f"Could not link event {event_id} to session {session_id} "
                    "(one or both nodes missing)."
                )
                return False
        except Exception as e:
            logger.error(
                f"Failed to link event {event_id} to session {session_id}: {e}",
                exc_info=True,
            )
            return False

    async def link_session_follows(
        self, current_session_id: str, previous_session_id: str
    ) -> bool:
        """Create FOLLOWS edge between sessions for ordering chain.

        Args:
            current_session_id: The newer session.
            previous_session_id: The older session it follows.

        Returns:
            True if edge was created, False otherwise.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot link sessions.")
            return False

        cypher = """
        MATCH (curr:Session {session_id: $current_id})
        MATCH (prev:Session {session_id: $previous_id})
        MERGE (curr)-[:FOLLOWS]->(prev)
        RETURN curr.session_id AS cid, prev.session_id AS pid
        """
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result = await session.run(
                    cypher,
                    {"current_id": current_session_id, "previous_id": previous_session_id},
                )
                record = await result.single()
                await result.consume()
                if record:
                    logger.info(
                        f"Session chain: {current_session_id} -[:FOLLOWS]-> {previous_session_id}"
                    )
                    return True
                logger.warning(
                    f"Could not link sessions {current_session_id} -> {previous_session_id} "
                    "(one or both missing)."
                )
                return False
        except Exception as e:
            logger.error(
                f"Failed to link sessions {current_session_id} -> {previous_session_id}: {e}",
                exc_info=True,
            )
            return False

    async def get_session_events(
        self, session_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all events in a session via INCLUDES edges, ordered by event_seq.

        Args:
            session_id: The session to query.
            limit: Max events to return.

        Returns:
            List of event dicts sorted by event_seq descending.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot get session events.")
            return []

        cypher = f"""
        MATCH (s:Session {{session_id: $session_id}})-[:INCLUDES]->(e:{NEO4J_NODE_LABEL})
        RETURN e.entity_id AS id, e.text AS text, e.event_seq AS event_seq,
               e.event_time AS event_time, e.memory_type AS memory_type
        ORDER BY e.event_seq DESC
        LIMIT $limit
        """
        results = []
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result_cursor = await session.run(
                    cypher, {"session_id": session_id, "limit": limit}
                )
                records = await result_cursor.data()
                for record in records:
                    results.append({
                        "id": record.get("id"),
                        "text": record.get("text"),
                        "event_seq": record.get("event_seq"),
                        "event_time": record.get("event_time"),
                        "memory_type": record.get("memory_type"),
                        "source": "graph",
                    })
            logger.info(
                f"get_session_events({session_id}): returned {len(results)} events."
            )
            return results
        except Exception as e:
            logger.error(
                f"Failed to get session events for {session_id}: {e}", exc_info=True
            )
            return []

    async def get_latest_session(
        self, project: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent Session node by last_event_seq.

        Args:
            project: Optional project filter.

        Returns:
            Dict with session properties, or None if no sessions exist.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot get latest session.")
            return None

        where = "WHERE s.project = $project" if project else ""
        params: Dict[str, Any] = {}
        if project:
            params["project"] = project

        cypher = f"""
        MATCH (s:Session)
        {where}
        RETURN s.session_id AS session_id, s.started_at AS started_at,
               s.ended_at AS ended_at, s.last_event_seq AS last_event_seq,
               s.summary AS summary, s.project AS project,
               s.thread_id AS thread_id
        ORDER BY s.last_event_seq DESC
        LIMIT 1
        """
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result_cursor = await session.run(cypher, params)
                record = await result_cursor.single()
                await result_cursor.consume()
                if record:
                    result = dict(record)
                    logger.info(f"Latest session: {result.get('session_id')}")
                    return result
                logger.info("No sessions found.")
                return None
        except Exception as e:
            logger.error(f"Failed to get latest session: {e}", exc_info=True)
            return None

    # --- Phase P9A.6: Graph-Augmented Retrieval ---

    async def query_graph_multihop(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Multi-hop graph traversal starting from entity nodes.

        Finds nodes matching entity names, then traverses relationships
        up to max_hops deep to find related memories.

        Args:
            entity_names: Entity names to use as starting points.
            max_hops: Maximum traversal depth (1-3, clamped).
            top_k: Maximum results to return.

        Returns:
            List of memory dicts with graph_score based on hop distance.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot query graph multihop.")
            return []

        if not entity_names:
            return []

        max_hops = max(1, min(max_hops, 3))

        # Build WHERE clause to match entity names in text or entity_id
        # Use CONTAINS for flexible matching
        name_conditions = []
        params: Dict[str, Any] = {"limit": top_k}
        for i, name in enumerate(entity_names[:5]):  # Cap at 5 entities
            param_key = f"name_{i}"
            name_conditions.append(
                f"(toLower(start.text) CONTAINS ${param_key} OR "
                f"toLower(start.entity_id) CONTAINS ${param_key})"
            )
            params[param_key] = name.lower()

        if not name_conditions:
            return []

        where_clause = " OR ".join(name_conditions)

        # Multi-hop traversal with distance scoring
        cypher = f"""
        MATCH (start:{NEO4J_NODE_LABEL})
        WHERE {where_clause}
        WITH start LIMIT 50
        MATCH path = (start)-[*1..{max_hops}]-(related:{NEO4J_NODE_LABEL})
        WHERE related <> start
        WITH DISTINCT related, min(length(path)) AS distance
        RETURN related.entity_id AS id,
               related.text AS text,
               related AS node_properties,
               distance,
               1.0 / (1.0 + distance) AS graph_score
        ORDER BY graph_score DESC, related.event_seq DESC
        LIMIT $limit
        """

        results = []
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result_cursor = await session.run(cypher, params)
                records = await result_cursor.data()

                for record in records:
                    node_props = dict(record.get("node_properties", {}))
                    results.append({
                        "id": record.get("id"),
                        "text": record.get("text"),
                        "source": "graph_multihop",
                        "score": record.get("graph_score", 0.0),
                        "graph_score": record.get("graph_score", 0.0),
                        "hop_distance": record.get("distance", 0),
                        "metadata": {
                            k: v for k, v in node_props.items()
                            if k not in ["entity_id", "text"]
                        },
                    })

            logger.info(
                f"Graph multihop query returned {len(results)} results "
                f"(entities={entity_names[:3]}, hops={max_hops})."
            )
            return results
        except Exception as e:
            logger.error(f"Failed to query graph multihop: {e}", exc_info=True)
            return []

    async def get_session_chain(
        self, session_id: str, depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Traverse session FOLLOWS chain to get session history.

        Args:
            session_id: Starting session ID.
            depth: Maximum number of sessions to traverse back.

        Returns:
            List of session dicts ordered from newest to oldest.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot get session chain.")
            return []

        depth = max(1, min(depth, 20))

        cypher = """
        MATCH (start:Session {session_id: $session_id})
        OPTIONAL MATCH path = (start)-[:FOLLOWS*1..DEPTH_PLACEHOLDER]->(ancestor:Session)
        WITH start, ancestor, length(path) AS chain_distance
        ORDER BY chain_distance ASC
        WITH collect({
            session_id: ancestor.session_id,
            started_at: ancestor.started_at,
            ended_at: ancestor.ended_at,
            last_event_seq: ancestor.last_event_seq,
            summary: ancestor.summary,
            project: ancestor.project,
            chain_distance: chain_distance
        }) AS ancestors
        RETURN {
            session_id: start.session_id,
            started_at: start.started_at,
            ended_at: start.ended_at,
            last_event_seq: start.last_event_seq,
            summary: start.summary,
            project: start.project,
            chain_distance: 0
        } AS current, ancestors
        """.replace("DEPTH_PLACEHOLDER", str(depth))

        results = []
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result_cursor = await session.run(
                    cypher, {"session_id": session_id}
                )
                record = await result_cursor.single()
                await result_cursor.consume()

                if record:
                    current = record.get("current")
                    if current:
                        results.append(dict(current))
                    ancestors = record.get("ancestors", [])
                    for a in ancestors:
                        if a and a.get("session_id"):
                            results.append(dict(a))

            logger.info(
                f"Session chain for {session_id}: {len(results)} sessions."
            )
            return results
        except Exception as e:
            logger.error(
                f"Failed to get session chain for {session_id}: {e}",
                exc_info=True,
            )
            return []

    async def find_related_decisions(
        self, query_text: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find decision-type nodes and their related memories.

        Useful for decision recall: finds decisions then traverses to
        related context, research, and outcomes.

        Args:
            query_text: The decision query text (used for text matching).
            top_k: Maximum results to return.

        Returns:
            List of decision + related memory dicts.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized.")
            return []

        # Find decisions and their 1-hop neighbors, filtering by query_text
        query_fragment = query_text[:50] if query_text else ""

        # Try query-filtered search first, fall back to recent decisions
        cypher = f"""
        MATCH (d:{NEO4J_NODE_LABEL})
        WHERE d.memory_type IN ['decision', 'plan', 'strategy']
          AND d.text IS NOT NULL
          AND toLower(d.text) CONTAINS toLower($query_fragment)
        WITH d
        ORDER BY d.event_seq DESC
        LIMIT 50
        OPTIONAL MATCH (d)-[r]-(related:{NEO4J_NODE_LABEL})
        WITH d, collect(DISTINCT {{
            id: related.entity_id,
            text: related.text,
            rel_type: type(r),
            memory_type: related.memory_type
        }})[0..3] AS neighbors
        RETURN d.entity_id AS id,
               d.text AS text,
               d.memory_type AS memory_type,
               d.event_seq AS event_seq,
               d.event_time AS event_time,
               neighbors
        LIMIT $limit
        """

        # Fallback query without text filter (recent decisions)
        cypher_fallback = f"""
        MATCH (d:{NEO4J_NODE_LABEL})
        WHERE d.memory_type IN ['decision', 'plan', 'strategy']
          AND d.text IS NOT NULL
        WITH d
        ORDER BY d.event_seq DESC
        LIMIT 50
        OPTIONAL MATCH (d)-[r]-(related:{NEO4J_NODE_LABEL})
        WITH d, collect(DISTINCT {{
            id: related.entity_id,
            text: related.text,
            rel_type: type(r),
            memory_type: related.memory_type
        }})[0..3] AS neighbors
        RETURN d.entity_id AS id,
               d.text AS text,
               d.memory_type AS memory_type,
               d.event_seq AS event_seq,
               d.event_time AS event_time,
               neighbors
        LIMIT $limit
        """

        results = []
        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result_cursor = await session.run(
                    cypher, {"limit": top_k, "query_fragment": query_fragment}
                )
                records = await result_cursor.data()

                # Fall back to recent decisions if query-filtered search returned nothing
                if not records and query_fragment:
                    logger.debug(
                        "find_related_decisions: no results for query fragment, "
                        "falling back to recent decisions."
                    )
                    result_cursor = await session.run(
                        cypher_fallback, {"limit": top_k}
                    )
                    records = await result_cursor.data()

                for record in records:
                    result = {
                        "id": record.get("id"),
                        "text": record.get("text"),
                        "source": "graph_decision",
                        "score": 0.0,
                        "metadata": {
                            "memory_type": record.get("memory_type"),
                            "event_seq": record.get("event_seq"),
                            "event_time": record.get("event_time"),
                            "related_nodes": record.get("neighbors", []),
                        },
                    }
                    results.append(result)

            logger.info(f"find_related_decisions returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Failed to find related decisions: {e}", exc_info=True)
            return []

    async def delete_graph_data(self, node_id: str) -> bool:
        """
        Deletes a node from the Neo4j graph by its unique ID ('entity_id').

        Args:
            node_id: The unique ID of the node to delete.

        Returns:
            True if deletion is successful or if the node didn't exist, False on error.
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot delete graph data.")
            return False

        # Cypher query to match the node by 'entity_id' and detach delete it
        cypher = f"""
        MATCH (n:{NEO4J_NODE_LABEL} {{entity_id: $node_id}})
        DETACH DELETE n
        """
        params = {"node_id": node_id}

        try:
            async with self.driver.session(database=self._DATABASE) as session:
                result = await session.run(cypher, params)
                summary = await result.consume() # Consume to get summary
                nodes_deleted = summary.counters.nodes_deleted
                logger.info(f"Attempted deletion for graph node ID {node_id}. Nodes deleted: {nodes_deleted}")
                # Consider successful if no error, regardless of whether node existed
                return True
        except Exception as e:
            logger.error(f"❌ Failed to delete graph data for ID {node_id}: {e}", exc_info=True)
            return False

# Example usage (optional, for testing)
async def _test_graph_client():
    print("Testing Graph Client...")
    # Works with or without auth depending on NEO4J_PASSWORD.
    if settings.NEO4J_PASSWORD:
        print("Using authenticated Neo4j connection for graph client test.")
    else:
        print("Using unauthenticated Neo4j connection for graph client test.")

    client = GraphClient()
    initialized = await client.initialize()

    if not initialized:
        print("❌ Neo4j initialization failed. Aborting test.")
        return

    # Test Upsert
    test_id = "test_graph_node_456"
    test_content = "Neo4j is a graph database."
    test_metadata = {"category": "database", "type": "graph"}
    print(f"\nAttempting upsert for ID: {test_id}")
    upsert_ok = await client.upsert_graph_data(test_id, test_content, test_metadata)
    print(f"Upsert successful: {upsert_ok}")
    assert upsert_ok

    # Test Query
    print("\nAttempting query (simple retrieve)...")
    matches = await client.query_graph(query_text="dummy", top_k=5) # Query text unused in simple strategy
    print(f"Query returned {len(matches)} matches.")
    found = False
    if matches:
        print("Sample matches:")
        for match in matches[:2]:
             print(match)
             if match.get("id") == test_id:
                  found = True
                  assert match.get("text") == test_content
                  assert match.get("metadata", {}).get("category") == "database"
    assert found # Check if the inserted node was retrieved

    # Test Delete
    print(f"\nAttempting delete for ID: {test_id}")
    delete_ok = await client.delete_graph_data(test_id)
    print(f"Delete successful: {delete_ok}")
    assert delete_ok

    # Test Query After Delete
    print("\nAttempting query after delete...")
    matches_after_delete = await client.query_graph(query_text="dummy", top_k=5)
    print(f"Query after delete returned {len(matches_after_delete)} matches.")
    found_after_delete = any(match.get("id") == test_id for match in matches_after_delete)
    assert not found_after_delete

    # Test Health Check
    print("\nAttempting health check...")
    health_ok = await client.check_connection()
    print(f"Health check successful: {health_ok}")
    assert health_ok

    # Close connection
    await client.close()
    print("\nGraph Client Test Complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Requires environment variables for Neo4j connection
    asyncio.run(_test_graph_client())
