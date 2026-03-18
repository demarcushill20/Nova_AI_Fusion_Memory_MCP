import logging
import asyncio
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

# Import dependent services and modules
try:
    from ..config import settings
    from .embedding_service import get_embedding, batch_get_embeddings
    from .pinecone_client import PineconeClient
    from .graph_client import GraphClient
    # Import reused Nova modules (assuming they are in the same directory)
    from .query_router import QueryRouter, RoutingMode
    from .hybrid_merger import HybridMerger
    from .reranker import CrossEncoderReranker, PineconeReranker
    from .sequence_service import SequenceService
    from .redis_timeline import RedisTimeline
    from .temporal_scorer import temporal_decay_score, load_half_lives
    from .composite_scorer import composite_score, normalize_semantic_score
    from .mmr_dedup import deduplicate_exact, mmr_rerank
    from .conflict_detector import check_semantic_duplicate, detect_conflicts, resolve_duplicate_action
except ImportError as e:
    print(f"Error importing modules in memory_service.py: {e}. Ensure all service files and Nova modules exist.")
    # Depending on severity, might raise error or proceed with caution
    raise

logger = logging.getLogger(__name__)

class MemoryService:
    """
    Orchestrates memory operations by integrating embedding, vector store (Pinecone),
    graph store (Neo4j), query routing, merging, and reranking components.
    """
    def __init__(self):
        """Initializes the MemoryService and its components."""
        logger.info("Initializing MemoryService...")
        # Initialize clients and Nova modules
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.pinecone_client = PineconeClient()
        self.graph_client = GraphClient()
        self.query_router = QueryRouter()
        self.hybrid_merger = HybridMerger() # Uses default RRF k=60
        self.reranker: Optional[CrossEncoderReranker] = None  # Initialize as None, lazy-loaded on first query
        self.pinecone_reranker: Optional[PineconeReranker] = None  # Fallback hosted reranker
        self.sequence_service = SequenceService(
            seq_file=settings.EVENT_SEQ_FILE,
            redis_url=settings.REDIS_URL if settings.REDIS_ENABLED else None,
            redis_enabled=settings.REDIS_ENABLED,
        )
        self.redis_timeline: Optional[RedisTimeline] = None

        # Flag to track initialization status
        self._initialized = False
        self._reranker_loaded = False
        logger.info("MemoryService components instantiated.")

    async def initialize(self):
        """
        Asynchronously initializes backend clients (Pinecone, Neo4j) and loads
        the reranker model. Should be called before using the service.
        """
        if self._initialized:
            logger.info("MemoryService already initialized.")
            return True

        logger.info("Starting asynchronous initialization of MemoryService...")
        init_tasks = {
            # Run the synchronous pinecone init in a separate thread
            "pinecone": asyncio.to_thread(self.pinecone_client.initialize),
            "graph": self.graph_client.initialize(),
        }

        logger.info("Attempting to gather initialization tasks: Pinecone, Graph...")
        results = await asyncio.gather(*init_tasks.values(), return_exceptions=True)
        logger.info("Initialization tasks gathered (may include exceptions).")

        # Check results
        pinecone_ok = isinstance(results[0], bool) and results[0]
        graph_ok = isinstance(results[1], bool) and results[1]

        if not pinecone_ok:
            logger.error("Pinecone client failed to initialize.")
        if not graph_ok:
            logger.error("Graph client failed to initialize.")

        # Set up rerankers for lazy-load cascade (no eager model loading)
        self._setup_reranker_cascade()

        # Phase 6: Initialize Redis (non-critical — falls back to file)
        redis_ok = await self.sequence_service.initialize_redis()
        if redis_ok and self.sequence_service._redis:
            self.redis_timeline = RedisTimeline(self.sequence_service._redis)
            logger.info("Redis timeline initialized.")
        else:
            self.redis_timeline = None
            logger.info("Redis timeline not available. Using Pinecone for temporal queries.")

        # Service is only initialized if critical components (Pinecone, Graph) succeed.
        if pinecone_ok and graph_ok:
            self._initialized = True
            logger.info(
                f"MemoryService initialization complete. Status - Pinecone: OK, Graph: OK, "
                f"Reranker: lazy-cascade (local+pinecone+rrf), "
                f"Redis: {'OK' if redis_ok else 'Disabled/Fallback'}"
            )
        else:
            self._initialized = False
            logger.error(
                f"MemoryService initialization FAILED. Status - Pinecone: {'OK' if pinecone_ok else 'Failed'}, "
                f"Graph: {'OK' if graph_ok else 'Failed'}, "
                f"Reranker: lazy-cascade, "
                f"Redis: {'OK' if redis_ok else 'Disabled/Fallback'}"
            )

        return self._initialized

    def _setup_reranker_cascade(self) -> None:
        """Initialize reranker objects for the lazy-load cascade.

        Does NOT load the CrossEncoder model yet -- that happens lazily on
        first query via ensure_loaded(). The PineconeReranker is lightweight
        (no model to load) and is always available as a fallback.
        """
        # 1. Local CrossEncoder (lazy-loaded, fastest when available)
        try:
            model_name = settings.RERANKER_MODEL_NAME
            if model_name:
                self.reranker = CrossEncoderReranker(model_name=model_name)
                logger.info(f"CrossEncoderReranker prepared for lazy loading: {model_name}")
            else:
                logger.warning("No RERANKER_MODEL_NAME configured. Local reranker disabled.")
                self.reranker = None
        except Exception as e:
            logger.error(f"Failed to create CrossEncoderReranker instance: {e}", exc_info=True)
            self.reranker = None

        # 2. Pinecone Inference API reranker (remote fallback)
        try:
            self.pinecone_reranker = PineconeReranker()
            logger.info("PineconeReranker prepared as fallback.")
        except Exception as e:
            logger.error(f"Failed to create PineconeReranker instance: {e}", exc_info=True)
            self.pinecone_reranker = None

    async def close(self):
        """Closes connections (e.g., Neo4j driver)."""
        logger.info("Closing MemoryService resources...")
        await self.graph_client.close()
        # Pinecone client might not need explicit closing depending on version
        logger.info("MemoryService resources closed.")

    async def perform_query(self, query_text: str, top_k_vector: int = 50, top_k_final: int = 15) -> List[Dict[str, Any]]:
        """
        Performs a memory query using routing-aware retrieval.

        Routing modes (Phase 4 + P9A.4):
        - TEMPORAL: pure temporal retrieval (event_seq DESC, no embeddings)
        - TEMPORAL_SEMANTIC: temporal window first, then semantic refinement
        - DECISION: graph-first + temporal weighting (decision recall)
        - PATTERN: high-weight graph query (pattern/workflow recall)
        - SESSION: timeline-first via session events
        - VECTOR/GRAPH/HYBRID: existing semantic pipeline (unchanged)

        Args:
            query_text: The user's query.
            top_k_vector: Number of initial results to fetch from Pinecone.
            top_k_final: Number of final results to return after reranking.

        Returns:
            A list of relevant memory items, sorted by relevance or recency.
        """
        if not self._initialized:
            logger.error("MemoryService not initialized. Cannot perform query.")
            return []
        if not query_text:
            logger.warning("Received empty query text.")
            return []

        logger.info(f"Performing query for: '{query_text[:100]}...'")

        # 1. Query Routing — determines retrieval strategy
        routing_mode = self.query_router.route(query_text)
        logger.info(f"QUERY_ROUTE query={query_text[:50]!r} mode={routing_mode.name}")

        # 2. Route to appropriate retrieval path
        if routing_mode == RoutingMode.TEMPORAL:
            # Pure temporal retrieval — no embeddings needed
            logger.info("Using TEMPORAL path: pure recency retrieval.")
            results = await self.get_recent_events(n=top_k_final)
            self._tag_routing_mode(results, routing_mode)
            return results

        elif routing_mode == RoutingMode.TEMPORAL_SEMANTIC:
            # Stage 1: Get temporal window
            logger.info("Using TEMPORAL_SEMANTIC path: temporal window + semantic refinement.")
            recent = await self.get_recent_events(n=top_k_vector)
            if not recent:
                logger.info("No recent events found, falling back to full semantic pipeline.")
                results = await self._semantic_query(query_text, top_k_vector, top_k_final)
                self._tag_routing_mode(results, routing_mode)
                return results

            # Stage 2: Semantic search constrained to the temporal window
            min_seq = min(
                r.get("metadata", {}).get("event_seq", 0) for r in recent
            )
            logger.info(f"Temporal window: event_seq >= {min_seq}. Running semantic within window.")
            results = await self._semantic_query(
                query_text, top_k_vector, top_k_final,
                pinecone_filter={"event_seq": {"$gte": min_seq}},
            )
            self._tag_routing_mode(results, routing_mode)
            return results

        elif routing_mode == RoutingMode.DECISION:
            # Decision recall: graph-first query filtered to decision types,
            # then semantic refinement with temporal weighting
            logger.info("Using DECISION path: graph-first + temporal weighting.")
            results = await self._decision_query(query_text, top_k_vector, top_k_final)
            self._tag_routing_mode(results, routing_mode)
            return results

        elif routing_mode == RoutingMode.PATTERN:
            # Pattern recall: high-weight graph query for workflow/pattern nodes
            logger.info("Using PATTERN path: graph-weighted semantic query.")
            results = await self._pattern_query(query_text, top_k_vector, top_k_final)
            self._tag_routing_mode(results, routing_mode)
            return results

        elif routing_mode == RoutingMode.SESSION:
            # Session replay: timeline-first via session events
            logger.info("Using SESSION path: timeline-first session replay.")
            results = await self._session_query(query_text, top_k_final)
            self._tag_routing_mode(results, routing_mode)
            return results

        else:
            # Existing behavior: full semantic pipeline (VECTOR, GRAPH, HYBRID)
            results = await self._semantic_query(query_text, top_k_vector, top_k_final)
            self._tag_routing_mode(results, routing_mode)
            return results

    @staticmethod
    def _tag_routing_mode(results: List[Dict[str, Any]], mode: RoutingMode) -> None:
        """Attach routing_mode to each result's metadata for observability."""
        for r in results:
            r.setdefault("metadata", {})["routing_mode"] = mode.name

    async def _decision_query(
        self,
        query_text: str,
        top_k_vector: int = 50,
        top_k_final: int = 15,
    ) -> List[Dict[str, Any]]:
        """Decision recall: graph-first with category filter + temporal weighting.

        Strategy:
        1. Run semantic query with Pinecone filter for decision-type memories
        2. Fall back to standard semantic query if no decision-typed results
        """
        try:
            # Try filtering for decision-category memories first
            results = await self._semantic_query(
                query_text, top_k_vector, top_k_final,
                pinecone_filter={"memory_type": {"$in": ["decision", "plan", "strategy"]}},
            )
            if results:
                return results
        except Exception as exc:
            logger.warning(
                "Decision-filtered query failed, falling back to standard semantic: %s",
                exc,
            )

        # Fallback: standard semantic pipeline
        logger.info("DECISION fallback -> standard semantic query (no decision-typed results found).")
        return await self._semantic_query(query_text, top_k_vector, top_k_final)

    async def _pattern_query(
        self,
        query_text: str,
        top_k_vector: int = 50,
        top_k_final: int = 15,
    ) -> List[Dict[str, Any]]:
        """Pattern recall: semantic query filtered to pattern/workflow memories.

        Strategy:
        1. Run semantic query with Pinecone filter for pattern-type memories
        2. Fall back to standard semantic query if no pattern-typed results
        """
        try:
            results = await self._semantic_query(
                query_text, top_k_vector, top_k_final,
                pinecone_filter={"memory_type": {"$in": ["pattern", "workflow", "best_practice", "context"]}},
            )
            if results:
                return results
        except Exception as exc:
            logger.warning(
                "Pattern-filtered query failed, falling back to standard semantic: %s",
                exc,
            )

        # Fallback: standard semantic pipeline
        logger.info("PATTERN fallback -> standard semantic query (no pattern-typed results found).")
        return await self._semantic_query(query_text, top_k_vector, top_k_final)

    async def _session_query(
        self,
        query_text: str,
        top_k_final: int = 15,
    ) -> List[Dict[str, Any]]:
        """Session replay: timeline-first via latest session events.

        Strategy:
        1. Find the latest session via graph_client.get_latest_session()
        2. Get session events via graph_client.get_session_events()
        3. Fall back to get_recent_events() if graph session data unavailable
        """
        try:
            # Try to get the latest session from graph
            latest_session = await self.graph_client.get_latest_session()
            if latest_session:
                session_id = latest_session.get("session_id")
                if session_id:
                    events = await self.graph_client.get_session_events(
                        session_id=session_id, limit=top_k_final
                    )
                    if events:
                        # Convert graph events to standard result format
                        results = []
                        for evt in events:
                            results.append({
                                "id": evt.get("id"),
                                "score": 0.0,
                                "metadata": {
                                    "text": evt.get("text", ""),
                                    "event_seq": evt.get("event_seq"),
                                    "event_time": evt.get("event_time"),
                                    "memory_type": evt.get("memory_type"),
                                    "session_id": session_id,
                                },
                                "source": "graph_session",
                            })
                        logger.info(
                            f"SESSION query: returned {len(results)} events from session {session_id}."
                        )
                        return results
        except Exception as exc:
            logger.warning(
                "Session graph query failed, falling back to recent events: %s",
                exc,
            )

        # Fallback: use recent events timeline
        logger.info("SESSION fallback -> get_recent_events (no graph session data).")
        return await self.get_recent_events(n=top_k_final)

    async def _semantic_query(
        self,
        query_text: str,
        top_k_vector: int = 50,
        top_k_final: int = 15,
        pinecone_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Full semantic retrieval pipeline: embed → vector+graph → RRF → rerank.

        Extracted from the original perform_query for reuse by both the
        standard and temporal-semantic paths.

        Args:
            query_text: The user's query.
            top_k_vector: Number of initial results to fetch.
            top_k_final: Number of final results to return.
            pinecone_filter: Optional Pinecone metadata filter to constrain the search.
        """
        # Get Query Embedding
        try:
            query_embedding = await asyncio.to_thread(get_embedding, query_text, self.embedding_model_name)
            if not any(query_embedding):
                 logger.error("Failed to get valid query embedding. Aborting query.")
                 return []
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}", exc_info=True)
            return []

        # Parallel Retrieval (Vector + Graph)
        # Request embedding values from Pinecone when MMR is enabled (avoids extra API calls)
        mmr_enabled = getattr(settings, "MMR_ENABLED", True)
        vector_results = []
        graph_results = []
        try:
            logger.debug(f"Initiating parallel retrieval (Vector k={top_k_vector}, Graph k={top_k_vector})...")
            results = await asyncio.gather(
                asyncio.to_thread(
                    self.pinecone_client.query_vector,
                    query_embedding,
                    top_k=top_k_vector,
                    filter=pinecone_filter,
                    include_values=mmr_enabled,
                ),
                self.graph_client.query_graph(query_text, top_k=top_k_vector),
                return_exceptions=True
            )

            if isinstance(results[0], list):
                vector_results = results[0]
                logger.info(f"Vector retrieval returned {len(vector_results)} results.")
            elif isinstance(results[0], Exception):
                logger.error(f"Vector retrieval failed: {results[0]}", exc_info=results[0])

            if isinstance(results[1], list):
                graph_results = results[1]
                logger.info(f"Graph retrieval returned {len(graph_results)} results.")
            elif isinstance(results[1], Exception):
                logger.error(f"Graph retrieval failed: {results[1]}", exc_info=results[1])

        except Exception as e:
            logger.error(f"Error during parallel retrieval: {e}", exc_info=True)

        # Hybrid Merging (RRF)
        if not vector_results and not graph_results:
             logger.warning("No results from either vector or graph store.")
             return []

        try:
            logger.debug(f"Merging {len(vector_results)} vector and {len(graph_results)} graph results...")
            fused_results = await asyncio.to_thread(
                self.hybrid_merger.merge_results, vector_results, graph_results
            )
            logger.info(f"Hybrid merging complete. {len(fused_results)} unique results after RRF.")
        except Exception as e:
            logger.error(f"Error during hybrid merging: {e}", exc_info=True)
            return []

        # Reranking — cascading strategy: local CrossEncoder -> Pinecone API -> RRF-only
        final_results, reranker_name = await self._cascading_rerank(
            query_text, fused_results, top_k_final
        )

        # Tag results with which reranker was used
        for r in final_results:
            r.setdefault("metadata", {})["reranker_used"] = reranker_name

        # Temporal decay + composite scoring (Phase P9A.2)
        final_results = self._apply_temporal_scoring(final_results, reranker_name)

        # MMR deduplication (Phase P9A.3) — remove near-duplicates, balance diversity
        final_results = self._apply_mmr_dedup(final_results, vector_results, top_k_final)

        return final_results

    def _apply_temporal_scoring(
        self,
        results: List[Dict[str, Any]],
        reranker_name: str,
    ) -> List[Dict[str, Any]]:
        """Apply temporal decay and composite scoring to reranked results.

        Gated by ``settings.TEMPORAL_DECAY_ENABLED``.  On any error the
        original list is returned unchanged so the query pipeline is never
        blocked by scoring failures.
        """
        try:
            if not getattr(settings, "TEMPORAL_DECAY_ENABLED", True):
                return results
            if not results:
                return results

            # Load (potentially overridden) half-lives once per call
            hl_json = getattr(settings, "TEMPORAL_DECAY_HALF_LIVES", None)
            half_lives = load_half_lives(hl_json)

            # Build weight dict from config
            temporal_weight = getattr(settings, "TEMPORAL_WEIGHT", 0.30)
            weights = {
                "semantic": max(0.0, 1.0 - temporal_weight - 0.15),  # remainder
                "temporal": temporal_weight,
                "frequency": 0.10,
                "importance": 0.05,
            }

            # Determine score type for normalization
            score_type = "rrf" if reranker_name == "rrf_only" else "rerank"

            for result in results:
                meta = result.get("metadata", {})
                event_time = meta.get("event_time", "")
                category = meta.get("category", meta.get("memory_type", "default"))

                # Get raw semantic score and normalize to [0,1]
                raw_semantic = result.get("rerank_score") or result.get("rrf_score", 0.5)
                semantic_norm = normalize_semantic_score(raw_semantic, score_type)

                # Compute temporal decay
                temporal = temporal_decay_score(event_time, category, half_lives=half_lives)

                # Composite
                comp = composite_score(
                    semantic_score=semantic_norm,
                    temporal_score=temporal,
                    weights=weights,
                )

                result["semantic_score_normalized"] = round(semantic_norm, 6)
                result["temporal_score"] = round(temporal, 6)
                result["composite_score"] = round(comp, 6)

            # Re-sort by composite score (stable sort preserves original order for ties)
            results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

            logger.debug(
                "Temporal scoring applied to %d results (weight=%.2f).",
                len(results),
                temporal_weight,
            )
        except Exception as exc:
            logger.warning(
                "Temporal scoring failed; returning results without temporal adjustment: %s",
                exc,
                exc_info=True,
            )

        return results

    def _apply_mmr_dedup(
        self,
        results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """Apply exact deduplication and MMR reranking to final results.

        Gated by ``settings.MMR_ENABLED``.  On any error the original list
        is returned unchanged — the query pipeline is never blocked by
        deduplication failures.

        Args:
            results: Scored results from the rerank + temporal scoring pipeline.
            vector_results: Raw Pinecone results (may contain embedding values
                when ``include_values=True`` was used in the query).
            top_n: Maximum number of results to return after MMR selection.
        """
        try:
            if not getattr(settings, "MMR_ENABLED", True):
                return results
            if not results:
                return results

            # Step 1: Exact deduplication (MD5 on text)
            before_exact = len(results)
            results = deduplicate_exact(results)
            after_exact = len(results)
            if before_exact != after_exact:
                logger.info(
                    "MMR exact dedup removed %d duplicates (%d -> %d).",
                    before_exact - after_exact,
                    before_exact,
                    after_exact,
                )

            # Step 2: MMR reranking (requires embeddings)
            if len(results) <= 1:
                return results

            # Build an ID-to-embedding map from Pinecone vector results
            embedding_map: Dict[str, list] = {}
            for vr in vector_results:
                vid = vr.get("id", "")
                values = vr.get("values")
                if vid and values:
                    embedding_map[vid] = values

            # Align embeddings with current results
            embeddings: List[list] = []
            results_with_embeddings: List[Dict[str, Any]] = []
            results_without_embeddings: List[Dict[str, Any]] = []

            for r in results:
                rid = r.get("id", "")
                emb = embedding_map.get(rid)
                if emb:
                    results_with_embeddings.append(r)
                    embeddings.append(emb)
                else:
                    results_without_embeddings.append(r)

            if len(results_with_embeddings) <= 1:
                # Not enough embeddings available for meaningful MMR
                logger.debug(
                    "MMR skipped: only %d results have embeddings.",
                    len(results_with_embeddings),
                )
                return results

            lambda_param = getattr(settings, "MMR_LAMBDA", 0.7)
            mmr_selected = mmr_rerank(
                results_with_embeddings,
                embeddings,
                lambda_param=lambda_param,
                top_n=top_n,
            )

            # Append any results that lacked embeddings at the end
            # (graph-only results, etc.) so they aren't silently dropped
            combined = mmr_selected + results_without_embeddings
            combined = combined[:top_n]

            logger.info(
                "MMR rerank complete: %d -> %d results (lambda=%.2f).",
                len(results),
                len(combined),
                lambda_param,
            )
            return combined

        except Exception as exc:
            logger.warning(
                "MMR dedup failed; returning results without MMR: %s",
                exc,
                exc_info=True,
            )
            return results

    async def _cascading_rerank(
        self,
        query_text: str,
        fused_results: List[Dict[str, Any]],
        top_k_final: int,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Try rerankers in cascade: local -> Pinecone API -> RRF-only.

        Returns:
            (reranked_results, reranker_name) where reranker_name is one of
            'cross_encoder', 'pinecone_api', or 'rrf_only'.
        """
        if not fused_results:
            return [], "rrf_only"

        loop = asyncio.get_event_loop()

        # --- Tier 1: Local CrossEncoder ---
        if self.reranker:
            try:
                loaded = await self.reranker.ensure_loaded()
                if loaded:
                    t0 = loop.time()
                    result = await self.reranker.rerank(query_text, fused_results, top_n=top_k_final)
                    rerank_ms = (loop.time() - t0) * 1000
                    logger.info(
                        f"RERANK_LATENCY reranker=cross_encoder ms={rerank_ms:.0f} "
                        f"n_input={len(fused_results)} n_output={len(result)}"
                    )
                    self._reranker_loaded = True
                    return result, "cross_encoder"
                else:
                    logger.warning("CrossEncoder not available after ensure_loaded(). Trying Pinecone API.")
            except Exception as e:
                logger.error(f"CrossEncoder rerank failed: {e}. Trying Pinecone API.", exc_info=True)

        # --- Tier 2: Pinecone Inference API ---
        if self.pinecone_reranker:
            try:
                t0 = loop.time()
                result = await self.pinecone_reranker.rerank(query_text, fused_results, top_n=top_k_final)
                rerank_ms = (loop.time() - t0) * 1000
                logger.info(
                    f"RERANK_LATENCY reranker=pinecone_api ms={rerank_ms:.0f} "
                    f"n_input={len(fused_results)} n_output={len(result)}"
                )
                return result, "pinecone_api"
            except Exception as e:
                logger.error(f"PineconeReranker failed: {e}. Falling back to RRF-only.", exc_info=True)

        # --- Tier 3: RRF-only (no reranking) ---
        logger.info("All rerankers unavailable. Returning RRF-fused results without reranking.")
        logger.info(
            f"RERANK_LATENCY reranker=rrf_only ms=0 "
            f"n_input={len(fused_results)} n_output={min(top_k_final, len(fused_results))}"
        )
        return fused_results[:top_k_final], "rrf_only"

    def _resolve_memory_id(self, content: str, memory_id: Optional[str] = None) -> str:
        """Returns the caller-provided ID or deterministic MD5(content)."""
        return memory_id or hashlib.md5(content.encode()).hexdigest()

    async def _inject_chronology(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Inject system-enforced chronology fields into metadata.

        Always sets event_seq (monotonic, system-assigned — never caller-provided).
        Sets event_time to current UTC if not already provided by the caller.
        Sets memory_type to 'scratch' if not already provided.

        This is the "can't forget" solution — agents don't need to remember
        to add timestamps because the system always does it.
        """
        metadata["event_seq"] = await self.sequence_service.next_seq()
        if "event_time" not in metadata or not metadata["event_time"]:
            metadata["event_time"] = datetime.now(timezone.utc).isoformat()
        if "memory_type" not in metadata or not metadata["memory_type"]:
            metadata["memory_type"] = "scratch"
        return metadata

    async def _persist_memory_item(
        self,
        item_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]],
        embedding: List[float],
    ) -> bool:
        """
        Persists one memory item to Pinecone and Neo4j.

        Returns:
            True if both writes succeed, otherwise False (with rollback attempts).
        """
        pinecone_meta = metadata.copy() if metadata else {}
        pinecone_meta["text"] = content

        pinecone_success = False
        graph_success = False

        try:
            pinecone_success = await asyncio.to_thread(
                self.pinecone_client.upsert_vector, item_id, embedding, pinecone_meta
            )
        except Exception as e:
            logger.error(
                f"Error during Pinecone upsert execution for ID {item_id}: {e}",
                exc_info=True,
            )
            pinecone_success = False

        try:
            graph_success = await self.graph_client.upsert_graph_data(
                item_id, content, metadata
            )
        except Exception as e:
            logger.error(
                f"Error during Graph upsert execution for ID {item_id}: {e}",
                exc_info=True,
            )
            graph_success = False

        if pinecone_success and graph_success:
            logger.info(f"Successfully upserted ID {item_id} to both Pinecone and Graph.")
            return True

        if pinecone_success and not graph_success:
            logger.error(
                f"Upsert failed for ID {item_id}: succeeded in Pinecone but failed in Graph."
            )
            logger.warning(
                f"Attempting rollback: deleting ID {item_id} from Pinecone due to graph failure."
            )
            rollback_ok = await asyncio.to_thread(self.pinecone_client.delete_vector, item_id)
            logger.warning(f"Pinecone rollback successful: {rollback_ok}")
            return False

        if not pinecone_success and graph_success:
            logger.error(
                f"Upsert failed for ID {item_id}: succeeded in Graph but failed in Pinecone."
            )
            logger.warning(
                f"Attempting rollback: deleting ID {item_id} from Graph due to Pinecone failure."
            )
            rollback_ok = await self.graph_client.delete_graph_data(item_id)
            logger.warning(f"Graph rollback successful: {rollback_ok}")
            return False

        logger.error(f"Upsert failed for ID {item_id}: failed in both Pinecone and Graph.")
        return False

    async def perform_upsert(
        self,
        content: str,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Upserts memory content into both Pinecone and Neo4j.

        Args:
            content: The text content to store.
            memory_id: Optional specific ID. If None, an MD5 hash of content is used.
            metadata: Optional dictionary of metadata.

        Returns:
            The ID of the upserted item, or None if upsert failed.
        """
        if not self._initialized:
            logger.error("MemoryService not initialized. Cannot perform upsert.")
            return None
        if not content:
            logger.warning("Received empty content for upsert.")
            return None

        item_id = self._resolve_memory_id(content, memory_id)
        logger.info(f"Performing upsert for ID: {item_id}, Content: '{content[:100]}...'")

        # System-enforced chronology (Phase 1) — cannot be forgotten
        metadata = metadata if metadata is not None else {}
        metadata = await self._inject_chronology(metadata)

        try:
            embedding = await asyncio.to_thread(
                get_embedding, content, self.embedding_model_name
            )
            if not any(embedding):
                logger.error(
                    f"Failed to get valid embedding for upsert ID {item_id}. Aborting."
                )
                return None
        except Exception as e:
            logger.error(
                f"Error getting embedding for upsert ID {item_id}: {e}",
                exc_info=True,
            )
            return None

        # P9A.5: Write-time semantic deduplication (best-effort)
        duplicate_info = None
        conflict_info = None
        try:
            if getattr(settings, "WRITE_DEDUP_ENABLED", True):
                threshold = getattr(settings, "WRITE_DEDUP_THRESHOLD", 0.92)
                duplicate_info = await check_semantic_duplicate(
                    self.pinecone_client, content, embedding,
                    threshold=threshold, exclude_id=item_id,
                )
                if duplicate_info:
                    category = metadata.get("category", metadata.get("memory_type", ""))
                    on_duplicate = metadata.pop("on_duplicate", "auto")
                    action = resolve_duplicate_action(duplicate_info, on_duplicate, category)
                    logger.info(
                        f"WRITE_DEDUP id={item_id} action={action} "
                        f"duplicate_id={duplicate_info['id']} score={duplicate_info['score']:.3f}"
                    )
                    if action == "skip":
                        return duplicate_info["id"]  # Return existing ID
                    elif action == "update":
                        item_id = duplicate_info["id"]  # Overwrite the existing entry

            if getattr(settings, "CONFLICT_DETECTION_ENABLED", True):
                category = metadata.get("category", metadata.get("memory_type", ""))
                if category == "decision":
                    conflict_info = await detect_conflicts(
                        self.pinecone_client, content, embedding, category="decision"
                    )
                    if conflict_info:
                        logger.info(
                            f"WRITE_CONFLICT id={item_id} conflicts={len(conflict_info)} "
                            f"ids={[c['id'] for c in conflict_info]}"
                        )
        except Exception as e:
            logger.warning(f"Write-time dedup/conflict check failed (non-fatal): {e}")

        success = await self._persist_memory_item(
            item_id=item_id,
            content=content,
            metadata=metadata,
            embedding=embedding,
        )

        # Phase 5: Auto-link event to session if session_id is provided
        if success and metadata.get("session_id"):
            try:
                await self.graph_client.link_event_to_session(
                    event_id=item_id, session_id=metadata["session_id"]
                )
            except Exception as e:
                # Non-fatal — event is already persisted, linking is best-effort
                logger.warning(
                    f"Failed to link event {item_id} to session "
                    f"{metadata['session_id']}: {e}"
                )

        # Phase 6: Record in Redis timeline (best-effort)
        if success and self.redis_timeline:
            try:
                scope = metadata.get("project", "global")
                await self.redis_timeline.record_event(
                    event_seq=metadata["event_seq"],
                    memory_id=item_id,
                    metadata_summary={
                        "memory_type": metadata.get("memory_type"),
                        "project": metadata.get("project"),
                        "session_id": metadata.get("session_id"),
                    },
                    scope=scope,
                )
            except Exception as e:
                logger.warning(f"Failed to record event in Redis timeline: {e}")

        return item_id if success else None

    async def perform_bulk_upsert(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upserts multiple memory items in one call.

        Expected item shape:
            {"content": str, "id": Optional[str], "metadata": Optional[dict]}
        """
        if not self._initialized:
            logger.error("MemoryService not initialized. Cannot perform bulk upsert.")
            return {
                "status": "error",
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "ids": [],
                "errors": [{"index": -1, "error": "MemoryService not initialized"}],
            }

        if not items:
            logger.warning("perform_bulk_upsert called with no items.")
            return {
                "status": "error",
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "ids": [],
                "errors": [{"index": -1, "error": "No items provided"}],
            }

        normalized_items: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append({"index": index, "error": "Each item must be an object"})
                continue

            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                errors.append(
                    {
                        "index": index,
                        "error": "Item.content is required and must be a non-empty string",
                    }
                )
                continue

            raw_metadata = item.get("metadata")
            if raw_metadata is None:
                metadata: Dict[str, Any] = {}
            elif isinstance(raw_metadata, dict):
                metadata = raw_metadata
            else:
                errors.append(
                    {
                        "index": index,
                        "error": "Item.metadata must be an object when provided",
                    }
                )
                continue

            raw_id = item.get("id")
            if raw_id is not None and not isinstance(raw_id, str):
                errors.append(
                    {"index": index, "error": "Item.id must be a string when provided"}
                )
                continue

            resolved_id = self._resolve_memory_id(content=content, memory_id=raw_id)
            normalized_items.append(
                {
                    "index": index,
                    "content": content,
                    "memory_id": resolved_id,
                    "metadata": metadata,
                }
            )

        successful_ids: List[str] = []

        # System-enforced chronology (Phase 1) — batch sequence allocation
        if normalized_items:
            now_iso = datetime.now(timezone.utc).isoformat()
            seq_batch = await self.sequence_service.next_batch(len(normalized_items))
            for i, entry in enumerate(normalized_items):
                entry["metadata"]["event_seq"] = seq_batch[i]
                if "event_time" not in entry["metadata"] or not entry["metadata"]["event_time"]:
                    entry["metadata"]["event_time"] = now_iso
                if "memory_type" not in entry["metadata"] or not entry["metadata"]["memory_type"]:
                    entry["metadata"]["memory_type"] = "scratch"

        if normalized_items:
            contents = [entry["content"] for entry in normalized_items]
            embeddings = await batch_get_embeddings(contents, self.embedding_model_name)

            persist_tasks = []
            valid_entries: List[Dict[str, Any]] = []
            for idx, entry in enumerate(normalized_items):
                embedding = embeddings[idx] if idx < len(embeddings) else []
                if not embedding or not any(embedding):
                    errors.append(
                        {
                            "index": entry["index"],
                            "id": entry["memory_id"],
                            "error": "Failed to generate embedding",
                        }
                    )
                    continue

                valid_entries.append(entry)
                persist_tasks.append(
                    self._persist_memory_item(
                        item_id=entry["memory_id"],
                        content=entry["content"],
                        metadata=entry["metadata"],
                        embedding=embedding,
                    )
                )

            if persist_tasks:
                persist_results = await asyncio.gather(
                    *persist_tasks, return_exceptions=True
                )
                for entry, result in zip(valid_entries, persist_results):
                    if isinstance(result, Exception):
                        errors.append(
                            {
                                "index": entry["index"],
                                "id": entry["memory_id"],
                                "error": f"Unexpected exception: {result}",
                            }
                        )
                        continue

                    if result:
                        successful_ids.append(entry["memory_id"])

                        # Phase 5: Auto-link event to session (best-effort)
                        sid = entry["metadata"].get("session_id")
                        if sid:
                            try:
                                await self.graph_client.link_event_to_session(
                                    event_id=entry["memory_id"],
                                    session_id=sid,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"bulk_upsert: failed to link {entry['memory_id']} "
                                    f"to session {sid}: {e}"
                                )

                        # Phase 6: Record in Redis timeline (best-effort)
                        if self.redis_timeline:
                            try:
                                scope = entry["metadata"].get("project", "global")
                                await self.redis_timeline.record_event(
                                    event_seq=entry["metadata"]["event_seq"],
                                    memory_id=entry["memory_id"],
                                    metadata_summary={
                                        "memory_type": entry["metadata"].get("memory_type"),
                                        "project": entry["metadata"].get("project"),
                                        "session_id": entry["metadata"].get("session_id"),
                                    },
                                    scope=scope,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"bulk_upsert: failed to record {entry['memory_id']} "
                                    f"in Redis timeline: {e}"
                                )
                    else:
                        errors.append(
                            {
                                "index": entry["index"],
                                "id": entry["memory_id"],
                                "error": "Upsert failed in one or more stores",
                            }
                        )

        total = len(items)
        succeeded = len(successful_ids)
        failed = total - succeeded

        if failed == 0:
            status = "success"
        elif succeeded > 0:
            status = "partial_success"
        else:
            status = "error"

        logger.info(
            f"Bulk upsert complete. Total={total}, Succeeded={succeeded}, Failed={failed}"
        )
        return {
            "status": status,
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "ids": successful_ids,
            "errors": errors,
        }

    async def perform_delete(self, memory_id: str) -> bool:
        """
        Deletes a memory item from both Pinecone and Neo4j.

        Args:
            memory_id: The ID of the item to delete.

        Returns:
            True if deletion was successful in at least one store (or item didn't exist), False otherwise.
        """
        if not self._initialized:
            logger.error("MemoryService not initialized. Cannot perform delete.")
            return False
        if not memory_id:
            logger.warning("Received empty memory_id for delete.")
            return False

        logger.info(f"Performing delete for ID: {memory_id}")

        # Attempt deletions in parallel
        try:
            results = await asyncio.gather(
                asyncio.to_thread(self.pinecone_client.delete_vector, memory_id), # Run sync in thread
                self.graph_client.delete_graph_data(memory_id), # Already async
                return_exceptions=True
            )

            pinecone_success = isinstance(results[0], bool) and results[0]
            graph_success = isinstance(results[1], bool) and results[1]

            if isinstance(results[0], Exception):
                 logger.error(f"Error during Pinecone delete thread execution for ID {memory_id}: {results[0]}", exc_info=results[0])
            if isinstance(results[1], Exception):
                 logger.error(f"Error during Graph delete execution for ID {memory_id}: {results[1]}", exc_info=results[1])

            if pinecone_success or graph_success:
                 logger.info(f"Deletion attempt for ID {memory_id} complete. Pinecone success: {pinecone_success}, Graph success: {graph_success}")
                 # Consider successful if at least one deletion worked or if ID didn't exist in one/both
                 return True
            else:
                 logger.error(f"Deletion failed for ID {memory_id} in both stores.")
                 return False # Failed in both

        except Exception as e:
            logger.error(f"Unexpected error during parallel delete for ID {memory_id}: {e}", exc_info=True)
            return False

    # --- Phase 2: Session Checkpoints ---

    CHECKPOINT_REQUIRED_FIELDS = {"session_id", "session_summary"}

    async def create_checkpoint(
        self,
        session_id: str,
        session_summary: str,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        open_threads: Optional[List[str]] = None,
        next_actions: Optional[List[str]] = None,
        project: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[str]:
        """Creates a session checkpoint — a structured summary of a completed session.

        Snapshots `last_event_seq` (the highest event_seq at checkpoint time)
        so downstream consumers know exactly which events preceded this checkpoint.

        System auto-injects event_time and event_seq via _inject_chronology.

        Returns:
            The ID of the checkpoint memory item, or None on failure.
        """
        if not session_id or not session_id.strip():
            logger.error("create_checkpoint: session_id is required.")
            return None
        if not session_summary or not session_summary.strip():
            logger.error("create_checkpoint: session_summary is required.")
            return None

        last_seq = self.sequence_service.current_seq()

        metadata: Dict[str, Any] = {
            "memory_type": "checkpoint",
            "session_id": session_id.strip(),
            "session_summary": session_summary.strip(),
            "last_event_seq": last_seq,
        }

        # Add optional fields only if provided
        if started_at:
            metadata["started_at"] = started_at
        if ended_at:
            metadata["ended_at"] = ended_at
        if open_threads:
            metadata["open_threads"] = open_threads
        if next_actions:
            metadata["next_actions"] = next_actions
        if project:
            metadata["project"] = project
        if thread_id:
            metadata["thread_id"] = thread_id

        content = f"Session checkpoint: {session_id.strip()}\n\n{session_summary.strip()}"

        logger.info(f"Creating checkpoint for session '{session_id}' (last_event_seq={last_seq})")
        checkpoint_id = await self.perform_upsert(content=content, metadata=metadata)

        # Phase 5: Create Session node in graph + FOLLOWS chain
        if checkpoint_id:
            try:
                await self.graph_client.create_session_node(
                    session_id=session_id.strip(),
                    started_at=started_at,
                    ended_at=ended_at,
                    last_event_seq=last_seq,
                    summary=session_summary.strip(),
                    project=project,
                    thread_id=thread_id,
                )

                # Link to previous session (FOLLOWS edge)
                prev_session = await self.graph_client.get_latest_session(
                    project=project
                )
                if prev_session and prev_session.get("session_id") != session_id.strip():
                    await self.graph_client.link_session_follows(
                        current_session_id=session_id.strip(),
                        previous_session_id=prev_session["session_id"],
                    )
            except Exception as e:
                # Non-fatal — checkpoint memory item is already persisted
                logger.warning(
                    f"Failed to create session graph structures for {session_id}: {e}"
                )

            # Phase 6: Record checkpoint in Redis timeline
            if self.redis_timeline:
                try:
                    scope = project or "global"
                    await self.redis_timeline.record_checkpoint(
                        session_id=session_id.strip(),
                        last_event_seq=last_seq,
                        summary=session_summary.strip(),
                        scope=scope,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to record checkpoint in Redis timeline: {e}"
                    )

        return checkpoint_id

    async def get_last_checkpoint(
        self,
        project: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent session checkpoint.

        Deterministic — does NOT use semantic search.
        Primary: Redis sorted-set O(1) lookup.
        Fallback: Pinecone metadata filter + client-side sort.

        Args:
            project: Optional filter to only return checkpoints for this project.
            thread_id: Optional filter to only return checkpoints for this thread.

        Returns:
            The checkpoint dict or None if no checkpoints exist.
        """
        if not self._initialized:
            logger.error("MemoryService not initialized. Cannot get checkpoint.")
            return None

        # Phase 6: Redis fast path (O(1) via ZREVRANGE 0 0)
        if self.redis_timeline and not thread_id:
            try:
                scope = project or "global"
                result = await self.redis_timeline.get_last_checkpoint(scope=scope)
                if result:
                    entry, last_event_seq = result
                    checkpoint = {
                        "session_id": entry.get("session_id"),
                        "summary": entry.get("summary"),
                        "last_event_seq": int(last_event_seq),
                        "source": "redis_timeline",
                    }
                    logger.info(
                        f"Found latest checkpoint (Redis): session_id={checkpoint['session_id']}, "
                        f"last_event_seq={checkpoint['last_event_seq']}"
                    )
                    return checkpoint
            except Exception as e:
                logger.warning(f"Redis checkpoint lookup failed, falling back to Pinecone: {e}")

        # Pinecone fallback path
        filter_dict: Dict[str, Any] = {"memory_type": {"$eq": "checkpoint"}}
        if project:
            filter_dict["project"] = {"$eq": project}
        if thread_id:
            filter_dict["thread_id"] = {"$eq": thread_id}

        try:
            from app.services.embedding_service import EMBEDDING_DIM

            dummy_vector = [0.0] * EMBEDDING_DIM
            results = await asyncio.to_thread(
                self.pinecone_client.query_vector,
                dummy_vector,
                top_k=20,
                filter=filter_dict,
            )

            if not results:
                logger.info("No checkpoints found matching the filter.")
                return None

            # Sort by event_seq descending, return the latest
            results.sort(
                key=lambda r: r.get("metadata", {}).get("event_seq", 0),
                reverse=True,
            )
            latest = results[0]

            # Sanitize for JSON serialization — convert any non-serializable values
            def _sanitize(obj: Any) -> Any:
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                return str(obj)

            latest = _sanitize(latest)
            logger.info(
                f"Found latest checkpoint (Pinecone): session_id="
                f"{latest.get('metadata', {}).get('session_id')}, "
                f"event_seq={latest.get('metadata', {}).get('event_seq')}"
            )
            return latest

        except Exception as e:
            logger.error(f"Failed to get last checkpoint: {e}", exc_info=True)
            return None

    # --- Phase 3: Temporal Retrieval ---

    OVER_FETCH_FACTOR = 5
    MAX_OVER_FETCH = 10000

    async def get_recent_events(
        self,
        n: int = 20,
        project: Optional[str] = None,
        thread_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        since_seq: Optional[int] = None,
        since_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves the N most recent memory events ordered by event_seq descending.

        Purely metadata-driven — does NOT use semantic similarity.
        Over-fetches from Pinecone with a dummy vector, filters by metadata,
        then sorts client-side by event_seq.

        Args:
            n: Number of events to return (default 20, max 200).
            project: Filter by project name.
            thread_id: Filter by thread_id.
            memory_type: Filter by memory_type (e.g. "scratch", "decision", "checkpoint").
            since_seq: Only return events with event_seq >= this value.
            since_time: Only return events with event_time >= this ISO 8601 string.

        Returns:
            List of memory items sorted by event_seq descending.
        """
        if not self._initialized:
            logger.error("MemoryService not initialized. Cannot get recent events.")
            return []

        n = max(1, min(n, 200))  # Clamp to [1, 200]

        # Phase 6: Use Redis timeline when available (O(log N) vs Pinecone dummy-vector)
        if self.redis_timeline and not thread_id and not since_time:
            # Redis path — fast sorted set queries
            # (thread_id and since_time filtering not yet in Redis, fall through to Pinecone)
            try:
                scope = project or "global"
                if since_seq is not None:
                    raw_timeline = await self.redis_timeline.get_since_seq(
                        since_seq=since_seq, scope=scope
                    )
                    # Reverse for descending order, then slice
                    raw_timeline.reverse()
                else:
                    raw_timeline = await self.redis_timeline.get_recent(
                        n=n * self.OVER_FETCH_FACTOR, scope=scope
                    )

                if raw_timeline:
                    # Filter by memory_type client-side if specified
                    results = []
                    for entry, seq in raw_timeline:
                        if memory_type and entry.get("memory_type") != memory_type:
                            continue
                        results.append({
                            "id": entry.get("id"),
                            "score": 0.0,
                            "metadata": {
                                "event_seq": seq,
                                "memory_type": entry.get("memory_type"),
                                "project": entry.get("project"),
                                "session_id": entry.get("session_id"),
                            },
                            "source": "redis_timeline",
                        })
                        if len(results) >= n:
                            break
                    if results:
                        logger.info(
                            f"get_recent_events (Redis): returning {len(results)} events."
                        )
                        return results
                    # If Redis returned nothing after filtering, fall through to Pinecone
            except Exception as e:
                logger.warning(f"Redis timeline query failed, falling back to Pinecone: {e}")

        # Pinecone fallback path
        filter_dict: Dict[str, Any] = {}
        if project:
            filter_dict["project"] = {"$eq": project}
        if thread_id:
            filter_dict["thread_id"] = {"$eq": thread_id}
        if memory_type:
            filter_dict["memory_type"] = {"$eq": memory_type}
        if since_seq is not None:
            filter_dict["event_seq"] = {"$gte": since_seq}
        if since_time:
            filter_dict["event_time"] = {"$gte": since_time}

        fetch_count = min(n * self.OVER_FETCH_FACTOR, self.MAX_OVER_FETCH)

        try:
            dummy_vector = [0.0] * 1536
            raw_results = await asyncio.to_thread(
                self.pinecone_client.query_vector,
                dummy_vector,
                top_k=fetch_count,
                filter=filter_dict if filter_dict else None,
            )

            if not raw_results:
                logger.info("get_recent_events: no results from Pinecone.")
                return []

            # Sort by event_seq descending (client-side)
            raw_results.sort(
                key=lambda r: r.get("metadata", {}).get("event_seq", 0),
                reverse=True,
            )

            results = raw_results[:n]
            logger.info(
                f"get_recent_events: returning {len(results)} of {len(raw_results)} "
                f"fetched (requested n={n})"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to get recent events: {e}", exc_info=True)
            return []

    async def check_health(self) -> Dict[str, str]:
        """
        Checks the health of the service and its dependencies.

        Returns:
            A dictionary indicating the status of each component.
        """
        if not self._initialized:
            return {"status": "error", "detail": "MemoryService not initialized"}

        statuses = {"status": "ok"} # Assume ok initially

        # Check dependencies in parallel
        try:
            results = await asyncio.gather(
                asyncio.to_thread(self.pinecone_client.check_connection), # Run sync in thread
                self.graph_client.check_connection(), # Already async
                return_exceptions=True
            )

            # Pinecone status
            if isinstance(results[0], bool) and results[0]:
                statuses["pinecone"] = "ok"
            else:
                statuses["pinecone"] = f"error: {results[0]}" if isinstance(results[0], Exception) else "error: Failed check"
                statuses["status"] = "error" # Overall status degraded

            # Graph status
            if isinstance(results[1], bool) and results[1]:
                statuses["graph"] = "ok"
            else:
                statuses["graph"] = f"error: {results[1]}" if isinstance(results[1], Exception) else "error: Failed check"
                statuses["status"] = "error" # Overall status degraded

            # Reranker status (based on loading)
            statuses["reranker"] = "loaded" if self._reranker_loaded else "disabled/failed"

            # Redis status (Phase 6)
            statuses["redis"] = "connected" if self.sequence_service._using_redis else "disabled/fallback"
            statuses["redis_timeline"] = "active" if self.redis_timeline else "inactive"

        except Exception as e:
            logger.error(f"Unexpected error during health check: {e}", exc_info=True)
            statuses["status"] = "error"
            statuses["detail"] = f"Unexpected error: {e}"

        return statuses
