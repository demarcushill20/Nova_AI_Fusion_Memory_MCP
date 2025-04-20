import os
import re
import torch
import torch.nn as nn
import numpy as np
import openai
import pinecone
import tkinter as tk
import tkinter.scrolledtext
import threading
import asyncio
import time
import logging
import inspect
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv
from contextlib import contextmanager
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Iterator, Generator, Union
from hashlib import md5
import faiss
from concurrent.futures import ThreadPoolExecutor
from cachetools import LRUCache, TTLCache
from functools import lru_cache
import json # For parsing LLM response

# Import Eleven Labs with the client-based approach
from lightrag.kg.neo4j_impl import Neo4JStorage # Correct class name
from neo4j import GraphDatabase, exceptions as neo4j_exceptions # For potential direct driver usage or error handling
from hybrid_merger import HybridMerger # Import the new merger class
from query_router import QueryRouter, RoutingMode # Import the new router
from reranker import CrossEncoderReranker # Import the new reranker

# ------------------------------
# Configure Logging - MODIFIED TO HIDE DEBUG OUTPUT
# ------------------------------
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up file handler for logs instead of console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only show warnings and errors

# Create a file handler for logs
file_handler = logging.FileHandler('logs/nova_ai.log', encoding='utf-8') # Specify UTF-8 encoding
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Prevent logs from propagating to root logger (which outputs to console)
logger.propagate = False

# Force production mode to minimize logging
PRODUCTION_MODE = True
logging_level = logging.WARNING  # Only warnings and errors

# Global thread pool for async operations
thread_pool = ThreadPoolExecutor(max_workers=10)
# Removed background loop globals and get_background_loop function

# run_async_with_result function removed as it causes issues with nested event loops.
# Calls should be refactored to use await directly in async contexts
# or use thread_pool.submit for sync contexts calling async code via a bridge if necessary.

# ------------------------------
# Initialize API Keys and Environment
# ------------------------------
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Configure ffmpeg path if it exists in the local directory
# Set up ffmpeg

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# ------------------------------
# Initialize Pinecone Client
# ------------------------------
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "nova-ai-memory"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine"
    )
index = pc.Index(index_name)

# ------------------------------
# Global Embedding Cache with LRU Eviction and Thread Safety
# ------------------------------
# Using TTLCache: items expire after 24 hours, max size 1000 items
embedding_cache = TTLCache(maxsize=1000, ttl=86400)  # 24 hours TTL
# Add a threading lock for thread-safe cache access
embedding_cache_lock = threading.RLock()

@lru_cache(maxsize=128)  # Additional LRU cache for extremely frequent requests
def get_embedding(text: str) -> List[float]:
    """
    Retrieves the embedding for the given text using OpenAI embeddings.
    Uses a thread-safe cache to reduce redundant API calls.
    """
    # Thread-safe cache lookup
    with embedding_cache_lock:
        if text in embedding_cache:
            return embedding_cache[text]
    try:
        response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
        embedding = response.data[0].embedding

        # Thread-safe cache update
        with embedding_cache_lock:
            embedding_cache[text] = embedding
        return embedding
    except Exception as e: # Catch specific exceptions if possible
        logger.error(f"Error getting embedding for text '{text[:50]}...': {e}")
        return [0.0] * 1536 # Return zero vector on error

# Batch embedding function for multiple texts
async def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts in a single API call when possible.
    Thread-safe implementation for concurrent access."""
    # First check cache for all texts (with thread safety)
    cached_embeddings = []
    texts_to_fetch = []
    text_indices = []

    with embedding_cache_lock:
        for i, text in enumerate(texts):
            if text in embedding_cache:
                cached_embeddings.append((i, embedding_cache[text]))
            else:
                texts_to_fetch.append(text)
                text_indices.append(i)

    # If all were cached, return immediately
    if not texts_to_fetch:
        # Sort by original index and extract just the embeddings
        return [emb for _, emb in sorted(cached_embeddings, key=lambda x: x[0])]

    # Otherwise fetch the missing embeddings
    try:
        response = await asyncio.to_thread(
            lambda: openai.embeddings.create(input=texts_to_fetch, model="text-embedding-ada-002")
        )

        # Process the results
        fetched_embeddings = []

        # Thread-safe cache update
        with embedding_cache_lock:
            for i, data in enumerate(response.data):
                embedding = data.embedding
                # Cache the result
                embedding_cache[texts_to_fetch[i]] = embedding
                fetched_embeddings.append((text_indices[i], embedding))

        # Combine cached and fetched results
        all_embeddings = cached_embeddings + fetched_embeddings
        # Sort by original index and extract just the embeddings
        return [emb for _, emb in sorted(all_embeddings, key=lambda x: x[0])]
    except Exception as e: # Catch specific exceptions if possible
        logger.error(f"Error getting batch embeddings: {e}")
        # Return zeros for the missing embeddings
        all_embeddings = cached_embeddings
        for i in text_indices:
            all_embeddings.append((i, [0.0] * 1536))
        return [emb for _, emb in sorted(all_embeddings, key=lambda x: x[0])]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes cosine similarity between two vectors
    """
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    if np.linalg.norm(v1_arr) == 0 or np.linalg.norm(v2_arr) == 0:
        return 0.0
    return float(np.dot(v1_arr, v2_arr) / (np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr)))

# ------------------------------
# Circuit Breaker and Fallback System
# ------------------------------
class CircuitBreakerError(Exception):
    pass

@contextmanager
def circuit_breaker(service_name: str):
    """
    Dummy circuit breaker context manager.
    In a real scenario, health checks would be here.
    """
    try:
        yield
    except Exception as e:
        raise CircuitBreakerError(f"{service_name} is currently unavailable: {e}") from e

def kg_search(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dummy local knowledge graph search for fallback.
    """
    logger.info("Performing knowledge graph fallback search for query: %s", query)
    return {"matches": [{"metadata": {"text": "Fallback result from local knowledge graph."}}]}

def validate_result(result: Dict[str, Any]) -> bool:
    """
    Validates that the query result contains matches.
    """
    valid = bool(result.get("matches"))
    if not valid:
        logger.debug("Validation failed: No matches found.")
    return valid

class FallbackSystem:
    def __init__(self, pinecone_index: Any):
        self.pinecone_index = pinecone_index

    def query_with_fallback(self, query: Dict[str, Any], max_retries: int = 3, initial_delay: float = 1.0) -> Dict[str, Any]:
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                with circuit_breaker("pinecone"):
                    result = self.pinecone_index.query(**query)
                if validate_result(result):
                    logger.debug("Pinecone query succeeded at attempt %d.", attempt)
                    return result
                else:
                    logger.warning("Validation failed; retry attempt %d.", attempt)
            except CircuitBreakerError as cbe:
                logger.error("Circuit breaker error on attempt %d: %s", attempt, cbe, exc_info=True)
            except Exception as e:
                logger.error("Error on attempt %d: %s", attempt, e, exc_info=True)
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
        logger.info("All attempts failed; using fallback local knowledge graph search.")
        return self.degraded_query(query)

    def degraded_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Executing degraded fallback query.")
        return kg_search(query)

# ------------------------------
# Asynchronous Upsert to Pinecone with Semantic Deduplication
# ------------------------------
async def async_upsert_memory_to_pinecone(
    user_query: str,
    nova_response: Optional[str],
    category: str = "conversation",
    filename: Optional[str] = None # Add optional filename parameter
) -> None:
    """
    Asynchronously upserts conversation or memory data into Pinecone.
    Uses a hash of the content as the ID for better deduplication.
    Includes original filename in metadata if provided.
    """
    try:
        # Determine the full text content based on category
        if category == "conversation":
            if not nova_response:
                 # If it's a conversation category but no response, maybe just store user query?
                 # Or raise error? For now, let's assume we store only user query if response missing.
                 # This behavior might need refinement based on desired logic.
                 logger.warning(f"Conversation category upsert called with no Nova response for query: {user_query[:100]}...")
                 full_memory = f"User: {user_query}" # Store only user query part
                 # Adjust category if needed, e.g., "user_input_only"
                 # category = "user_input_only"
            else:
                 full_memory = f"User: {user_query}\nNova: {nova_response}"
        else:
            # For non-conversation categories, user_query holds the full text
            full_memory = user_query
            if not full_memory:
                 logger.warning(f"Upsert called with empty content for category {category}. Skipping.")
                 return

        # Generate ID based on content hash
        vector_id = md5(full_memory.encode()).hexdigest()

        # Prepare metadata, ensuring 'text' is included
        metadata = {"text": full_memory, "category": category}
        if filename:
            metadata["source_filename"] = filename # Add filename if provided

        # Get embedding (can be done concurrently with metadata prep)
        new_embedding = await asyncio.to_thread(get_embedding, full_memory)

        # Prepare vector for upsert
        upsert_vector = [{"id": vector_id, "values": new_embedding, "metadata": metadata}]

        # Run Pinecone upsert in a thread to avoid blocking
        await asyncio.to_thread(index.upsert, vectors=upsert_vector)
        log_text = full_memory[:150] + ('...' if len(full_memory) > 150 else '')
        logger.info(f"✅ Upserted ID {vector_id} (Category: {category}, Filename: {filename or 'N/A'}): '{log_text}'")

    except ValueError as ve:
         logger.error(f"❌ Value error during async upsert: {ve}")
    except Exception as e:
        logger.error(f"❌ Unexpected error in async upsert: {e}", exc_info=True)

# Removed run_async function as background loop is removed. Use run_async_with_result instead.

# ------------------------------
# Old Transformer-based Memory Ranker (REMOVED - Replaced by CrossEncoderReranker)
# ------------------------------
# class MemoryRankingTransformer(nn.Module): ... (Removed)

# ------------------------------
# Helper Functions
# ------------------------------
def resolve_relative_date(query: str) -> str:
    now = datetime.now()
    if "today" in query.lower():
        today_str = now.strftime("%B %d, %Y")
        query = query.replace("today", today_str).replace("Today", today_str)
    if "yesterday" in query.lower():
        yesterday = now - timedelta(days=1)
        yesterday_str = yesterday.strftime("%B %d, %Y")
        query = query.replace("yesterday", yesterday_str).replace("Yesterday", yesterday_str)
    if "last week" in query.lower():
        last_monday = now - timedelta(days=now.weekday() + 7)
        last_sunday = last_monday + timedelta(days=6)
        last_week_str = f"between {last_monday.strftime('%B %d, %Y')} and {last_sunday.strftime('%B %d, %Y')}"
        query = query.replace("last week", last_week_str).replace("Last week", last_week_str)
    if "last month" in query.lower():
        first_day_this_month = now.replace(day=1)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        last_month_str = f"between {first_day_last_month.strftime('%B %d, %Y')} and {last_day_last_month.strftime('%B %d, %Y')}"
        query = query.replace("last month", last_month_str).replace("Last month", last_month_str)
    return query

def choose_model(query: str) -> str:
    return "gpt-4o"  # Always using GPT-4o

def extract_subtopics_via_llm(context: str) -> List[str]:
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": f"Extract key technical subtopics from the following context: {context}"}]
        )
        topics = response.choices[0].message.content
        return [topic.strip().lower() for topic in topics.split(",")]
    except Exception as e:
        logger.error("❌ Error extracting subtopics: %s", e)
        return []

def semantic_reranker(memories: List[str], query: str) -> List[str]:
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": f"Re-rank these Project Nova memories by relevance to '{query}':\n" + "\n".join(memories)}]
        )
        reranked = [mem.strip() for mem in response.choices[0].message.content.split("\n") if mem.strip()]
        return reranked
    except Exception as e:
        logger.error("❌ Error in semantic reranker: %s", e)
        return memories

# Removed old rank_memories_with_transformer function

async def retrieve_pinecone_memory_async(query_text: str, category: str = "conversation", fallback_system: Optional[FallbackSystem] = None) -> List[Dict[str, Any]]:
    """
    Asynchronously retrieves memory results (including metadata and scores) from Pinecone.
    Incorporates enhanced recall patterns and multi-step fallback search.
    """
    try:
        # --- Enhanced Recall Pattern Detection & Category Handling ---
        query_text_lower = query_text.lower()
        recall_patterns = [
            "remember", "recall", "memory", "past", "previous",
            "do you know", "do you have", "heard of", "familiar with",
            "know anything about", "tell me about", "what about", "know of"
        ]
        is_recall_query = any(pattern in query_text_lower for pattern in recall_patterns)
        logger.warning(f"[DBG.2] Is recall query: {is_recall_query}")

        # Determine primary category for search
        primary_category = category # Keep original category unless overridden
        if is_recall_query:
            primary_category = "past_conversation"
            logger.warning(f"[DBG.2] RECALL query detected. Overriding primary category to '{primary_category}'")
        else:
            # If not a recall query, use the provided category (usually 'conversation')
            logger.warning(f"[DBG.2] Not a recall query. Using primary category: '{primary_category}'")

        # Existing boost keywords logic
        boost_keywords = ["twin-charging", "ai system", "awd launch", "launch", "horsepower"]
        if "project nova" in query_text_lower and not any(kw in query_text_lower for kw in boost_keywords):
            query_text_augmented = query_text + " " + " ".join(boost_keywords)
        else:
            query_text_augmented = query_text

        query_embedding = await asyncio.to_thread(get_embedding, query_text_augmented)

        # Add debug log for filter
        filter_dict = {"category": primary_category} # Use primary_category for the first attempt
        logger.warning(f"[DBG.2] Using initial Pinecone filter: {filter_dict}")

        query_params = {
            "vector": query_embedding,
            "top_k": 50, # Retrieve more for reranker
            "include_metadata": True,
            "include_values": False, # Don't need vectors after retrieval usually
            "filter": filter_dict
        }

        # --- Initial Query ---
        if fallback_system:
            results = await asyncio.to_thread(fallback_system.query_with_fallback, query_params)
        else:
            results = await asyncio.to_thread(lambda: index.query(**query_params))

        match_count = len(results.get("matches", []))
        logger.warning(f"[DBG.2] Pinecone returned {match_count} matches for category '{primary_category}'")

        # --- Fallback Logic ---
        if match_count == 0:
            # 1. Fallback to 'past_conversation' if not already tried
            if primary_category != "past_conversation":
                logger.warning(f"[DBG.2] No results in '{primary_category}'. Trying fallback search in 'past_conversation'")
                query_params["filter"] = {"category": "past_conversation"}
                if fallback_system:
                    results = await asyncio.to_thread(fallback_system.query_with_fallback, query_params)
                else:
                    results = await asyncio.to_thread(lambda: index.query(**query_params))
                match_count = len(results.get("matches", []))
                logger.warning(f"[DBG.2] Fallback search (past_conversation) returned {match_count} matches.")

            # 2. Fallback to no category filter if still no results
            if match_count == 0:
                logger.warning(f"[DBG.2] No results after category fallbacks. Trying search without category filter.")
                query_params.pop("filter", None) # Remove the filter key
                if fallback_system:
                    results = await asyncio.to_thread(fallback_system.query_with_fallback, query_params)
                else:
                    results = await asyncio.to_thread(lambda: index.query(**query_params))
                match_count = len(results.get("matches", []))
                logger.warning(f"[DBG.2] Unfiltered search returned {match_count} matches.")

        # --- Process and Return Results ---
        if results.get("matches"):
            # Log the results (regardless of which search attempt found them)
            final_category_searched = query_params.get("filter", {}).get("category", "unfiltered")
            logger.warning(f"[DBG.2] Logging top 5 matches from final search (category='{final_category_searched}')")
            for idx, match in enumerate(results["matches"][:5]):
                match_id = match.get("id", "unknown")
                match_score = match.get("score", 0.0)
                match_category = match.get("metadata", {}).get("category", "unknown")
                match_text_preview = match.get("metadata", {}).get("text", "")[:100] + "..."
                logger.warning(f"[DBG.2] Match {idx}: id={match_id}, score={match_score:.4f}, category={match_category}")
                logger.warning(f"[DBG.2] Match {idx} text: {match_text_preview}")

            # Return the raw matches list including scores and metadata for the merger/reranker
            return results["matches"]

        # Return empty list if no matches found
        return []
    except Exception as e:
        logger.error("[DBG.2] ❌ Error retrieving memory: %s", e, exc_info=True)
        # Return empty list on error
        return []


async def load_all_conversations_async() -> List[str]:
    """Asynchronously loads all conversation texts from Pinecone."""
    try:
        dummy_query = "conversation"
        # get_embedding is synchronous, run in thread
        dummy_embedding = await asyncio.to_thread(get_embedding, dummy_query)
        # index.query is synchronous, run in thread
        results = await asyncio.to_thread(
            lambda: index.query(vector=dummy_embedding, top_k=10000, include_metadata=True, include_values=False, filter={"category": "conversation"}) # Increase top_k significantly
        )
        conversations = [match.get("metadata", {}).get("text", "No text found") for match in results.get("matches", [])]
        logger.info(f"Loaded {len(conversations)} past conversations from Pinecone.")
        return conversations
    except Exception as e:
        logger.error("❌ Error loading conversations asynchronously: %s", e, exc_info=True)
        return []

# ------------------------------
# Short-Term Memory Cache (FAISS + LRU)
# ------------------------------
class ShortTermMemoryCache:
    def __init__(self, pinecone_index: Any, cache_capacity: int = 30, embedding_dim: int = 1536, similarity_threshold: float = 0.82) -> None:
        self.lru_cache: OrderedDict[bytes, Any] = OrderedDict()
        self.embedding_dim = embedding_dim
        self.semantic_index = faiss.IndexFlatL2(embedding_dim)
        self.semantic_data: List[Any] = []
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.cache_capacity = cache_capacity
        self.similarity_threshold = similarity_threshold
        self.pinecone = pinecone_index

    def _format_result(self, result: Any) -> Any:
        return result

    def _update_lru(self, query_embedding: np.ndarray, result: Any) -> None:
        key = query_embedding.tobytes()
        if key in self.lru_cache:
            self.lru_cache.move_to_end(key)
        else:
            self.lru_cache[key] = result
            if len(self.lru_cache) > self.cache_capacity:
                self.lru_cache.popitem(last=False)

    def _update_semantic_cache(self, query_embedding: np.ndarray, result: Any, metadata: Dict[str, Any]) -> None:
        vec = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, self.embedding_dim)
        self.semantic_index.add(vec)
        self.semantic_data.append(result)
        idx = len(self.semantic_data) - 1
        self.metadata[idx] = metadata

    def _update_caches(self, query_embedding: np.ndarray, result: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._update_lru(query_embedding, result)
        self._update_semantic_cache(query_embedding, result, metadata if metadata else {})

    def _handle_empty_result(self) -> None:
        return None

    async def retrieve_async(self, query_embedding: np.ndarray) -> Optional[Any]:
        """Asynchronous version of retrieve."""
        key = query_embedding.tobytes()
        if key in self.lru_cache:
            return self._format_result(self.lru_cache[key])

        query_vec = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, self.embedding_dim)
        if self.semantic_index.ntotal > 0:
            distances, indices = self.semantic_index.search(query_vec, k=5)
            best_distance = distances[0][0]
            if best_distance < (1 - self.similarity_threshold):
                idx = indices[0][0]
                result = self.semantic_data[idx]
                self._update_lru(query_embedding, result)
                return self._format_result(result)

        try:
            pinecone_results = await asyncio.to_thread(
                lambda: self.pinecone.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True) # Ensure list for JSON
            )
        except Exception as e:
            logger.error("Pinecone query failed: %s", e)
            return self._handle_empty_result()

        if pinecone_results and pinecone_results.get("matches"):
            best_match = pinecone_results["matches"][0]
            self._update_caches(query_embedding, best_match, metadata=best_match.get("metadata", {}))
            return self._format_result(best_match)

        return self._handle_empty_result()

    # Synchronous retrieve method removed. Use retrieve_async.
# ------------------------------
# NovaMemory: Manage Long-Term Memories
# ------------------------------
class NovaMemory:
    def __init__(self, graph_store: Optional[Neo4JStorage]):
        """
        Initializes the NovaMemory system.

        Args:
            graph_store: An initialized Neo4jKGStore instance or None if connection failed.
        """
        self.graph_store = graph_store
        self.user_info: Dict[str, Any] = {}
        # self.initialize_nova_identity() # Moved to async initialization
        self.past_conversations: List[str] = [] # Initialize as empty list, loaded in async init
        # Add fallback_system property to avoid undefined reference
        self.fallback_system = None

    async def initialize_nova_identity_async(self) -> None:
        """Async method to initialize identity in Pinecone if needed."""
        try:
            # Check if identity exists
            identity_check = await retrieve_pinecone_memory_async("nova_identity", category="identity")
            if not identity_check:
                 await async_upsert_memory_to_pinecone("I am Nova AI, an evolving assistant.", None, category="identity") # Pass None for nova_response
                 logger.info("Initialized Nova identity in Pinecone.")

            # Check if purpose exists
            purpose_check = await retrieve_pinecone_memory_async("nova_purpose", category="identity")
            if not purpose_check:
                 await async_upsert_memory_to_pinecone("My mission is to support, learn, and grow.", None, category="identity") # Pass None for nova_response
                 logger.info("Initialized Nova purpose in Pinecone.")
        except Exception as e:
             logger.error("❌ Error during async identity initialization: %s", e, exc_info=True)
    
    async def async_extract_and_ingest_graph_data(
        self,
        user_query: Optional[str] = None, # Make optional
        nova_response: Optional[str] = None, # Make optional
        full_conversation_text: Optional[str] = None # Add optional full text
    ) -> None:
        """
        Extracts entities and relationships from conversation text using an LLM
        and ingests them into the Neo4j graph database.
        Can accept separate user/nova parts OR the full conversation text directly.

        Args:
            user_query: The user's part of the conversation (if not using full_conversation_text).
            nova_response: Nova's part of the conversation (if not using full_conversation_text).
            full_conversation_text: The complete text of the conversation to process.
        """
        if not self.graph_store:
            logger.warning("Graph store not available. Skipping graph ingestion.")
            return

        if full_conversation_text:
            conversation_text = full_conversation_text
        elif user_query and nova_response:
            conversation_text = f"User: {user_query}\nNova: {nova_response}"
        else:
            logger.error("❌ async_extract_and_ingest_graph_data called without sufficient text (need full_conversation_text or both user_query and nova_response).")
            return
        # Define the desired JSON structure for the LLM
        # Note: Using simpler IDs for this example. Production might need UUIDs.
        json_format = """
        {
          "entities": [
            {"id": "entity_name_1", "label": "EntityType", "name": "Entity Name 1", "properties": {"description": "Optional description"}},
            {"id": "entity_name_2", "label": "EntityType", "name": "Entity Name 2", "properties": {}}
          ],
          "relationships": [
            {"source_id": "entity_name_1", "target_id": "entity_name_2", "type": "RELATIONSHIP_TYPE", "properties": {"context": "Optional context"}}
          ]
        }
        """
        prompt = f"""
        Extract key entities (people, places, concepts, topics mentioned) and their relationships from the following conversation text.
        Generate unique, descriptive IDs for entities based on their name and type (e.g., 'person_john_doe', 'concept_ai').
        Provide the output strictly in the following JSON format:
        {json_format}

        Rules:
        - Assign a unique string ID to each entity (lowercase, use underscores).
        - Use appropriate labels (e.g., Person, Concept, Topic, Location, Organization).
        - Use meaningful relationship types (e.g., MENTIONS, DISCUSSES, RELATED_TO, LOCATED_IN, WORKS_FOR, PART_OF).
        - Keep properties concise. If no specific property, use an empty dict {{}}.
        - Only include relationships where both source and target entities are extracted.

        Conversation Text:
        ---
        {conversation_text}
        ---

        JSON Output:
        """
        extracted_data_json = "" # Initialize to handle potential errors before assignment
        try:
            logger.info("Attempting graph data extraction via LLM...")
            response = await asyncio.to_thread(
                lambda: openai.chat.completions.create(
                    model="gpt-4o", # Using gpt-4o as planned
                    messages=[{"role": "system", "content": "You are a data extraction assistant outputting JSON."},
                              {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"} # Request JSON output
                )
            )
            extracted_data_json = response.choices[0].message.content
            logger.debug("LLM extraction response: %s", extracted_data_json)

            # Parse the JSON response
            extracted_data = json.loads(extracted_data_json)
            nodes = extracted_data.get("entities", [])
            edges = extracted_data.get("relationships", [])

            # Validate extracted data structure (basic check)
            if not isinstance(nodes, list) or not isinstance(edges, list):
                 raise ValueError("LLM response did not contain valid 'entities' or 'relationships' lists.")

            # Prepare edges with correct keys if needed (assuming 'source_id', 'target_id', 'type')
            # Based on common Neo4j practices and potential LightRAG needs
            prepared_edges = []
            node_ids = {node['id'] for node in nodes if 'id' in node} # Set of valid node IDs from extraction
            for edge in edges:
                # Ensure source/target IDs exist in the extracted nodes for consistency
                if "source_id" in edge and "target_id" in edge and "type" in edge and \
                   edge["source_id"] in node_ids and edge["target_id"] in node_ids:
                     prepared_edges.append({
                         "start_node_id": edge["source_id"], # Assuming LightRAG uses start/end node id
                         "end_node_id": edge["target_id"],
                         "type": edge["type"].upper().replace(" ", "_"), # Standardize relationship type
                         "properties": edge.get("properties", {})
                     })
                else:
                     logger.warning("Skipping edge due to missing keys or invalid/missing node IDs: %s", edge)


            # --- Batch Upsert Logic ---
            cypher_statements = [] # List of tuples: (query, params)
            nodes_to_process = 0
            edges_to_process = 0

            # Prepare node MERGE statements
            for node_dict in nodes:
                node_id = node_dict.get("id")
                if node_id:
                    nodes_to_process += 1
                    label = node_dict.get("label", "Unknown")
                    # Use SET for properties, MERGE for node creation/matching
                    node_query = f"""
                    MERGE (n:base {{entity_id: $entity_id}})
                    ON CREATE SET n += $properties, n:`{label}`
                    ON MATCH SET n += $properties, n:`{label}`
                    """
                    node_props = {
                        "name": node_dict.get("name", node_id), # Use name or id as default name
                        **(node_dict.get("properties", {})) # Add other properties
                    }
                    # Ensure entity_id is not duplicated in properties if already the key
                    node_props.pop("entity_id", None)
                    node_props.pop("label", None) # Label is handled by SET n:`{label}`

                    cypher_statements.append((node_query, {"entity_id": node_id, "properties": node_props}))

            # Prepare edge MERGE statements
            for edge_dict in prepared_edges:
                 source_id = edge_dict.get("start_node_id")
                 target_id = edge_dict.get("end_node_id")
                 rel_type = edge_dict.get("type", "RELATED") # Already formatted
                 edge_props = edge_dict.get("properties", {})

                 if source_id and target_id:
                     edges_to_process += 1
                     # Use MERGE for relationship creation/matching
                     edge_query = f"""
                     MATCH (source:base {{entity_id: $source_id}})
                     MATCH (target:base {{entity_id: $target_id}})
                     MERGE (source)-[r:`{rel_type}`]->(target)
                     ON CREATE SET r = $properties
                     ON MATCH SET r += $properties
                     """
                     # Ensure properties are valid types
                     cypher_statements.append((edge_query, {"source_id": source_id, "target_id": target_id, "properties": edge_props}))

            logger.info(f"Prepared {nodes_to_process} node and {edges_to_process} edge upsert statements for batch transaction.")

            # Execute batch upsert in a single transaction
            if cypher_statements:
                async def _execute_batch_upsert(tx, statements: List[Tuple[str, Dict]]):
                    total_succeeded = 0
                    total_failed = 0
                    for query, params in statements:
                        try:
                            result = await tx.run(query, params)
                            await result.consume() # Consume results even if not explicitly used
                            total_succeeded += 1
                        except Exception as batch_exc:
                            # Log specific error during statement execution within the transaction
                            logger.error(f"❌ Error executing Cypher in batch: Query='{query}' | Params='{params}' | Error='{batch_exc}'")
                            total_failed += 1
                            # Decide whether to continue or raise to rollback the transaction
                            # For now, we log and continue to attempt other statements.
                    logger.info(f"Batch transaction summary: Succeeded={total_succeeded}, Failed={total_failed}")
                    if total_failed > 0:
                         # Optionally raise an error here if any statement fails to ensure transaction rollback
                         # raise Exception(f"{total_failed} statements failed in the batch transaction.")
                         pass # Currently allows partial success within the transaction

                try:
                    # Ensure driver exists before creating session
                    if not self.graph_store._driver:
                         logger.error("❌ Neo4j driver not initialized. Cannot execute transaction.")
                         raise ConnectionError("Neo4j driver not initialized.")
                    # Use the graph_store's driver and database info directly
                    async with self.graph_store._driver.session(database=self.graph_store._DATABASE) as session:
                         await session.execute_write(_execute_batch_upsert, cypher_statements)
                    logger.info(f"Graph ingestion batch transaction for this text completed.")
                except Exception as tx_exc:
                     # This catches errors during the transaction execution itself (e.g., connection issues)
                     logger.error(f"❌ Neo4j transaction failed for batch upsert: {tx_exc}", exc_info=True)
            else:
                 logger.info("No nodes or edges extracted from LLM response to ingest.")

        except json.JSONDecodeError as jde:
            logger.error("❌ Failed to parse JSON response from LLM for graph extraction: %s", jde)
            logger.error("LLM Response was: %s", extracted_data_json) # Log the problematic JSON
        except AttributeError as ae:
             # This error suggests self.graph_store is None or doesn't have expected methods
             logger.error("❌ Graph store method not found (add_nodes/add_edges?). Is Neo4j connected and LightRAG installed correctly? %s", ae, exc_info=True)
        except openai.APIError as oae:
             logger.error("❌ OpenAI API error during graph extraction: %s", oae)
        except Exception as e:
            # Catch other potential errors during the process
            logger.error("❌ Error during graph extraction and ingestion: %s", e, exc_info=True)

    async def chatgpt_4o_conversation_async(self, user_input: str) -> str:
        """
        Asynchronous version of chatgpt_4o_conversation.
        MODIFIED TO SHOW CHAIN OF THOUGHT
        """
        try:
            # Retrieve memories using the updated function (no ranker needed here)
            pinecone_memories_raw = await retrieve_pinecone_memory_async(user_input, category="conversation", fallback_system=self.fallback_system)
            # Extract text from raw results
            pinecone_memories = [res.get('metadata', {}).get('text', '') for res in pinecone_memories_raw if res.get('metadata', {}).get('text')]

            system_prompt = """You are Nova AI, a helpful assistant.

For every query, show your thinking process transparently (not in HTML comments). Structure your thinking as follows:

===== NOVA THINKING PROCESS =====
Step 1: Analyze the problem
[Your analysis here]

Step 2: Consider different approaches
[Different approaches here]

Step 3: Select the best approach
[Selected approach here]

Step 4: Generate an initial answer
[Initial answer here]

Step 5: Self-reflection
[Evaluate if your initial answer fully addresses the question]
[Identify potential errors, biases, or missing details]
[Improve the answer if necessary]
===== END THINKING PROCESS =====

[Your final refined answer here]"""

            if pinecone_memories:
                memory_text = "\n".join(f"- {m}" for m in pinecone_memories[:5]) # Limit context in fallback
                system_prompt += f"\nHere is what you remember:\n{memory_text}"

            response = await asyncio.to_thread(
                lambda: openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ]
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing conversation: {str(e)}"

# ------------------------------
# NovaAI: Main Application Class
# ------------------------------
class NovaAI:
    def __init__(self):
        # Initialize with empty properties that will be set during async initialization
        self.memory = None
        self.query_router = None
        self.hybrid_merger = None
        self.reranker = None
        self.fallback_system = None
        self.fused_results_cache = {}
        self.fused_results_cache_lock = threading.RLock()

    async def initialize(self):
        """Async initialization of NovaAI components"""
        print("Initializing NovaAI components...")
        
        # Initialize Neo4j connection for graph storage
        try:
            # Create a namespace (probably a string identifier)
            namespace = "nova_ai"
            
            # Create a config dictionary with connection details
            global_config = {
                "uri": os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
                "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password"),
                "database": "neo4j"  # Default Neo4j database name
            }
            
            # Define an embedding function (this is likely passed to Neo4JStorage to generate embeddings)
            # For compatibility with the get_embedding function
            async def embedding_func(text):
                if isinstance(text, list):
                    return await batch_get_embeddings(text)
                else:
                    return get_embedding(text)
            
            # Create the Neo4JStorage instance with correct parameters
            graph_store = Neo4JStorage(namespace, global_config, embedding_func)
            print("✅ Neo4j connection established")
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            graph_store = None
        
        # Initialize memory system
        self.memory = NovaMemory(graph_store)
        await self.memory.initialize_nova_identity_async()
        
        # Initialize fallback system
        self.fallback_system = FallbackSystem(index)
        # Make sure memory has a reference to the fallback system
        self.memory.fallback_system = self.fallback_system
        
        # Initialize components for hybrid retrieval
        # Create a wrapper for QueryRouter that adds the 'determine_routing' method
        class QueryRouterWrapper:
            def __init__(self):
                self._router = QueryRouter()
            
            def determine_routing(self, query_text):
                """Wrapper for the route method that the code expects."""
                return self._router.route(query_text)
        
        self.query_router = QueryRouterWrapper()
        self.hybrid_merger = HybridMerger()
        
        # Initialize reranker
        try:
            self.reranker = CrossEncoderReranker()
            print("✅ Cross-encoder reranker object created. Attempting to load model...")
            # Explicitly load the model after successful instantiation
            model_loaded = await self.reranker.load_model()
            if model_loaded:
                print("✅ Cross-encoder reranker model loaded successfully.")
            else:
                print("❌ Failed to load cross-encoder reranker model. Reranking disabled.")
                self.reranker = None # Disable reranker if model loading fails
        except Exception as e:
            print(f"❌ Error initializing or loading reranker: {e}")
            self.reranker = None
            
        print("NovaAI initialization complete")

    async def chatbot_response_async(self, user_input: str) -> str:
        """
        Process user input asynchronously and generate a response using the hybrid retrieval system.
        """
        user_input_lower = user_input.lower()
        logger.warning(f"--- Starting chatbot_response_async for input: '{user_input[:100]}...' ---")
        
        # Step 1: Query Routing
        logger.warning("Step 1: Determining query routing...")
        routing_mode = self.query_router.determine_routing(user_input)
        logger.warning(f"[DBG.2.1] Routing decision: {routing_mode.name}") # Added DBG log
        logger.warning(f"Step 1a: Query routing determined: {routing_mode.name}")
        
        # Step 2: Cache Check
        logger.warning("Step 2: Checking fused results cache...")
        cache_key = md5(user_input.encode()).hexdigest()
        with self.fused_results_cache_lock:
            if cache_key in self.fused_results_cache:
                logger.warning(f"Step 2a: Cache hit! Using cached results for query.")
                merged_results = self.fused_results_cache[cache_key]
                logger.info(f"Using {len(merged_results)} cached merged results for key: {cache_key}")
            else:
                logger.warning(f"Step 2b: Cache miss. Will perform retrieval.")
                
                # Step 3: Prepare Retrievals
                logger.warning("Step 3: Preparing retrievals based on routing mode...")
                tasks = []
                vector_task_index = -1
                graph_task_index = -1
                
                # Always add vector retrieval
                vector_task_index = len(tasks)
                vector_task = retrieve_pinecone_memory_async(
                    user_input, 
                    category="conversation", 
                    fallback_system=self.fallback_system
                )
                tasks.append(vector_task)
                logger.warning(f"Step 3a: Added vector retrieval task (index {vector_task_index}).")
                
                # Add graph retrieval based on routing
                if routing_mode != RoutingMode.VECTOR and self.memory.graph_store:
                    graph_task_index = len(tasks)
                    # Use get_knowledge_graph with node_label parameter
                    # Using 'base' as the node label since that's what's used in async_extract_and_ingest_graph_data
                    graph_task = self.memory.graph_store.get_knowledge_graph(node_label="base")
                    tasks.append(graph_task)
                    logger.warning(f"Step 3b: Added graph retrieval task (index {graph_task_index}).")
                else:
                    logger.warning(f"Step 3b: Skipping graph retrieval - mode={routing_mode.name}, graph_store_available={self.memory.graph_store is not None}")
                
                # Step 4: Execute Retrievals
                logger.warning(f"Step 4: Executing {len(tasks)} retrieval tasks...")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                logger.warning(f"Step 4a: All retrieval tasks completed.")
                
                # Process vector results
                vector_results_raw = []
                if vector_task_index != -1:
                    vector_result_or_exc = results[vector_task_index]
                    if isinstance(vector_result_or_exc, list):
                        vector_results_raw = vector_result_or_exc
                        logger.warning(f"[DBG.2.1] Raw vector results count: {len(vector_results_raw)}") # Added DBG log
                        logger.warning(f"Step 4b: Vector retrieval successful ({len(vector_results_raw)} results).")
                        logger.info(f"Vector retrieval successful ({len(vector_results_raw)} results).")
                    elif isinstance(vector_result_or_exc, Exception):
                        logger.error("❌ Error retrieving from vector store: %s", vector_result_or_exc)
                    else:
                        logger.warning(f"Unexpected result type from vector task: {type(vector_result_or_exc)}")
                
                # Process graph results  
                graph_results_raw = []
                if graph_task_index != -1:
                    graph_result_or_exc = results[graph_task_index]
                    if isinstance(graph_result_or_exc, list):
                        graph_results_raw = graph_result_or_exc
                        logger.warning(f"[DBG.2.1] Raw graph results count: {len(graph_results_raw)}") # Added DBG log
                        logger.warning(f"Step 4c: Graph retrieval successful ({len(graph_results_raw)} results).")
                        logger.info(f"Graph retrieval successful ({len(graph_results_raw)} results).")
                    elif isinstance(graph_result_or_exc, Exception):
                        logger.error("❌ Error retrieving from graph store: %s", graph_result_or_exc)
                    else:
                        logger.warning(f"Unexpected result type from graph task: {type(graph_result_or_exc)}")
                
                # Step 5: Hybrid Merging
                logger.warning("Step 5: Starting hybrid merging...")
                merged_results = self.hybrid_merger.merge_results(
                    vector_results=vector_results_raw,
                    graph_results=graph_results_raw
                )
                logger.warning(f"Step 5a: Merging complete ({len(merged_results)} results).")
                
                # Cache the merged results
                with self.fused_results_cache_lock:
                    self.fused_results_cache[cache_key] = merged_results
                    logger.warning(f"Step 5b: Stored {len(merged_results)} merged results in cache.")
                    logger.info(f"Stored {len(merged_results)} merged results in cache for key: {cache_key}")
        
        # Step 6: Reranking
        logger.warning("Step 6: Starting reranking...")
        if self.reranker and merged_results:
            logger.warning(f"Step 6a: Reranker available. Reranking {len(merged_results)} merged results...")
            # Call the async rerank method directly
            reranked_results = await self.reranker.rerank(query=user_input, results=merged_results, top_n=15)
            logger.warning(f"Step 6b: Reranking complete. Top results count: {len(reranked_results)}")
        else:
            reranked_results = merged_results[:15]
            if not self.reranker:
                logger.warning("Step 6a: Reranker not available, using top merged results directly.")
            elif not merged_results:
                logger.warning("Step 6a: No merged results to rerank.")
            logger.warning(f"Step 6b: Reranking skipped/complete. Using {len(reranked_results)} results.")

        # Extract text from the (potentially reranked) results for context
        combined_memories = [res.get('text', '') for res in reranked_results if res.get('text')]
        logger.warning(f"Step 7: Extracted text from {len(combined_memories)} final results for context.")
        logger.info(f"Final context memories count: {len(combined_memories)}.")

        # Step 8: Context Formulation & LLM Call
        if combined_memories and not any(x in user_input_lower for x in ["search online", "breaking news"]):
            # Limit context size passed to LLM
            MAX_CONTEXT_TOKENS = 3000 # Example limit, adjust as needed
            memory_context = ""
            current_tokens = 0
            # Simple token estimation (split by space) - replace with proper tokenizer if needed
            for mem in combined_memories:
                mem_tokens = len(mem.split())
                if current_tokens + mem_tokens <= MAX_CONTEXT_TOKENS:
                    memory_context += mem + "\n"
                    current_tokens += mem_tokens
                else:
                    break # Stop adding memories if token limit is reached

            if not memory_context.strip(): # Handle case where even first memory is too long
                memory_context = "Could not retrieve relevant memories within token limits."

            # Memory context formulation
            memory_context = "Based on my retrieved memories:\n" + memory_context.strip()
            combined_query = memory_context + "\n\nAnswer the following question: " + user_input
            logger.warning("Step 8: Formulating final context for LLM...")
            logger.warning("Step 8b: Preparing LLM call with context...")
            
            try:
                response_obj = await asyncio.to_thread(
                    lambda: openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": """You are Nova AI, a helpful assistant.

For every query, show your thinking process transparently (not in HTML comments). Structure your thinking as follows:

===== NOVA THINKING PROCESS =====
Step 1: Analyze the problem
[Your analysis here]

Step 2: Consider different approaches
[Different approaches here]

Step 3: Select the best approach
[Selected approach here]

Step 4: Generate an initial answer
[Initial answer here]

Step 5: Self-reflection
[Evaluate if your initial answer fully addresses the question]
[Identify potential errors, biases, or missing details]
[Improve the answer if necessary]
===== END THINKING PROCESS =====

[Your final refined answer here]"""},
                            {"role": "user", "content": combined_query}
                        ]
                    )
                )
                answer = response_obj.choices[0].message.content
                logger.warning("Step 8c: Received response from LLM based on retrieved context.")
            except Exception as e:
                logger.error("Error in combined memory + GPT-4o conversation: %s", e)
                logger.warning("Step 8d: Preparing fallback LLM call...")
                answer = await self.memory.chatgpt_4o_conversation_async(user_input)
                logger.warning("Step 8e: Fallback LLM call completed.")
        else:
            # Fallback if memory retrieval fails, context is empty, or user asked for online search
            logger.warning("Step 8f: Preparing direct LLM call (no context)...")
            answer = await self.memory.chatgpt_4o_conversation_async(user_input)
            logger.warning("Step 8g: Direct LLM call completed.")

        # Extract just the final answer for speech
        final_answer = answer
        if "===== END THINKING PROCESS =====" in answer:
            final_answer = answer.split("===== END THINKING PROCESS =====")[1].strip()

        # Save the final answer part of the conversation, triggering graph ingestion
        try:
            # Await the async save function for the final turn
            await async_upsert_memory_to_pinecone(user_query=user_input, nova_response=final_answer)
            logger.warning("Step 9: Final conversation turn saved.")
            
            # Trigger graph ingestion if graph store is available
            if self.memory.graph_store:
                try:
                    await self.memory.async_extract_and_ingest_graph_data(
                        user_query=user_input, 
                        nova_response=final_answer
                    )
                    logger.warning("Step 10: Graph ingestion triggered.")
                except Exception as graph_err:
                    logger.error("❌ Error during graph ingestion: %s", graph_err, exc_info=True)
        except Exception as mem_save_err:
            logger.error("❌ Error during final memory save: %s", mem_save_err, exc_info=True)
            
        logger.warning(f"--- Completed chatbot_response_async for input: '{user_input[:100]}...' ---")
        return answer

# ------------------------------
# GUI Implementation (Tkinter)
# ------------------------------
class NovaGUI:
    def __init__(self, root: tk.Tk, nova_ai: NovaAI):
        self.nova = nova_ai
        self.root = root
        self.root.title("Nova AI Assistant")
        self.chat_box = tkinter.scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
        self.chat_box.pack(pady=10)
        self.chat_box.insert(tk.END, "Nova: Hello! How can I assist you?\n")
        self.entry = tk.Entry(root, width=50)
        self.entry.pack(pady=5)
        self.entry.bind("<Return>", self.send_message)
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack()

    def send_message(self, event: Optional[Any] = None) -> None:
        user_input = self.entry.get()
        if user_input:
            self.chat_box.insert(tk.END, f"You: {user_input}\n")
            # Use thread_pool for non-blocking UI
            future = thread_pool.submit(self.nova.chatbot_response_async, user_input)

            def handle_response(future):
                try:
                    response = future.result()
                    self.chat_box.insert(tk.END, f"Nova: {response}\n\n")
                except Exception as e:
                    self.chat_box.insert(tk.END, f"Nova: Error: {str(e)}\n\n")

            # Add a callback to update UI when response is ready
            future.add_done_callback(handle_response)
            self.entry.delete(0, tk.END)

# ------------------------------
# Import Past Conversations from Files
# ------------------------------
def load_saved_conversations(folder_path: str) -> List[Tuple[str, str]]: # Return tuples
    """Loads conversations, returning a list of (filename, content) tuples."""
    from glob import glob
    # Construct absolute path for glob
    abs_folder_path = os.path.abspath(folder_path)
    logger.info(f"Looking for conversation files in: {abs_folder_path}")
    file_paths = sorted(glob(os.path.join(abs_folder_path, "*.txt")))
    conversations_data = []
    if not file_paths:
        logger.warning(f"No .txt files found in {abs_folder_path}")
        return []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip(): # Only add if content is not empty
                    conversations_data.append((filename, content))
                else:
                    logger.warning(f"Skipping empty file: {filename}")
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
    logger.info(f"Loaded {len(conversations_data)} non-empty conversations.")
    return conversations_data

async def upload_past_conversations_async(folder_path: str) -> None: # Removed default value
    """
    Loads conversations from text files in the specified folder and upserts them
    into Pinecone asynchronously in batches. Includes filename in metadata.
    """
    # Load conversations returns list of (filename, content) tuples now
    conversations_with_filenames = load_saved_conversations(folder_path)

    # Process conversations in batches to avoid overloading
    batch_size = 5
    for i in range(0, len(conversations_with_filenames), batch_size): # Use the correct variable name
        batch = conversations_with_filenames[i:i+batch_size] # Use the correct variable name
        tasks = []
        for j, (filename, convo_text) in enumerate(batch, start=i+1):
            # Use filename as part of the user_query field for non-conversation upsert
            # Or pass it as the new filename argument
            task = async_upsert_memory_to_pinecone(
                user_query=convo_text, # Pass full text as user_query for 'past_conversation' category
                nova_response=None,     # Set response to None as it's not a live turn
                category="past_conversation", # Use a specific category
                filename=filename       # Pass the original filename
            )
            tasks.append(task)
        # Wait for the batch to complete before starting the next one
        await asyncio.gather(*tasks)

    print("✅ Past conversations successfully stored in Nova AI's memory.")

# Synchronous wrapper upload_past_conversations removed. Callers should await async version.
async def ingest_past_conversations_to_graph_async(folder_path: str, nova_memory: NovaMemory):
    """
    Loads conversations from text files and ingests their graph data (entities/relationships)
    into Neo4j asynchronously in batches. Requires an initialized NovaMemory instance.
    """
    if not nova_memory.graph_store:
        logger.error("❌ Cannot ingest graph data: Neo4j graph store is not initialized in NovaMemory.")
        return

    logger.info(f"Starting graph data ingestion from folder: {folder_path}")
    conversations_with_filenames = load_saved_conversations(folder_path) # Assumes this returns [(filename, content), ...]

    if not conversations_with_filenames:
        logger.warning(f"No conversation files found or loaded from {folder_path}. Skipping graph ingestion.")
        return

    batch_size = 5 # Process in smaller batches due to potential LLM rate limits/costs
    processed_files = 0
    failed_files = 0

    # Process files sequentially instead of using gather (Corrected Indentation)
    for idx, (filename, convo_text) in enumerate(conversations_with_filenames):
        logger.info(f"Processing file {idx+1}/{len(conversations_with_filenames)}: {filename} for graph ingestion...")
        try:
            # Directly await the processing for each file
            await nova_memory.async_extract_and_ingest_graph_data(full_conversation_text=convo_text)
            processed_files += 1
        except Exception as e:
            logger.error(f"❌ Error processing graph data for file {filename}: {e}", exc_info=True)
            failed_files += 1


    logger.info(f"✅ Graph data ingestion complete. Processed: {processed_files}, Failed: {failed_files}")

# Synchronous wrapper ingest_past_conversations_to_graph removed. Callers should await async version.

# ------------------------------
# Main Execution
# ------------------------------
async def main_cli(nova_ai: NovaAI):
    """Asynchronous main function for the CLI."""
    print("Nova: Hello! How can I assist you? (Type 'exit' to quit)")
    while True:
        try:
            # Use asyncio.to_thread for non-blocking input
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.lower() == 'exit':
                break
            if user_input:
                # Call the async response function directly
                response = await nova_ai.chatbot_response_async(user_input)
                print(f"Nova: {response}\n") # Print the response
        except EOFError: # Handle Ctrl+D
            break
        except KeyboardInterrupt: # Handle Ctrl+C
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            logger.error("❌ Error in async CLI loop: %s", e, exc_info=True) # Log error
    print("\nNova: Goodbye!")

async def main():
    """Main async entry point to handle argument parsing and execution."""
    print("--- Entered async main ---") # Debugging

    # --- Argument Parsing ---
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Run Nova AI")
    parser.add_argument('--cli', action='store_true', help='Run in command-line interface mode instead of GUI.')
    parser.add_argument('--upload', action='store_true', help='Upload past conversations to Pinecone.')
    parser.add_argument('--ingest_graph', action='store_true', help='Load conversations and ingest graph data into Neo4j.')
    parser.add_argument('--conversations_dir', type=str, default="../Conversations", help='Directory containing conversation files.')
    args = parser.parse_args()

    # --- Initialize NovaAI *once* ---
    # Note: Initialization involves potentially blocking I/O (Neo4j check),
    # but we run it within the main async context started by asyncio.run(main())
    print("Instantiating Nova AI (sync part)...")
    nova_ai_instance = NovaAI() # Sync part of init
    print("Nova AI Instantiated. Starting async initialization...")
    await nova_ai_instance.initialize() # Async part of init
    print("Nova AI Initialized.")

    # --- Execute based on args ---
    if args.upload:
        print(f"Uploading past conversations from {args.conversations_dir} to Pinecone (async)...")
        # Directly await the async function
        await upload_past_conversations_async(folder_path=args.conversations_dir)
        print("Pinecone upload complete.")

    elif args.ingest_graph:
        print(f"Starting graph ingestion from {args.conversations_dir}...")
        if nova_ai_instance.memory.graph_store:
            try:
                # Directly await the async function
                await ingest_past_conversations_to_graph_async(folder_path=args.conversations_dir, nova_memory=nova_ai_instance.memory)
                print("Graph ingestion process complete.")
            except Exception as e:
                logger.error(f"❌ Top-level error during graph ingestion: {e}", exc_info=True)
                print(f"An error occurred during graph ingestion: {e}")
        else:
            print("Graph store not available. Cannot ingest graph data.")

    elif args.cli:
        print("Starting Nova AI in CLI mode...")
        # Directly await the async CLI function, passing the single instance
        try:
            await main_cli(nova_ai_instance)
        except KeyboardInterrupt:
            print("\nExiting...")

    else:
        # Default to GUI mode if no relevant args
        print("Starting Nova AI in GUI mode...")
        try:
            # GUI runs synchronously in the main thread after async setup
            import tkinter as tk
            import tkinter.scrolledtext # Keep imports
            # GUI interaction needs significant refactoring to work with async NovaAI methods.
            # The previous sync wrappers are removed.
            # For now, GUI mode will likely fail or hang when trying to call nova_ai methods.
            print("WARNING: GUI mode is likely non-functional due to async refactoring.")
            print("Please use --cli mode.")
            root = tk.Tk() # Keep the root window creation for now, even if GUI is broken
            # gui = NovaGUI(root, nova_ai_instance) # This would need changes in NovaGUI
            root.mainloop() # This blocks until GUI is closed
        except ImportError:
            print("Tkinter not found. Cannot start GUI. Try running with --cli flag.")
        except Exception as e:
            print(f"Failed to start GUI: {e}. Try running with --cli flag.")

if __name__ == "__main__":
    print("--- Entered __main__ block ---") # ADDED FOR DEBUGGING
    # Redirect logic remains commented out
    print("--- Skipped stdout redirection ---") # ADDED FOR DEBUGGING

    # Get or create an event loop and run the main async function
    try:
        # Attempt to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("Event loop is already running. Creating task...")
            # If the loop is already running, schedule main() as a task.
            # This might happen in certain environments or if integrated differently.
            # Note: This specific path might need more complex handling depending on the context.
            # For a simple script execution, loop.run_until_complete is usually sufficient.
            # We'll prioritize the run_until_complete path first.
            # For now, log a warning if we hit this unusual case.
            logger.warning("Detected an already running event loop. Scheduling main() as a task.")
            task = loop.create_task(main())
            # How to wait for this task synchronously is complex, might indicate deeper issues.
            # Reverting to run_until_complete which is more standard for script entry points.
            # Let's refine the logic: get_event_loop() then run_until_complete is standard.
            loop.run_until_complete(main()) # Try running directly on the potentially existing loop
        else:
            # If no loop is running, use run_until_complete to run main()
            print("No running event loop detected. Running main() with loop.run_until_complete...")
            loop.run_until_complete(main())

    except RuntimeError as e:
         # This might catch errors if get_event_loop() fails in specific contexts
         print(f"\nRuntimeError managing event loop: {e}")
         logger.error(f"❌ RuntimeError managing event loop: {e}", exc_info=True)
    except KeyboardInterrupt:
         print("\nKeyboard interrupt received. Exiting.")
    except Exception as main_exc:
         print(f"\nAn unexpected error occurred in main execution: {main_exc}")
         logger.error(f"❌ Unexpected error in main execution: {main_exc}", exc_info=True)
    finally:
        # Optional: Close the loop if we created it and it's not closed
        # This part can be tricky; asyncio.run handles this automatically.
        # loop.close() # Be cautious with closing loops you didn't explicitly create.
        pass