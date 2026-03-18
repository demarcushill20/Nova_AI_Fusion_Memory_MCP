"""Lightweight entity extraction for graph-augmented retrieval (P9A.6).

Extracts entity mentions from query text using regex patterns.
Used to find starting nodes for multi-hop graph traversal.
No LLM dependency — pure regex for known project entities + basic NER patterns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity mention found in query text."""
    name: str
    entity_type: str  # "project", "technology", "concept", "person", "unknown"
    confidence: float = 1.0  # 1.0 for exact match, lower for fuzzy


# Known project entities (expandable)
KNOWN_ENTITIES: dict[str, str] = {
    # Projects
    "novacore": "project",
    "nova-core": "project",
    "nova core": "project",
    "novatrade": "project",
    "nova-trade": "project",
    "nova trade": "project",
    "fusion memory": "project",
    "nova-memory": "project",
    # Technologies
    "pinecone": "technology",
    "neo4j": "technology",
    "redis": "technology",
    "langfuse": "technology",
    "telegram": "technology",
    "python": "technology",
    "fastapi": "technology",
    "docker": "technology",
    "nginx": "technology",
    "metatrader": "technology",
    "mt5": "technology",
    "metaapi": "technology",
    "tradingview": "technology",
    # Concepts
    "circuit breaker": "concept",
    "dead man switch": "concept",
    "heartbeat": "concept",
    "reranker": "concept",
    "embedding": "concept",
    "checkpoint": "concept",
    "webhook": "concept",
}

# Regex patterns for generic entity extraction
_CAPITALIZED_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_TECH_PATTERN = re.compile(
    r"\b(API|MCP|SDK|CLI|VPS|LLM|NER|RRF|MMR|NDCG|FTMO|MT5)\b"
)


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract entity mentions from query text.

    Strategy:
    1. Match against known entity dictionary (high confidence)
    2. Find capitalized multi-word phrases (medium confidence)
    3. Find known tech acronyms (high confidence)

    Args:
        text: The query text to extract entities from.

    Returns:
        List of extracted entities, sorted by confidence descending.
    """
    if not text:
        return []

    entities: list[ExtractedEntity] = []
    seen_names: set[str] = set()
    text_lower = text.lower()

    # 1. Known entities (exact match, case-insensitive)
    for name, etype in KNOWN_ENTITIES.items():
        if name in text_lower and name not in seen_names:
            entities.append(ExtractedEntity(name=name, entity_type=etype, confidence=1.0))
            seen_names.add(name)

    # 2. Capitalized multi-word phrases (potential proper nouns / project names)
    for match in _CAPITALIZED_PHRASE.finditer(text):
        phrase = match.group(1)
        phrase_lower = phrase.lower()
        if phrase_lower not in seen_names:
            entities.append(
                ExtractedEntity(name=phrase, entity_type="unknown", confidence=0.6)
            )
            seen_names.add(phrase_lower)

    # 3. Tech acronyms
    for match in _TECH_PATTERN.finditer(text):
        acronym = match.group(1)
        acronym_lower = acronym.lower()
        if acronym_lower not in seen_names:
            entities.append(
                ExtractedEntity(name=acronym, entity_type="technology", confidence=0.8)
            )
            seen_names.add(acronym_lower)

    # Sort by confidence descending
    entities.sort(key=lambda e: e.confidence, reverse=True)

    if entities:
        logger.debug(
            "Extracted %d entities from query: %s",
            len(entities),
            [(e.name, e.entity_type, e.confidence) for e in entities],
        )

    return entities


def extract_entity_names(text: str) -> list[str]:
    """Convenience: extract just entity names from text.

    Args:
        text: The query text.

    Returns:
        List of entity name strings.
    """
    return [e.name for e in extract_entities(text)]
