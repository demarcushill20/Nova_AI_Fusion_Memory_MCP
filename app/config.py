from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None

    # Pinecone Configuration
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    PINECONE_INDEX: str = "nova-ai-memory"  # Default index name from Nova_AI.py

    # Neo4j Configuration
    NEO4J_URI: str = "bolt://neo4j:7687"  # Default Docker Compose service URI
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: Optional[str] = None  # Optional when Neo4j auth is disabled
    NEO4J_DATABASE: str = "neo4j"  # Default database

    # Reranker Configuration (Optional)
    RERANKER_MODEL_NAME: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Embedding Model Configuration (Optional - for future flexibility)
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Chronology Configuration (Phase 1)
    EVENT_SEQ_FILE: str = "/data/event_seq.counter"  # Monotonic sequence counter file

    # Redis Configuration (Phase 6 — optional, enables O(log N) timeline queries)
    REDIS_URL: Optional[str] = "redis://redis_db:6379/0"  # Docker Compose alias
    REDIS_ENABLED: bool = True  # Set False to use file-based fallback

    # Temporal Decay Scoring (Phase P9A.2)
    TEMPORAL_DECAY_ENABLED: bool = True  # Set False to skip temporal scoring
    TEMPORAL_DECAY_HALF_LIVES: Optional[str] = None  # JSON override, e.g. '{"debug": 3}'
    TEMPORAL_WEIGHT: float = 0.30  # Weight of temporal signal in composite score

    # MMR Deduplication (Phase P9A.3)
    MMR_ENABLED: bool = True  # Set False to skip MMR deduplication
    MMR_LAMBDA: float = 0.7  # 1.0 = pure relevance, 0.0 = pure diversity

    # Write-Time Deduplication (Phase P9A.5)
    WRITE_DEDUP_ENABLED: bool = True  # Set False to skip write-time semantic dedup
    WRITE_DEDUP_THRESHOLD: float = 0.92  # Cosine similarity threshold for duplicate
    CONFLICT_DETECTION_ENABLED: bool = True  # Detect conflicting decisions on upsert

    # Query Router Configuration (Phase P9A.4)
    QUERY_ROUTER_LLM_ENABLED: bool = False  # Set True to enable LLM-based query classification

    @model_validator(mode="before")
    @classmethod
    def _normalize_mixed_case_env_keys(cls, data):
        """
        Handles environments where keys can appear lowercased (e.g. openai_api_key).
        """
        if not isinstance(data, dict):
            return data

        key_aliases = (
            ("OPENAI_API_KEY", "openai_api_key"),
            ("PINECONE_API_KEY", "pinecone_api_key"),
            ("PINECONE_ENV", "pinecone_env"),
            ("NEO4J_URI", "neo4j_uri"),
            ("NEO4J_USER", "neo4j_user"),
            ("NEO4J_PASSWORD", "neo4j_password"),
            ("NEO4J_DATABASE", "neo4j_database"),
            ("PINECONE_INDEX", "pinecone_index"),
            ("RERANKER_MODEL_NAME", "reranker_model_name"),
            ("EMBEDDING_MODEL", "embedding_model"),
            ("REDIS_URL", "redis_url"),
            ("REDIS_ENABLED", "redis_enabled"),
            ("TEMPORAL_DECAY_ENABLED", "temporal_decay_enabled"),
            ("TEMPORAL_DECAY_HALF_LIVES", "temporal_decay_half_lives"),
            ("TEMPORAL_WEIGHT", "temporal_weight"),
            ("MMR_ENABLED", "mmr_enabled"),
            ("MMR_LAMBDA", "mmr_lambda"),
            ("WRITE_DEDUP_ENABLED", "write_dedup_enabled"),
            ("WRITE_DEDUP_THRESHOLD", "write_dedup_threshold"),
            ("CONFLICT_DETECTION_ENABLED", "conflict_detection_enabled"),
            ("QUERY_ROUTER_LLM_ENABLED", "query_router_llm_enabled"),
        )

        for canonical, lowercase in key_aliases:
            if canonical not in data and lowercase in data:
                data[canonical] = data[lowercase]

        return data


try:
    settings = Settings()
except Exception as e:
    print(
        f"Error loading settings: {e}. Ensure all required environment variables or .env file entries are set."
    )
    raise
