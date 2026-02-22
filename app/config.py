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
    EMBEDDING_MODEL: str = "text-embedding-ada-002"

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
