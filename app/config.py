import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    # OpenAI Configuration
    OPENAI_API_KEY: str

    # Pinecone Configuration
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    PINECONE_INDEX: str = "nova-ai-memory" # Default index name from Nova_AI.py

    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687" # Default local URI
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j" # Default database

    # Reranker Configuration (Optional)
    RERANKER_MODEL_NAME: Optional[str] = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    # Embedding Model Configuration (Optional - for future flexibility)
    EMBEDDING_MODEL: str = "text-embedding-ada-002"

    class Config:
        # Load environment variables from a .env file if it exists
        env_file = ".env"
        env_file_encoding = 'utf-8'
        # Allow extra fields if needed, though we define all expected ones
        extra = 'ignore'

# Instantiate settings globally for easy import
# This will automatically load variables upon import of this module
try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {e}. Ensure all required environment variables or .env file entries are set.")
    # Depending on the desired behavior, you might exit or use default/dummy values
    # For now, let it raise the error during import if critical variables are missing.
    raise