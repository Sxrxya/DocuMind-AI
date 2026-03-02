"""
DocuMind-AI — Application Configuration

Loads settings from .env via pydantic-settings.
All RAG parameters (chunk size, top-k, model names) are centralised here.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = two levels up from this file (backend/core/config.py → DocuMind-AI/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Centralised application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: str = "groq"  # "groq" | "ollama"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k: int = 5

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Paths ---
    data_dir: str = str(PROJECT_ROOT / "data")
    models_dir: str = str(PROJECT_ROOT / "models")
    vectorstore_dir: str = str(PROJECT_ROOT / "vectorstore")


# Singleton — import this from anywhere
settings = Settings()
