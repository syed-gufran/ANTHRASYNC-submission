"""Central configuration, loaded from environment variables (.env).

Keeping every tunable in one typed object means the rest of the codebase never
reads ``os.environ`` directly — it just imports ``settings``. This makes the
system easy to reconfigure (model, chunk size, top-k) without touching code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = one level above this file's package directory.
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Credentials ---
    openai_api_key: str = ""

    # --- Models (OpenAI for both generation and embeddings) ---
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0

    # --- Paths ---
    data_dir: str = "data"
    vector_dir: str = ".faiss"

    # --- Chunking / retrieval ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    @property
    def data_path(self) -> Path:
        return ROOT_DIR / self.data_dir

    @property
    def vector_path(self) -> Path:
        return ROOT_DIR / self.vector_dir


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read .env only once)."""
    return Settings()


settings = get_settings()
