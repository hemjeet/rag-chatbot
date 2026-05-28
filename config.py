"""
Centralized configuration for the RAG system.
All model registries, default settings, and environment-based config live here.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


# ──────────────────────────────────────────────
# Provider & Model Registries
# ──────────────────────────────────────────────

LLM_PROVIDERS = {
    "openai": {
        "display_name": "OpenAI",
        "models": {
            "gpt-5": "GPT-5",
            "gpt-4": "GPT-4",
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini",
        },
        "base_url": None,  # Uses default OpenAI URL
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "display_name": "DeepSeek",
        "models": {
            "deepseek-chat": "DeepSeek Chat",
            "deepseek-reasoner": "DeepSeek Reasoner",
            "deepseek-v4-flash": "DeepSeek V4 Flash",
        },
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
    },
}

EMBEDDING_MODELS = {
    "text-embedding-3-small": "Small (fastest, lower quality)",
    "text-embedding-3-large": "Large (slower, higher quality)",
    "text-embedding-ada-002": "Ada-002 (balanced)",
}

SUPPORTED_FILE_EXTENSIONS = {
    ".txt", ".pdf", ".docx", ".md", ".csv", ".html", ".htm",
}


# ──────────────────────────────────────────────
# Application Settings (loaded from .env)
# ──────────────────────────────────────────────

class Settings(BaseSettings):
    """Application settings with .env file support."""

    # API Keys
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    nvidia_api_key: str = ""

    # Model defaults
    default_llm_provider: str = "openai"
    default_model: str = "gpt-4o"
    default_embedding_model: str = "text-embedding-3-small"
    default_temperature: float = 0.0

    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 5
    bm25_weight: float = 0.4
    faiss_weight: float = 0.6
    memory_window: int = 10  # Number of exchanges to keep in memory

    # Paths
    vector_store_path: str = "faiss_index"
    upload_dir: str = "uploads"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def get_api_key(self, provider: str) -> str:
        """Get the API key for a given provider."""
        if provider == "openai":
            return self.openai_api_key
        elif provider == "deepseek":
            return self.deepseek_api_key
        return ""

    def ensure_dirs(self):
        """Create necessary directories."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
