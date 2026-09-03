"""Central configuration for AttentionOS."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    database_url: str = "sqlite:///./attention_os.db"
    database_url_fallback: str = "sqlite:///./attention_os.db"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    environment: str = "development"
    log_level: str = "INFO"

    model_config = {"env_file": str(BASE_DIR / ".env"), "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
