#!/usr/bin/env python3
"""Centralized application settings using pydantic-settings."""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Dict, List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    # Core model & API providers
    MODEL_ID: str = Field(default="mistralai/ministral-8b")
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    JINA_API_KEY: str | None = None
    TENCENT_APPID: str | None = None
    TENCENT_SECRET_ID: str | None = None
    TENCENT_SECRET_KEY: str | None = None
    # System paths and tools
    FFMPEG_PATH: str = Field(default="ffmpeg")
    # Upload policy
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_EXTS: List[str] = Field(default_factory=lambda: ["mp4","avi","mov","mkv","webm"])
    # Progress weights
    PROGRESS_WEIGHTS: Dict[str, float] = Field(default_factory=lambda: {
        "extract_audio":0.10,
        "asr":0.20,
        "merge_text":0.20,
        "summary":0.20,
        "multimodal":0.30,
    })
    # Multimodal defaults
    MULTIMODAL_FRAME_FPS: float = 0.2
    MULTIMODAL_SIMILARITY_THRESHOLD: float = 0.9
    MULTIMODAL_MAX_CONCURRENT_SEGMENTS: int = 3
    MULTIMODAL_ENABLE_TEXT_ALIGNMENT: bool = True
    MULTIMODAL_MAX_ALIGNED_FRAMES: int = 3
    MULTIMODAL_EMBED_MODEL: str = "jina-embeddings-v4"
    MULTIMODAL_BATCH_SIZE: int = 10
    MULTIMODAL_API_DELAY: float = 0.1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

