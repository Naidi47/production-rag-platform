from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb"
    PGVECTOR_DIMENSION: int = 384

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MODEL_DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
    EMBEDDING_BATCH_SIZE: int = Field(default=16, ge=1, le=256)
    RERANKER_BATCH_SIZE: int = Field(default=8, ge=1, le=128)
    RERANKER_MAX_LENGTH: int = Field(default=256, ge=32, le=1024)
    PRELOAD_MODELS: bool = False
    HF_HOME: str = "./data/huggingface"

    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0)
    LLM_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    LLM_MAX_TOKENS: int = Field(default=1024, ge=64, le=8192)

    CHUNK_SIZE: int = Field(default=800, ge=100, le=10000)
    CHUNK_OVERLAP: int = Field(default=120, ge=0, le=5000)
    MAX_UPLOAD_MB: int = Field(default=25, ge=1, le=500)
    STORAGE_DIR: str = "./data/uploads"

    TOP_K_RETRIEVAL: int = Field(default=20, ge=1, le=100)
    TOP_K_RERANK: int = Field(default=5, ge=1, le=50)
    RRF_K: int = Field(default=60, ge=1, le=1000)
    GUARDRAIL_SIMILARITY_THRESHOLD: float = Field(default=0.55, ge=0.0, le=1.0)

    LOG_LEVEL: str = "INFO"
    ENV: Literal["development", "production"] = "development"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    @field_validator("PGVECTOR_DIMENSION")
    @classmethod
    def validate_embedding_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("PGVECTOR_DIMENSION must be positive")
        return value

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info) -> int:
        chunk_size = info.data.get("CHUNK_SIZE", 800)
        if value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @property
    def async_database_url(self) -> str:
        raw = self.DATABASE_URL.strip()
        if raw.startswith("postgresql+asyncpg://"):
            return raw
        if raw.startswith("postgresql://"):
            return raw.replace("postgresql://", "postgresql+asyncpg://", 1)
        if raw.startswith("postgres://"):
            return raw.replace("postgres://", "postgresql+asyncpg://", 1)
        return raw

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.CORS_ORIGINS.split(",") if item.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


config = get_settings()
