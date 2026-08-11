from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=20, ge=1, le=100)
    top_k_rerank: int = Field(default=5, ge=1, le=50)
    filters: dict[str, Any] | None = None
    document_ids: list[str] | None = None


class SourceResult(BaseModel):
    chunk_id: str
    content: str
    document_id: str
    page_number: int | None = None
    metadata: dict[str, Any] | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SourceResult]
