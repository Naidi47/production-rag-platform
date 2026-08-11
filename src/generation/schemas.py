from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    document_ids: list[str] | None = None
    search_filters: dict[str, Any] | None = None


class SourceCitation(BaseModel):
    chunk_id: str
    content_snippet: str
    document_id: str
    page_number: int | None = None
    relevance_score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceCitation]
    hallucination_check_passed: bool
    confidence: float
