from typing import Any

from pydantic import BaseModel, Field


class TestQueryModel(BaseModel):
    query_text: str = Field(min_length=1)
    expected_answer: str | None = None
    expected_chunk_ids: list[str] = Field(default_factory=list)
    category: str | None = None
    difficulty: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
