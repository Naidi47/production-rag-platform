from typing import Any, Sequence
from uuid import UUID

from src.db.repository import Repository


class SemanticSearch:
    def __init__(self, repo: Repository):
        self.repo = repo

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        document_ids: list[UUID] | None = None,
    ) -> Sequence[Any]:
        return await self.repo.semantic_search_chunks(
            query_embedding, top_k=top_k, filters=filters, document_ids=document_ids
        )
