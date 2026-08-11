from typing import Sequence
from uuid import UUID

from src.db.repository import Repository


class KeywordSearch:
    def __init__(self, repo: Repository):
        self.repo = repo

    async def search(
        self,
        query_text: str,
        top_k: int = 10,
        document_ids: list[UUID] | None = None,
    ) -> Sequence:
        return await self.repo.keyword_search_chunks(
            query_text, top_k=top_k, document_ids=document_ids
        )
