import asyncio
from typing import Any
from uuid import UUID

from src.config import config
from src.db.repository import Repository
from src.ingestion.embedder import Embedder
from src.retrieval.aggregator import reciprocal_rank_fusion
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.reranker import Reranker
from src.retrieval.semantic_search import SemanticSearch


class RetrievalService:
    def __init__(self, repo: Repository, embedder: Embedder, reranker: Reranker):
        self.repo = repo
        self.embedder = embedder
        self.reranker = reranker

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        top_k_rerank: int | None = None,
        filters: dict[str, Any] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        top_k = top_k or config.TOP_K_RETRIEVAL
        top_k_rerank = top_k_rerank or config.TOP_K_RERANK

        parsed_document_ids = None
        if document_ids:
            try:
                parsed_document_ids = [UUID(value) for value in document_ids]
            except ValueError as exc:
                raise ValueError("document_ids must contain valid UUIDs") from exc

        embedding = await asyncio.to_thread(self.embedder.encode, [query])
        query_embedding = embedding[0].astype(float).tolist()

        semantic = SemanticSearch(self.repo)
        keyword = KeywordSearch(self.repo)
        sem_rows, key_rows = await asyncio.gather(
            semantic.search(
                query_embedding, top_k=top_k, filters=filters, document_ids=parsed_document_ids
            ),
            keyword.search(query, top_k=top_k, document_ids=parsed_document_ids),
        )

        semantic_results = [(row.Chunk.id, float(row.distance)) for row in sem_rows]
        keyword_results = [(row.Chunk.id, float(row.rank)) for row in key_rows]
        fused = reciprocal_rank_fusion(semantic_results, keyword_results, k=config.RRF_K)
        fused_ids = [chunk_id for chunk_id, _ in fused[:top_k]]
        if not fused_ids:
            return []

        chunks = await self.repo.get_chunks_by_ids(fused_ids)
        chunk_map = {chunk.id: chunk for chunk in chunks}
        ordered = [chunk_map[cid] for cid in fused_ids if cid in chunk_map]
        if not ordered:
            return []

        reranked = await asyncio.to_thread(
            self.reranker.rerank, query, [chunk.content for chunk in ordered]
        )
        return [
            {
                "chunk_id": str(ordered[index].id),
                "content": ordered[index].content,
                "document_id": str(ordered[index].document_id),
                "page_number": ordered[index].page_number,
                "metadata": ordered[index].metadata,
                "score": float(score),
            }
            for index, score in reranked[:top_k_rerank]
        ]
