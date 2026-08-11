from typing import Any
from uuid import UUID

from src.generation.citation_extractor import extract_citations
from src.generation.guardrails import Guardrails
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.ingestion.embedder import Embedder
from src.retrieval.service import RetrievalService


class GenerationService:
    def __init__(
        self,
        retrieval: RetrievalService,
        llm: LLMClient,
        prompt_builder: PromptBuilder,
        embedder: Embedder,
    ):
        self.retrieval = retrieval
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.guardrails = Guardrails(embedder)

    async def ask(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        chunks = await self.retrieval.search(
            query, filters=filters, document_ids=document_ids
        )
        if not chunks:
            return {
                "query": query,
                "answer": "I don't have enough information.",
                "sources": [],
                "hallucination_check_passed": True,
                "confidence": 0.0,
            }

        raw_answer = await self.llm.chat(self.prompt_builder.build(query, chunks))
        clean_answer, citation_ids = extract_citations(raw_answer)

        retrieved_ids = {UUID(c["chunk_id"]) for c in chunks}
        valid_ids = [cid for cid in citation_ids if cid in retrieved_ids]
        guard_result = await self.guardrails.check(clean_answer, valid_ids, chunks)

        chunk_map = {UUID(c["chunk_id"]): c for c in chunks}
        sources = []
        for cid in valid_ids:
            chunk = chunk_map.get(cid)
            if chunk:
                sources.append(
                    {
                        "chunk_id": str(cid),
                        "content_snippet": chunk["content"][:500],
                        "document_id": str(chunk["document_id"]),
                        "page_number": chunk["page_number"],
                        "relevance_score": chunk["score"],
                    }
                )

        unique_sources = list({item["chunk_id"]: item for item in sources}.values())
        return {
            "query": query,
            "answer": clean_answer,
            "sources": unique_sources,
            "hallucination_check_passed": not guard_result["is_hallucination"],
            "confidence": guard_result["confidence"],
        }
