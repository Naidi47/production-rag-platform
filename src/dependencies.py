from collections.abc import AsyncIterator
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.engine import get_session
from src.db.repository import Repository
from src.evaluation.evaluator import Evaluator
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.service import GenerationService
from src.ingestion.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.service import RetrievalService


async def get_db() -> AsyncIterator[AsyncSession]:
    async for session in get_session():
        yield session


def get_repository(db: AsyncSession = Depends(get_db)) -> Repository:
    return Repository(db)


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()


@lru_cache
def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


def get_retrieval_service(
    repo: Repository = Depends(get_repository),
    embedder: Embedder = Depends(get_embedder),
    reranker: Reranker = Depends(get_reranker),
) -> RetrievalService:
    return RetrievalService(repo, embedder, reranker)


def get_generation_service(
    retrieval: RetrievalService = Depends(get_retrieval_service),
    llm: LLMClient = Depends(get_llm_client),
    prompt: PromptBuilder = Depends(get_prompt_builder),
    embedder: Embedder = Depends(get_embedder),
) -> GenerationService:
    return GenerationService(retrieval, llm, prompt, embedder)


def get_evaluator(
    repo: Repository = Depends(get_repository),
    retrieval: RetrievalService = Depends(get_retrieval_service),
    generation: GenerationService = Depends(get_generation_service),
    embedder: Embedder = Depends(get_embedder),
) -> Evaluator:
    return Evaluator(repo, retrieval, generation, embedder)
