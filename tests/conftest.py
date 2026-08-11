import os
from collections.abc import AsyncGenerator

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import config

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    config.async_database_url.rsplit("/", 1)[0] + "/ragdb_test",
)


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    from src.db.models import Base

    db_engine = create_async_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    async with db_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
    yield db_engine
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await db_engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_embedder():
    class FakeEmbedder:
        dimension = config.PGVECTOR_DIMENSION

        def encode(self, texts):
            return np.ones((len(texts), self.dimension), dtype=np.float32)

    return FakeEmbedder()


@pytest.fixture
def mock_reranker():
    class FakeReranker:
        def rerank(self, query, chunks):
            return [(i, float(len(chunks) - i)) for i in range(len(chunks))]

    return FakeReranker()


@pytest.fixture
def mock_llm_client():
    class FakeLLM:
        async def chat(self, messages, temperature=0.1, max_retries=None):
            return "I don't have enough information."

        async def close(self):
            pass

        def count_tokens(self, text):
            return 0

    return FakeLLM()


@pytest.fixture
def app(db_session, mock_embedder, mock_reranker, mock_llm_client):
    from src.dependencies import get_db, get_embedder, get_llm_client, get_reranker
    from src.main import create_app

    application = create_app()

    async def override_get_db():
        yield db_session

    application.dependency_overrides[get_db] = override_get_db
    application.dependency_overrides[get_embedder] = lambda: mock_embedder
    application.dependency_overrides[get_reranker] = lambda: mock_reranker
    application.dependency_overrides[get_llm_client] = lambda: mock_llm_client
    return application


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        yield async_client
