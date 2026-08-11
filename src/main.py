from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from src.config import config
from src.db.engine import engine
from src.db.models import Base
from src.dependencies import get_embedder, get_llm_client, get_reranker
from src.evaluation.router import router as evaluation_router
from src.generation.router import router as generation_router
from src.ingestion.router import router as ingestion_router
from src.retrieval.router import router as retrieval_router
from src.utils.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

    if config.PRELOAD_MODELS:
        get_embedder()
        get_reranker()

    yield

    llm = get_llm_client()
    await llm.close()
    get_llm_client.cache_clear()
    get_embedder.cache_clear()
    get_reranker.cache_clear()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Platform",
        version="1.0.0",
        description="Async hybrid RAG API for PDF document question answering.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["system"])
    async def root():
        return {"name": "RAG Platform", "version": "1.0.0", "docs": "/docs"}

    @app.get("/health", tags=["system"])
    async def health():
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return {"status": "ok", "db": "connected"}
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "db": "unavailable", "detail": str(exc)},
            )

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(ingestion_router)
    app.include_router(retrieval_router)
    app.include_router(generation_router)
    app.include_router(evaluation_router)
    return app
