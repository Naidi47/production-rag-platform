#!/usr/bin/env python3
import asyncio
import logging

from sqlalchemy.ext.asyncio import async_sessionmaker

from src.db.engine import engine
from src.db.repository import Repository

ROWS = [
    {
        "query_text": "What is the main topic of the sample document?",
        "expected_answer": "A sample PDF for testing the RAG platform.",
        "expected_chunk_ids": [],
        "category": "factual",
        "difficulty": "easy",
    },
    {
        "query_text": "How does the chunking strategy handle overlap?",
        "expected_answer": "It uses a sliding window with configurable overlap.",
        "expected_chunk_ids": [],
        "category": "technical",
        "difficulty": "medium",
    },
]


async def seed() -> None:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        repo = Repository(session)
        for row in ROWS:
            await repo.upsert_test_query(row)
        await session.commit()
    logging.info("Seeded %d evaluation queries.", len(ROWS))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed())
