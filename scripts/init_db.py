#!/usr/bin/env python3
import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import config
from src.db.models import Base

logging.basicConfig(level=logging.INFO)


async def init_db() -> None:
    db_engine = create_async_engine(config.async_database_url, pool_pre_ping=True)
    try:
        async with db_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)
        logging.info("Database initialized successfully.")
    finally:
        await db_engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_db())
