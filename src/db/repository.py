from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import and_, desc, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Chunk, Document, EvaluationRun, TestQuery


class Repository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_document(
        self,
        filename: str,
        content_type: str,
        file_path: str | None = None,
        ingestion_metadata: dict[str, Any] | None = None,
    ) -> Document:
        doc = Document(
            filename=filename,
            content_type=content_type,
            file_path=file_path,
            ingestion_metadata=ingestion_metadata or {},
            status="processing",
        )
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def update_document_status(self, doc_id: UUID, status: str) -> None:
        await self.session.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(status=status, updated_at=func.now())
        )
        await self.session.commit()

    async def insert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if chunks:
            await self.session.execute(insert(Chunk), chunks)
            await self.session.commit()

    async def semantic_search_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        document_ids: list[UUID] | None = None,
    ) -> Sequence[Any]:
        distance = Chunk.embedding.op("<=>")(query_embedding)
        stmt = select(Chunk, distance.label("distance"))
        conditions = []
        if filters:
            conditions.extend(Chunk.metadata.contains({k: v}) for k, v in filters.items())
        if document_ids:
            conditions.append(Chunk.document_id.in_(document_ids))
        if conditions:
            stmt = stmt.where(and_(*conditions))
        result = await self.session.execute(stmt.order_by(distance).limit(top_k))
        return result.all()

    async def keyword_search_chunks(
        self,
        query_text: str,
        top_k: int = 10,
        document_ids: list[UUID] | None = None,
    ) -> Sequence[Any]:
        tsquery = func.plainto_tsquery("english", query_text)
        rank = func.ts_rank_cd(Chunk.content_tsv, tsquery)
        stmt = select(Chunk, rank.label("rank")).where(Chunk.content_tsv.op("@@")(tsquery))
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        result = await self.session.execute(stmt.order_by(desc(rank)).limit(top_k))
        return result.all()

    async def get_chunks_by_ids(self, chunk_ids: list[UUID]) -> Sequence[Chunk]:
        if not chunk_ids:
            return []
        result = await self.session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
        rows = result.scalars().all()
        mapping = {row.id: row for row in rows}
        return [mapping[cid] for cid in chunk_ids if cid in mapping]

    async def get_all_test_queries(self) -> Sequence[TestQuery]:
        result = await self.session.execute(select(TestQuery).order_by(TestQuery.created_at))
        return result.scalars().all()

    async def upsert_test_query(self, item: dict[str, Any]) -> None:
        result = await self.session.execute(
            select(TestQuery).where(TestQuery.query_text == item["query_text"])
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = TestQuery(query_text=item["query_text"])
            self.session.add(row)
        row.expected_answer = item.get("expected_answer")
        row.expected_chunk_ids = item.get("expected_chunk_ids", [])
        row.category = item.get("category")
        row.difficulty = item.get("difficulty")
        row.metadata = item.get("metadata", {})
        await self.session.flush()

    async def add_evaluation_run(self, run: EvaluationRun) -> None:
        self.session.add(run)
        await self.session.flush()
