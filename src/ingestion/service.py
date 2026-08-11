import asyncio
import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from src.config import config
from src.db.repository import Repository
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import Embedder
from src.ingestion.parser import PDFParseError, parse_pdf


class IngestionService:
    def __init__(self, repo: Repository, embedder: Embedder):
        self.repo = repo
        self.embedder = embedder

    async def ingest_file(self, file: UploadFile) -> dict:
        if not file.filename:
            raise ValueError("Missing filename")
        if file.content_type not in {"application/pdf", "application/x-pdf"}:
            raise ValueError("Only PDF files are supported")

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename).name).strip("._")
        if not safe_name:
            safe_name = "document.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"

        storage_dir = Path(config.STORAGE_DIR)
        storage_dir.mkdir(parents=True, exist_ok=True)
        stored_path = storage_dir / f"{uuid4()}_{safe_name}"

        raw = await file.read()
        if not raw:
            raise ValueError("Uploaded file is empty")
        if len(raw) > config.max_upload_bytes:
            raise ValueError(f"File exceeds MAX_UPLOAD_MB={config.MAX_UPLOAD_MB}")

        stored_path.write_bytes(raw)
        doc = await self.repo.create_document(
            filename=safe_name,
            content_type="application/pdf",
            file_path=str(stored_path),
            ingestion_metadata={"size_bytes": len(raw)},
        )

        try:
            pages = await asyncio.to_thread(parse_pdf, stored_path)
            chunks = chunk_pages(pages, doc_id=doc.id)
            if not chunks:
                raise PDFParseError("PDF contains no extractable text")

            embeddings = await asyncio.to_thread(
                self.embedder.encode, [chunk["content"] for chunk in chunks]
            )
            if len(embeddings) != len(chunks):
                raise RuntimeError("Embedding count does not match chunk count")

            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk["embedding"] = embedding.astype(float).tolist()

            await self.repo.insert_chunks(chunks)
            await self.repo.update_document_status(doc.id, "completed")
            return {
                "document_id": str(doc.id),
                "filename": safe_name,
                "chunk_count": len(chunks),
                "status": "completed",
            }
        except Exception as exc:
            await self.repo.session.rollback()
            await self.repo.update_document_status(doc.id, "failed")
            raise PDFParseError(f"Ingestion failed: {exc}") from exc
