from typing import Any
from uuid import UUID

from src.config import config


def _find_boundary(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return end
    lookback = max(start + 1, end - int((end - start) * 0.2))
    for pos in range(end - 1, lookback - 1, -1):
        if text[pos] == "\n":
            return pos + 1
        if text[pos] in ".!?;" and pos + 1 < len(text) and text[pos + 1].isspace():
            return pos + 1
    return end


def chunk_pages(
    pages: list[dict],
    doc_id: UUID,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    size = config.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_number = page.get("page_number")
        text = (page.get("text") or "").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            raw_end = min(start + size, len(text))
            end = _find_boundary(text, start, raw_end)
            chunk_text = text[start:end].strip()

            if chunk_text:
                lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
                chunks.append(
                    {
                        "document_id": doc_id,
                        "content": chunk_text,
                        "chunk_index": len(chunks),
                        "page_number": page_number,
                        "metadata": {
                            "headings": [lines[0]] if lines and len(lines[0]) <= 100 else [],
                            "source_page": page_number,
                        },
                    }
                )

            if end >= len(text):
                break
            next_start = max(0, end - overlap)
            if next_start <= start:
                next_start = end
            start = next_start
    return chunks
