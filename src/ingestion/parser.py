from pathlib import Path

from pypdf import PdfReader


class PDFParseError(Exception):
    """Raised when a PDF cannot be parsed."""


def parse_pdf(file_path: str | Path) -> list[dict]:
    try:
        reader = PdfReader(str(file_path), strict=False)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "").strip()
            pages.append({"page_number": page_number, "text": text, "bbox": None})
        if not pages:
            raise PDFParseError("PDF contains no pages")
        return pages
    except PDFParseError:
        raise
    except Exception as exc:
        raise PDFParseError(f"Failed to parse PDF {file_path}: {exc}") from exc
