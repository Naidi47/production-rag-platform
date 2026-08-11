import io

import pytest
from pypdf import PdfReader
from sqlalchemy import select

from src.db.models import Chunk

MINIMAL_PDF = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 46 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Hello, this is a sample PDF for testing.) Tj\nET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000254 00000 n \n0000000353 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n432\n%%EOF'


def test_fixture_pdf_is_readable():
    reader = PdfReader(io.BytesIO(MINIMAL_PDF))
    assert len(reader.pages) == 1
    assert "sample PDF" in (reader.pages[0].extract_text() or "")


@pytest.mark.asyncio
async def test_upload_pdf(client, db_session):
    response = await client.post(
        "/api/v1/ingestion/upload",
        files={"file": ("sample.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["chunk_count"] > 0

    result = await db_session.execute(select(Chunk))
    assert len(result.scalars().all()) >= data["chunk_count"]
