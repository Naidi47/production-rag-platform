import pytest
from sqlalchemy import insert

from src.db.models import Chunk, Document


@pytest.mark.asyncio
async def test_hybrid_search_returns_deduplicated_chunks(client, db_session):
    doc = Document(filename="test.txt", content_type="text/plain", status="completed")
    db_session.add(doc)
    await db_session.flush()
    await db_session.execute(
        insert(Chunk),
        [
            {
                "document_id": doc.id,
                "content": "Artificial intelligence is transforming software engineering.",
                "embedding": [0.1] * 384,
                "chunk_index": 0,
                "page_number": 1,
                "metadata": {},
            },
            {
                "document_id": doc.id,
                "content": "Machine learning models require training data.",
                "embedding": [0.2] * 384,
                "chunk_index": 1,
                "page_number": 1,
                "metadata": {},
            },
        ],
    )
    await db_session.commit()

    response = await client.post(
        "/api/v1/retrieval/search",
        json={"query": "artificial intelligence software", "top_k": 10, "top_k_rerank": 5},
    )
    assert response.status_code == 200
    ids = [row["chunk_id"] for row in response.json()["results"]]
    assert len(ids) == len(set(ids))
