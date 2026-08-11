import pytest
from sqlalchemy import insert

from src.db.models import Chunk, Document
from src.dependencies import get_llm_client


@pytest.mark.asyncio
async def test_ask_includes_citations_and_sources(client, db_session, app):
    doc = Document(filename="geo.txt", content_type="text/plain", status="completed")
    db_session.add(doc)
    await db_session.flush()
    known_uuid = "22222222-2222-2222-2222-222222222222"
    await db_session.execute(
        insert(Chunk),
        [{
            "id": known_uuid,
            "document_id": doc.id,
            "content": "Paris is the capital of France.",
            "embedding": [0.1] * 384,
            "chunk_index": 0,
            "page_number": 1,
            "metadata": {},
        }],
    )
    await db_session.commit()

    class FakeLLM:
        async def chat(self, messages, temperature=0.1, max_retries=None):
            return f"Paris is the capital of France. [Source: {known_uuid}]"

        async def close(self):
            pass

    app.dependency_overrides[get_llm_client] = lambda: FakeLLM()
    response = await client.post(
        "/api/v1/generation/ask",
        json={"query": "What is the capital of France?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["sources"][0]["chunk_id"] == known_uuid
