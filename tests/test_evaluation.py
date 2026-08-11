import pytest

from src.db.models import TestQuery


@pytest.mark.asyncio
async def test_evaluator_endpoint_returns_metrics(client, db_session):
    db_session.add(
        TestQuery(
            query_text="What is the sample about?",
            expected_answer="Testing.",
            expected_chunk_ids=[],
            category="factual",
            difficulty="easy",
        )
    )
    await db_session.commit()

    response = await client.post("/api/v1/evaluation/run", json={"run_name": "test-run"})
    assert response.status_code == 200
    metrics = response.json()["metrics"]
    assert "hallucination_rate" in metrics
    assert metrics["query_count"] >= 1
