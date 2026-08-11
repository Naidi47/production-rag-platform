from uuid import UUID

import pytest

from src.evaluation.metrics import mean_reciprocal_rank, recall_at_k
from src.generation.citation_extractor import extract_citations
from src.ingestion.chunker import chunk_pages
from src.retrieval.aggregator import reciprocal_rank_fusion


def test_citation_extraction():
    cid = UUID("11111111-1111-1111-1111-111111111111")
    clean, ids = extract_citations(f"Answer. [Source: {cid}]")
    assert clean == "Answer."
    assert ids == [cid]


def test_metrics_accept_string_and_uuid_ids():
    cid = "11111111-1111-1111-1111-111111111111"
    assert recall_at_k([cid], [UUID(cid)], 5) == 1.0
    assert mean_reciprocal_rank([cid], [UUID(cid)]) == 1.0


def test_rrf_uses_both_rankings():
    a = UUID("11111111-1111-1111-1111-111111111111")
    b = UUID("22222222-2222-2222-2222-222222222222")
    result = reciprocal_rank_fusion([(a, 0.1)], [(b, 1.0)])
    assert {item[0] for item in result} == {a, b}


def test_chunker_preserves_page_and_rejects_invalid_overlap():
    doc_id = UUID("11111111-1111-1111-1111-111111111111")
    chunks = chunk_pages([{"page_number": 3, "text": "A " * 100}], doc_id, 50, 10)
    assert chunks
    assert all(chunk["page_number"] == 3 for chunk in chunks)
    with pytest.raises(ValueError):
        chunk_pages([{"page_number": 1, "text": "hello"}], doc_id, 10, 10)
