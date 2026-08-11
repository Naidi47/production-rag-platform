from fastapi import APIRouter, Depends, HTTPException

from src.dependencies import get_retrieval_service
from src.retrieval.schemas import SearchRequest, SearchResponse
from src.retrieval.service import RetrievalService

router = APIRouter(prefix="/api/v1/retrieval", tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, service: RetrievalService = Depends(get_retrieval_service)):
    try:
        results = await service.search(
            body.query,
            top_k=body.top_k,
            top_k_rerank=body.top_k_rerank,
            filters=body.filters,
            document_ids=body.document_ids,
        )
        return {"query": body.query, "results": results}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
