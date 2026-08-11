from fastapi import APIRouter, Depends, HTTPException

from src.dependencies import get_generation_service
from src.generation.schemas import AskRequest, AskResponse
from src.generation.service import GenerationService

router = APIRouter(prefix="/api/v1/generation", tags=["generation"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    service: GenerationService = Depends(get_generation_service),
):
    try:
        return await service.ask(
            body.query, filters=body.search_filters, document_ids=body.document_ids
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
