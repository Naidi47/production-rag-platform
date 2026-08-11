from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.db.repository import Repository
from src.dependencies import get_embedder, get_repository
from src.ingestion.schemas import IngestionResponse
from src.ingestion.service import IngestionService

router = APIRouter(prefix="/api/v1/ingestion", tags=["ingestion"])


@router.post("/upload", response_model=IngestionResponse)
async def upload_file(
    file: UploadFile = File(...),
    repo: Repository = Depends(get_repository),
    embedder=Depends(get_embedder),
):
    try:
        return await IngestionService(repo, embedder).ingest_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
