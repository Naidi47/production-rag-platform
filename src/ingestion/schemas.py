from pydantic import BaseModel, Field


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int = Field(ge=0)
    status: str
