from pydantic import BaseModel, Field


class RunEvalRequest(BaseModel):
    run_name: str = Field(..., min_length=1, max_length=256)


class RunEvalResponse(BaseModel):
    run_name: str
    metrics: dict[str, float]
