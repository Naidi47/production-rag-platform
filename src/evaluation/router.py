from fastapi import APIRouter, Depends

from src.dependencies import get_evaluator
from src.evaluation.evaluator import Evaluator
from src.evaluation.schemas import RunEvalRequest, RunEvalResponse

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


@router.post("/run", response_model=RunEvalResponse)
async def run_evaluation(body: RunEvalRequest, evaluator: Evaluator = Depends(get_evaluator)):
    metrics = await evaluator.run(body.run_name)
    return {"run_name": body.run_name, "metrics": metrics}
