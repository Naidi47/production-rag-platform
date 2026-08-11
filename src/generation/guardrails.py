import asyncio
from uuid import UUID

import numpy as np

from src.config import config
from src.ingestion.embedder import Embedder


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


class Guardrails:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.similarity_threshold = config.GUARDRAIL_SIMILARITY_THRESHOLD

    async def check(
        self,
        answer: str,
        valid_citation_ids: list[UUID],
        chunks: list[dict],
    ) -> dict:
        answer = answer.strip()
        refusal = answer.lower() == "i don't have enough information."
        reasons: list[str] = []
        hallucination = False

        if answer and not refusal and not valid_citation_ids:
            hallucination = True
            reasons.append("No valid citations found")

        context = " ".join(c.get("content", "") for c in chunks).strip()
        similarity = 0.0
        if answer and context:
            emb_answer, emb_context = await asyncio.gather(
                asyncio.to_thread(self.embedder.encode, [answer]),
                asyncio.to_thread(self.embedder.encode, [context]),
            )
            similarity = _cosine_similarity(emb_answer[0], emb_context[0])
            if not refusal and similarity < self.similarity_threshold:
                hallucination = True
                reasons.append(f"Low semantic similarity to context ({similarity:.2f})")

        return {
            "is_hallucination": hallucination,
            "confidence": max(0.0, min(1.0, similarity)),
            "reasons": reasons,
        }
