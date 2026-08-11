from collections.abc import Sequence
from uuid import UUID

import numpy as np


def _normalize_id(value: UUID | str) -> str:
    return str(value)


def recall_at_k(
    retrieved_ids: Sequence[UUID | str],
    gt_ids: Sequence[UUID | str],
    k: int,
) -> float:
    if not gt_ids or k <= 0:
        return 0.0
    retrieved = {_normalize_id(value) for value in retrieved_ids[:k]}
    ground_truth = {_normalize_id(value) for value in gt_ids}
    return len(retrieved & ground_truth) / len(ground_truth)


def mean_reciprocal_rank(
    retrieved_ids: Sequence[UUID | str],
    gt_ids: Sequence[UUID | str],
) -> float:
    ground_truth = {_normalize_id(value) for value in gt_ids}
    if not ground_truth:
        return 0.0
    for rank, value in enumerate(retrieved_ids, start=1):
        if _normalize_id(value) in ground_truth:
            return 1.0 / rank
    return 0.0


def answer_relevance(answer_embedding: np.ndarray, query_embedding: np.ndarray) -> float:
    return _cosine_similarity(answer_embedding, query_embedding)


def faithfulness(answer_embedding: np.ndarray, context_embedding: np.ndarray) -> float:
    return _cosine_similarity(answer_embedding, context_embedding)


def hallucination_rate(flagged_count: int, total: int) -> float:
    return 0.0 if total <= 0 else flagged_count / total


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if norm == 0 else float(np.dot(a, b) / norm)
