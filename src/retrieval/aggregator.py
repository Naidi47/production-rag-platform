from collections import defaultdict
from uuid import UUID


def reciprocal_rank_fusion(
    semantic_results: list[tuple[UUID, float]],
    keyword_results: list[tuple[UUID, float]],
    k: int = 60,
) -> list[tuple[UUID, float]]:
    scores: dict[UUID, float] = defaultdict(float)
    for rank, (chunk_id, _) in enumerate(semantic_results, start=1):
        scores[chunk_id] += 1.0 / (k + rank)
    for rank, (chunk_id, _) in enumerate(keyword_results, start=1):
        scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
