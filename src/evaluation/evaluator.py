import asyncio
from datetime import datetime, timezone
from uuid import UUID

from src.db.models import EvaluationRun
from src.db.repository import Repository
from src.evaluation.metrics import answer_relevance, faithfulness, mean_reciprocal_rank, recall_at_k
from src.generation.service import GenerationService
from src.ingestion.embedder import Embedder
from src.retrieval.service import RetrievalService


class Evaluator:
    def __init__(self, repo: Repository, retrieval_service: RetrievalService,
                 generation_service: GenerationService, embedder: Embedder):
        self.repo = repo
        self.retrieval = retrieval_service
        self.generation = generation_service
        self.embedder = embedder

    async def inject_external_dataset(self, raw_items: list[dict]) -> None:
        for item in raw_items:
            if "query_text" not in item:
                raise ValueError("Each evaluation item requires query_text")
            await self.repo.upsert_test_query(item)
        await self.repo.session.commit()

    async def run(self, run_name: str) -> dict[str, float]:
        queries = await self.repo.get_all_test_queries()
        run = EvaluationRun(
            run_name=run_name,
            metrics={},
            started_at=datetime.now(timezone.utc),
        )
        await self.repo.add_evaluation_run(run)

        accum = {key: [] for key in ("recall@5", "mrr", "relevance", "faithfulness", "hallucination_flags")}

        for tq in queries:
            search_results = await self.retrieval.search(tq.query_text, top_k=5, top_k_rerank=5)
            retrieved_ids = [r["chunk_id"] for r in search_results]
            gt_ids = []
            for cid in tq.expected_chunk_ids or []:
                try:
                    gt_ids.append(UUID(cid))
                except (ValueError, TypeError):
                    continue

            accum["recall@5"].append(recall_at_k(retrieved_ids, gt_ids, 5))
            accum["mrr"].append(mean_reciprocal_rank(retrieved_ids, gt_ids))

            gen_result = await self.generation.ask(tq.query_text)
            emb_answer = await asyncio.to_thread(self.embedder.encode, [gen_result["answer"]])
            emb_query = await asyncio.to_thread(self.embedder.encode, [tq.query_text])
            accum["relevance"].append(answer_relevance(emb_answer[0], emb_query[0]))

            context_text = " ".join(s["content_snippet"] for s in gen_result["sources"])
            if context_text.strip():
                emb_context = await asyncio.to_thread(self.embedder.encode, [context_text])
                accum["faithfulness"].append(faithfulness(emb_answer[0], emb_context[0]))
            else:
                accum["faithfulness"].append(0.0)

            accum["hallucination_flags"].append(
                0.0 if gen_result["hallucination_check_passed"] else 1.0
            )

        final_metrics = {
            "query_count": float(len(queries)),
            "recall@5": _mean(accum["recall@5"]),
            "mrr": _mean(accum["mrr"]),
            "relevance": _mean(accum["relevance"]),
            "faithfulness": _mean(accum["faithfulness"]),
            "hallucination_rate": _mean(accum["hallucination_flags"]),
        }
        run.metrics = final_metrics
        run.completed_at = datetime.now(timezone.utc)
        await self.repo.session.commit()
        return final_metrics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
