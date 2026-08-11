#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from src.db.engine import engine
from src.db.repository import Repository
from src.evaluation.evaluator import Evaluator
from src.evaluation.report import Report
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import PromptBuilder
from src.generation.service import GenerationService
from src.ingestion.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.service import RetrievalService


async def main(run_name: str, dataset_path: str | None, output_dir: str) -> None:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        repo = Repository(session)
        embedder = Embedder()
        retrieval = RetrievalService(repo, embedder, Reranker())
        generation = GenerationService(retrieval, LLMClient(), PromptBuilder(), embedder)
        evaluator = Evaluator(repo, retrieval, generation, embedder)

        if dataset_path:
            raw = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Evaluation dataset must be a JSON array")
            await evaluator.inject_external_dataset(raw)

        metrics = await evaluator.run(run_name)

    report = Report(metrics, run_name)
    report.save(output_dir)
    print(report.to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG evaluation CLI")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-dir", default="./eval-reports")
    args = parser.parse_args()
    asyncio.run(main(args.run_name, args.dataset_path, args.output_dir))
