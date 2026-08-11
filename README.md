# RAG Platform

Production-oriented asynchronous Retrieval-Augmented Generation (RAG) API for PDF document Q&A.

## Included

- PDF validation, persistent storage, page-aware chunking and embeddings.
- PostgreSQL + pgvector dense retrieval.
- PostgreSQL full-text keyword retrieval.
- Reciprocal Rank Fusion (RRF).
- Cross-encoder reranking.
- OpenAI-compatible LLM client with retries.
- Citation extraction and semantic guardrails.
- Evaluation metrics: Recall@5, MRR, relevance, faithfulness and hallucination rate.
- Docker Compose with pgvector and a separate integration-test database.
- CPU-friendly defaults and lazy ML model loading.
- Unit tests and PostgreSQL integration tests.
- CI workflow, example evaluation dataset and persistent model/data directories.

## Requirements

- Python 3.11+
- Docker Desktop + Docker Compose (recommended)
- An OpenAI-compatible API key for generation

## Fastest Docker setup

PowerShell:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `LLM_API_KEY` in `.env`, then:

```powershell
docker compose up --build
```

Endpoints:

- `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health: `http://localhost:8000/health`

The first run downloads the embedding and reranker models. They are cached under `data/huggingface`.

## Upload a PDF

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/ingestion/upload" `
  -F "file=@C:\path\to\mydoc.pdf"
```

## Search

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/retrieval/search" `
  -H "Content-Type: application/json" `
  -d '{"query":"What is this document about?","top_k":10,"top_k_rerank":5}'
```

## Ask a question

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/generation/ask" `
  -H "Content-Type: application/json" `
  -d '{"query":"What is this document about?"}'
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API metadata |
| GET | `/health` | Database health |
| POST | `/api/v1/ingestion/upload` | Upload and index a PDF |
| POST | `/api/v1/retrieval/search` | Hybrid retrieval |
| POST | `/api/v1/generation/ask` | RAG answer with citations |
| POST | `/api/v1/evaluation/run` | Run evaluation |

## CPU-friendly configuration

Default model:

`sentence-transformers/all-MiniLM-L6-v2` → 384 dimensions.

Recommended for a low-memory CPU laptop:

```ini
MODEL_DEVICE=cpu
PRELOAD_MODELS=false
EMBEDDING_BATCH_SIZE=8
RERANKER_BATCH_SIZE=4
RERANKER_MAX_LENGTH=256
```

Do not change `PGVECTOR_DIMENSION` unless the selected embedding model has the matching dimension.

## Local Python setup

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Run PostgreSQL + pgvector:

```powershell
docker compose up -d postgres
```

Initialize and start the API:

```powershell
python scripts/init_db.py
uvicorn src.main:create_app --factory --reload
```

## Tests

Unit tests, no database required:

```powershell
pytest tests/test_unit.py -q
```

Full integration tests:

```powershell
docker compose up -d postgres
pytest -q
```

## Evaluation

```powershell
python scripts/evaluate_rag.py `
  --run-name baseline `
  --dataset-path .\\eval-dataset.example.json `
  --output-dir .\\eval-reports
```

## Project structure

```text
rag-platform/
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── eval-dataset.example.json
├── pyproject.toml
├── README.md
├── data/
│   ├── huggingface/
│   └── uploads/
├── docker/postgres/init/01-create-test-db.sql
├── scripts/
│   ├── evaluate_rag.py
│   ├── init_db.py
│   └── seed_test_data.py
├── src/
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   ├── db/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── utils/
└── tests/
    ├── conftest.py
    ├── fixtures/sample.pdf
    ├── test_unit.py
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_generation.py
    └── test_evaluation.py
```

## Important implementation details

1. Only PDF uploads are accepted.
2. Uploads are size-limited by `MAX_UPLOAD_MB`.
3. Uploaded files are stored under `STORAGE_DIR`, not `/tmp`.
4. The embedding model dimension is checked against `PGVECTOR_DIMENSION`.
5. `document_ids` are applied during both dense and keyword retrieval.
6. Citation IDs are validated against retrieved chunks before sources are returned.
7. ML models load lazily unless `PRELOAD_MODELS=true`.
8. The shared LLM HTTP client is closed during application shutdown.
9. Internal exception details are not exposed by the global 500 handler.
10. Evaluation IDs are normalized so UUID/string representations do not silently produce zero Recall/MRR.

## Stop / reset

```powershell
docker compose down
```

Delete the PostgreSQL volume too:

```powershell
docker compose down -v
```

Delete downloaded models/uploads:

```powershell
Remove-Item -Recurse -Force .\\data\\huggingface\\*
Remove-Item -Recurse -Force .\\data\\uploads\\*
```

## Troubleshooting

### Port 5432 is already used

Change the host mapping from `5432:5432` to `5433:5432` and update the local `DATABASE_URL`.

### Port 8000 is already used

Change the host mapping from `8000:8000` to another host port.

### First startup is slow

The first startup downloads ML models. Keep `data/huggingface` to reuse the cache.

### LLM errors

Verify:

```ini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
LLM_MODEL=gpt-4o-mini
```

The model must exist at the selected OpenAI-compatible provider.

## Before production

- Replace the default PostgreSQL password.
- Store `LLM_API_KEY` in a secret manager.
- Restrict CORS.
- Add authentication and authorization.
- Put the API behind HTTPS.
- Use object storage for multi-instance deployments.
- Use Alembic/database migrations instead of `create_all` for controlled production schema evolution.
