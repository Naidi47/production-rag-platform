# Production RAG Platform

> A production-oriented Retrieval-Augmented Generation (RAG) API for document ingestion, hybrid retrieval, cross-encoder reranking, grounded LLM generation, citations, and retrieval evaluation.

<p align="center">

**FastAPI · PostgreSQL · pgvector · Sentence Transformers · Cross-Encoder · OpenAI-Compatible LLMs · Docker**

</p>

---

## Overview

**Production RAG Platform** is an asynchronous, API-first Retrieval-Augmented Generation system designed for reliable question answering over PDF documents.

The platform implements an end-to-end RAG pipeline:

```text
PDF Document
     │
     ▼
Document Validation
     │
     ▼
PDF Parsing
     │
     ▼
Page-Aware Chunking
     │
     ▼
Embedding Generation
     │
     ▼
PostgreSQL + pgvector
     │
     ├─────────────────────┐
     ▼                     ▼
Dense Semantic Search   Keyword Search
     │                     │
     └──────────┬──────────┘
                ▼
       Reciprocal Rank Fusion
                │
                ▼
       Cross-Encoder Reranking
                │
                ▼
         Relevant Context
                │
                ▼
       Grounded LLM Generation
                │
                ▼
       Answer + Citations
```

The goal is not simply to call an LLM with retrieved text, but to provide a structured retrieval and generation pipeline with configurable ranking, reranking, citation validation, guardrails, and evaluation capabilities.

---

## Key Features

### Document ingestion

* PDF-only document ingestion
* File validation and upload size limits
* Persistent document storage
* Page-aware text extraction
* Configurable chunk size and overlap
* Batch embedding generation
* Persistent Hugging Face model caching

### Hybrid retrieval

The retrieval layer combines two complementary approaches:

**Dense semantic retrieval**

Uses sentence-transformer embeddings and PostgreSQL/pgvector to retrieve semantically similar chunks.

**Keyword retrieval**

Uses PostgreSQL full-text search to capture exact terminology, names, identifiers, and keyword-heavy queries.

The two result sets are combined using **Reciprocal Rank Fusion (RRF)**.

### Cross-encoder reranking

Initial retrieval produces a broader candidate set. A cross-encoder then scores the query-document pairs to produce a more precise final ranking.

```text
Query
  │
  ▼
Candidate Retrieval
  │
  ├── Dense Search
  │
  └── Keyword Search
          │
          ▼
       RRF Fusion
          │
          ▼
    Cross-Encoder
      Reranking
          │
          ▼
    Top-K Context
```

### Grounded generation

The generation layer:

* Builds prompts from retrieved context
* Uses an OpenAI-compatible LLM API
* Supports configurable retries and timeouts
* Extracts citations from generated responses
* Validates citations against retrieved chunks
* Applies semantic guardrails before returning answers

### Evaluation

The platform includes an evaluation framework for measuring retrieval and answer quality.

Supported metrics include:

* Recall@5
* Mean Reciprocal Rank (MRR)
* Relevance
* Faithfulness
* Hallucination rate

Evaluation results can be written to the `eval-reports/` directory.

### Production-oriented engineering

The application includes:

* Asynchronous FastAPI endpoints
* SQLAlchemy async database access
* PostgreSQL + pgvector
* Shared HTTP client lifecycle management
* Lazy ML model loading
* Structured logging
* Centralized configuration
* Global exception handling
* Docker Compose deployment
* Unit and integration tests
* CI workflow

---

# Architecture

## High-Level Architecture

```text
                         ┌──────────────────────┐
                         │       Client         │
                         │  Swagger / Frontend  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI API     │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
          ┌────────────┐    ┌──────────────┐    ┌──────────────┐
          │ Ingestion  │    │  Retrieval   │    │  Generation  │
          └─────┬──────┘    └──────┬───────┘    └──────┬───────┘
                │                  │                    │
                ▼                  ▼                    ▼
          ┌────────────┐    ┌──────────────┐    ┌──────────────┐
          │ PDF Parser │    │ Dense Search │    │ Prompt Build │
          │ + Chunker  │    │ Keyword      │    │ LLM Client   │
          └─────┬──────┘    │ Search       │    │ Guardrails   │
                │            │ RRF          │    │ Citations    │
                │            │ Reranker     │    └──────┬───────┘
                │            └──────┬───────┘           │
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ PostgreSQL + pgvector│
                         └──────────────────────┘
```

---

# Technology Stack

| Layer                   | Technology               |
| ----------------------- | ------------------------ |
| API                     | FastAPI                  |
| Runtime                 | Python 3.11+             |
| Database                | PostgreSQL 15            |
| Vector Database         | pgvector                 |
| ORM                     | SQLAlchemy 2.x           |
| Async Driver            | asyncpg                  |
| Embeddings              | Sentence Transformers    |
| Default Embedding Model | `all-MiniLM-L6-v2`       |
| Reranker                | Cross-Encoder            |
| Default Reranker        | `ms-marco-MiniLM-L-6-v2` |
| LLM Interface           | OpenAI-compatible API    |
| PDF Processing          | pypdf                    |
| Validation              | Pydantic                 |
| Configuration           | pydantic-settings        |
| Containerization        | Docker / Docker Compose  |
| Testing                 | pytest / pytest-asyncio  |
| Linting                 | Ruff                     |
| CI                      | GitHub Actions           |

---

# Project Structure

```text
production-rag-platform/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docker/
│   └── postgres/
│       └── init/
│           └── 01-create-test-db.sql
│
├── data/
│   ├── huggingface/
│   └── uploads/
│
├── eval-reports/
│
├── scripts/
│   ├── evaluate_rag.py
│   ├── init_db.py
│   └── seed_test_data.py
│
├── src/
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   │
│   ├── db/
│   │   ├── engine.py
│   │   ├── models.py
│   │   └── repository.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── parser.py
│   │   ├── router.py
│   │   ├── schemas.py
│   │   └── service.py
│   │
│   ├── retrieval/
│   │   ├── aggregator.py
│   │   ├── keyword_search.py
│   │   ├── reranker.py
│   │   ├── router.py
│   │   ├── schemas.py
│   │   ├── semantic_search.py
│   │   └── service.py
│   │
│   ├── generation/
│   │   ├── citation_extractor.py
│   │   ├── guardrails.py
│   │   ├── llm_client.py
│   │   ├── prompt_builder.py
│   │   ├── router.py
│   │   ├── schemas.py
│   │   └── service.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── report.py
│   │   ├── router.py
│   │   ├── schemas.py
│   │   └── test_queries.py
│   │
│   └── utils/
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── sample.pdf
│   ├── test_unit.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_evaluation.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── eval-dataset.example.json
├── pyproject.toml
└── README.md
```

---

# Getting Started

## Prerequisites

### Required

* Python 3.11+
* Docker Desktop
* Docker Compose
* Git

### Required for LLM generation

An API key for an OpenAI-compatible LLM provider.

The default configuration uses:

```text
https://api.openai.com/v1
```

The LLM provider can be changed through environment variables.

---

# Option 1 — Docker Compose

Docker Compose is the recommended way to run the complete stack.

## 1. Clone the repository

```bash
git clone https://github.com/Naidi47/production-rag-platform.git
cd production-rag-platform
```

## 2. Create environment configuration

### PowerShell

```powershell
Copy-Item .env.example .env
```

Open the file:

```powershell
notepad .env
```

Configure:

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini
```

Do **not** commit `.env` to Git.

## 3. Start the application

```powershell
docker compose up --build
```

Docker Compose starts:

```text
PostgreSQL + pgvector
        │
        ▼
     FastAPI
        │
        ▼
 RAG Application
```

The first startup may take longer because the embedding and reranker models need to be downloaded.

---

# API

Once the application is running:

## API root

```text
http://localhost:8000
```

## Swagger UI

```text
http://localhost:8000/docs
```

## ReDoc

```text
http://localhost:8000/redoc
```

## Health check

```text
http://localhost:8000/health
```

---

# API Endpoints

| Method | Endpoint                   | Description                     |
| ------ | -------------------------- | ------------------------------- |
| `GET`  | `/`                        | API metadata                    |
| `GET`  | `/health`                  | Database/application health     |
| `POST` | `/api/v1/ingestion/upload` | Upload and index a PDF          |
| `POST` | `/api/v1/retrieval/search` | Perform hybrid retrieval        |
| `POST` | `/api/v1/generation/ask`   | Generate a grounded RAG answer  |
| `POST` | `/api/v1/evaluation/run`   | Execute the evaluation pipeline |

---

# Upload a Document

Using PowerShell:

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/ingestion/upload" `
  -F "file=@C:\path\to\document.pdf"
```

The ingestion pipeline performs:

```text
PDF
 ↓
Validation
 ↓
Text extraction
 ↓
Page-aware chunking
 ↓
Embedding generation
 ↓
PostgreSQL / pgvector
```

Uploaded documents are stored using the configured `STORAGE_DIR`.

---

# Perform Retrieval

Example:

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/retrieval/search" `
  -H "Content-Type: application/json" `
  -d '{"query":"What is this document about?","top_k":10,"top_k_rerank":5}'
```

Retrieval consists of:

1. Dense vector search
2. PostgreSQL keyword search
3. Reciprocal Rank Fusion
4. Cross-encoder reranking
5. Top-K context selection

---

# Ask a RAG Question

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/generation/ask" `
  -H "Content-Type: application/json" `
  -d '{"query":"What is this document about?"}'
```

The generation pipeline:

```text
User Query
    │
    ▼
Hybrid Retrieval
    │
    ▼
Reranked Context
    │
    ▼
Prompt Construction
    │
    ▼
LLM
    │
    ▼
Citation Extraction
    │
    ▼
Citation Validation
    │
    ▼
Final Answer
```

---

# Configuration

Configuration is controlled through environment variables.

Create `.env` from `.env.example`.

## Database

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb
PGVECTOR_DIMENSION=384
```

## Embedding Model

```env
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
MODEL_DEVICE=auto
EMBEDDING_BATCH_SIZE=16
```

The default embedding model produces **384-dimensional vectors**, therefore:

```env
PGVECTOR_DIMENSION=384
```

must match the selected model's output dimension.

## Reranker

```env
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_BATCH_SIZE=8
RERANKER_MAX_LENGTH=256
```

## LLM

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=3
LLM_MAX_TOKENS=1024
```

## Chunking

```env
CHUNK_SIZE=800
CHUNK_OVERLAP=120
MAX_UPLOAD_MB=25
```

## Retrieval

```env
TOP_K_RETRIEVAL=20
TOP_K_RERANK=5
RRF_K=60
GUARDRAIL_SIMILARITY_THRESHOLD=0.55
```

---

# CPU-Friendly Configuration

The default configuration is designed to work on CPU-only development environments.

Recommended settings for a lower-memory machine:

```env
MODEL_DEVICE=cpu
PRELOAD_MODELS=false
EMBEDDING_BATCH_SIZE=8
RERANKER_BATCH_SIZE=4
RERANKER_MAX_LENGTH=256
```

Models are loaded lazily by default:

```env
PRELOAD_MODELS=false
```

This reduces initial application startup overhead.

Downloaded models are cached under:

```text
data/huggingface/
```

---

# Local Python Development

Docker Compose is recommended for the full stack, but the API can also be run directly from a Python virtual environment.

## Create virtual environment

```powershell
py -3.11 -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project:

```powershell
pip install -e ".[dev]"
```

Create environment configuration:

```powershell
Copy-Item .env.example .env
```

Start PostgreSQL:

```powershell
docker compose up -d postgres
```

Initialize the database:

```powershell
python scripts/init_db.py
```

Start FastAPI:

```powershell
uvicorn src.main:create_app --factory --reload
```

The API will be available at:

```text
http://localhost:8000
```

---

# Development Docker Profile

For containerized development with source-code reload:

```powershell
docker compose --profile dev up --build
```

The development API is exposed on:

```text
http://localhost:8001
```

---

# Testing

## Unit Tests

Unit tests that do not require the PostgreSQL integration environment:

```powershell
pytest tests/test_unit.py -q
```

## Full Test Suite

Start PostgreSQL:

```powershell
docker compose up -d postgres
```

Run:

```powershell
pytest -q
```

The test suite covers areas including:

* ingestion
* retrieval
* generation
* evaluation
* application behavior

---

# RAG Evaluation

The repository includes an example evaluation dataset:

```text
eval-dataset.example.json
```

Run the evaluator:

```powershell
python scripts/evaluate_rag.py `
  --run-name baseline `
  --dataset-path .\eval-dataset.example.json `
  --output-dir .\eval-reports
```

The evaluation framework is intended to measure both retrieval and answer quality.

Example metrics:

```text
Recall@5
MRR
Relevance
Faithfulness
Hallucination Rate
```

Evaluation outputs are written to:

```text
eval-reports/
```

---

# Retrieval Design

A key design decision is to avoid relying exclusively on vector similarity.

## Dense Retrieval

Dense retrieval captures semantic relationships between queries and document chunks.

```text
Query
  ↓
Embedding Model
  ↓
Vector
  ↓
pgvector similarity search
```

## Keyword Retrieval

Keyword retrieval complements semantic search when exact terms are important.

Examples include:

* product names
* identifiers
* technical terminology
* acronyms
* exact phrases

## Reciprocal Rank Fusion

The dense and keyword result lists are combined using RRF.

Conceptually:

```text
Dense Results ──────┐
                    ├──► RRF ──► Candidate Set
Keyword Results ────┘
```

## Cross-Encoder Reranking

The candidate set is then reranked using a cross-encoder:

```text
Query + Candidate Chunk
          │
          ▼
   Cross-Encoder Score
          │
          ▼
    Final Ranking
```

This multi-stage architecture separates:

* **high-recall candidate retrieval**
* **high-precision reranking**

---

# Citation and Grounding Strategy

The generation layer does not blindly trust arbitrary citation identifiers generated by the LLM.

Retrieved chunks are associated with source metadata, and generated citation references are validated against the retrieved context before sources are returned.

This helps reduce invalid citations and makes the answer traceable to retrieved document chunks.

The system also applies semantic guardrails based on configurable similarity thresholds.

---

# Production Considerations

This repository is designed with production-oriented patterns, but it should **not be interpreted as a fully production-hardened deployment out of the box**.

Before exposing the service publicly, consider implementing:

* Authentication
* Authorization
* Rate limiting
* HTTPS/TLS
* Secret management
* Restricted CORS
* Request validation and abuse protection
* Centralized observability
* Metrics and tracing
* Database migrations
* Object storage for uploaded documents
* Background workers for expensive ingestion
* Horizontal scaling
* Persistent external model storage
* Database backups
* Resource quotas
* Network isolation

The default Docker Compose credentials are development credentials and should not be used for an internet-facing deployment.

---

# Security

## Secrets

Never commit:

```text
.env
API keys
access tokens
database credentials
private certificates
```

The repository includes `.env.example` for configuration documentation.

## Uploaded Documents

Uploaded documents are stored locally according to:

```env
STORAGE_DIR=./data/uploads
```

For multi-instance production deployments, object storage is recommended.

---

# Performance Considerations

The RAG pipeline intentionally uses a multi-stage retrieval architecture.

A typical flow is:

```text
                Recall
                  ▲
                  │
       ┌────────────────────┐
       │ Dense + Keyword    │
       │ Retrieval          │
       └─────────┬──────────┘
                 │
                 ▼
       ┌────────────────────┐
       │ RRF Candidate      │
       │ Fusion             │
       └─────────┬──────────┘
                 │
                 ▼
       ┌────────────────────┐
       │ Cross-Encoder      │
       │ Reranking          │
       └─────────┬──────────┘
                 │
                 ▼
              Precision
```

The initial retrieval stage favors recall, while reranking improves the quality of the final context passed to the LLM.

---

# Operational Commands

## Start

```powershell
docker compose up --build
```

## Start in background

```powershell
docker compose up -d
```

## View logs

```powershell
docker compose logs -f
```

## Stop

```powershell
docker compose down
```

## Stop and remove database volume

```powershell
docker compose down -v
```

> `docker compose down -v` removes the PostgreSQL Docker volume and therefore deletes the local database contents.

---

# Troubleshooting

## Python version

The project requires:

```text
Python >= 3.11
```

Check:

```powershell
python --version
```

or:

```powershell
py --version
```

## PostgreSQL port conflict

The default PostgreSQL port is:

```text
5432
```

If another service is using it, change the host-side Docker mapping.

For example:

```yaml
ports:
  - "5433:5432"
```

Then update the local database configuration accordingly.

## API port conflict

The default API port is:

```text
8000
```

You can change the host-side mapping:

```yaml
ports:
  - "8001:8000"
```

## Slow first startup

The first startup can be slow because the embedding and reranker models must be downloaded.

Once downloaded, the models are cached under:

```text
data/huggingface/
```

## LLM configuration errors

Verify:

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini
```

The selected model must exist at the configured OpenAI-compatible provider.

---

# CI

The repository includes a GitHub Actions workflow under:

```text
.github/workflows/ci.yml
```

The CI pipeline is intended to provide automated validation for the repository as changes are pushed.

---

# Design Principles

The project follows several architectural principles:

### Separation of concerns

Ingestion, retrieval, generation, database access, and evaluation are implemented as separate application layers.

### Async I/O

FastAPI and SQLAlchemy's asynchronous APIs are used for network and database operations.

### Configurability

Models, retrieval parameters, chunking parameters, LLM configuration, and application behavior are controlled through environment variables.

### Retrieval before generation

The LLM is treated as a generation component rather than the source of truth.

### Multi-stage retrieval

The system separates candidate retrieval from expensive reranking.

### Evaluation-driven development

Retrieval and generation quality should be measured rather than inferred solely from qualitative examples.

---

# Roadmap

Potential future improvements include:

* [ ] Authentication and authorization
* [ ] Multi-user document isolation
* [ ] Background ingestion workers
* [ ] Streaming generation
* [ ] Conversation/session memory
* [ ] Additional document formats
* [ ] OCR support for scanned PDFs
* [ ] Metadata filtering
* [ ] Advanced hybrid retrieval weighting
* [ ] Query rewriting
* [ ] Parent-child retrieval
* [ ] Chunk-level observability
* [ ] OpenTelemetry tracing
* [ ] Prometheus metrics
* [ ] Redis-based caching
* [ ] Object-storage integration
* [ ] Alembic migrations
* [ ] Kubernetes deployment
* [ ] GPU inference configuration
* [ ] Automated retrieval benchmarking

---

# License

No license has currently been specified for this repository.

If this project is intended to be publicly reusable, add an appropriate open-source license before accepting external contributions.

---

# Author

**Naidi47**

GitHub:

https://github.com/Naidi47

Repository:

https://github.com/Naidi47/production-rag-platform

---

## Project Summary

Production RAG Platform provides an end-to-end foundation for document-grounded question answering:

```text
PDF
 │
 ▼
Ingestion
 │
 ▼
Chunking
 │
 ▼
Embeddings
 │
 ▼
PostgreSQL + pgvector
 │
 ├───────────────┐
 ▼               ▼
Semantic       Keyword
Search         Search
 │               │
 └───────┬───────┘
         ▼
        RRF
         │
         ▼
   Cross-Encoder
    Reranking
         │
         ▼
    Relevant
     Context
         │
         ▼
       LLM
         │
         ▼
 Answer + Citations
```

The project is intended as a foundation for building, evaluating, and extending production-oriented RAG applications.
