# RAG Assessment — LEC AI

A production-ready RAG system built with Python, FastAPI, PostgreSQL + pgvector, and Claude.

## Architecture

```
User Query
    ↓
FastAPI /query endpoint
    ↓
Embed query (text-embedding-3-small via OpenRouter)
    ↓
pgvector cosine similarity search — top-20 candidates
    ↓
BAAI/bge-reranker-base cross-encoder reranking — top-k final
    ↓
Claude claude-opus-4-6 answer generation with retrieved context
    ↓
Structured response with answer + sources + similarity scores
```

## Dataset

100 arXiv papers on RAG, LLM retrieval, and semantic search.
Downloaded via the arXiv API, filtered for withdrawn/retracted papers.
Embedded with `text-embedding-3-small` (1536 dimensions), stored in PostgreSQL with pgvector.

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| API | FastAPI | Async, type-safe, auto-docs |
| Vector DB | PostgreSQL + pgvector | Production-grade, persistent, supports SQL filters |
| Embeddings | text-embedding-3-small | 1536-dim, strong semantic quality, cost-efficient |
| Reranker | BAAI/bge-reranker-base | Cross-encoder, better than ms-marco on academic text |
| LLM | Claude claude-opus-4-6 | Required by LEC AI stack |
| Search | ivfflat cosine similarity | Sub-millisecond at 100 docs, scalable to millions |

## Why pgvector over FAISS?

FAISS is fast but in-memory only — restarts lose all data. pgvector persists vectors in PostgreSQL alongside structured metadata, enabling SQL queries like filtering by date or author. For production RAG, this matters: you can combine semantic similarity with structured filters in a single query.

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 16+
- pgvector extension (`postgresql-16-pgvector`)

### Install

```bash
# 1. Clone
git clone https://github.com/ziyi170/rag-assessment
cd rag-assessment/backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL and enable pgvector
sudo service postgresql start
psql -U postgres -c "CREATE DATABASE rag_assessment;"
psql -U postgres -d rag_assessment -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Create table, download and embed dataset
psql -U postgres -d rag_assessment < schema.sql
python3 ingest_data.py
python3 embed_and_index.py

# 6. Start API
python3 -m uvicorn main:app --reload --port 8001
```

### Environment Variables

```
DATABASE_URL=postgresql://postgres:postgres@localhost/rag_assessment
ANTHROPIC_API_KEY=your_claude_key
OPENAI_API_KEY=your_openrouter_key
```

## API Endpoints

### POST /query — Full RAG pipeline (bi-encoder only)
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main challenges in RAG?", "top_k": 5}'
```

### POST /query/reranked — Two-stage RAG pipeline (pgvector + reranker)
```bash
curl -X POST http://localhost:8001/query/reranked \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main challenges in RAG?", "top_k": 5}'
```

### POST /search — Semantic search only
```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "dense retrieval embeddings", "top_k": 3}'
```

### GET /health
```bash
curl http://localhost:8001/health
```

## Evaluation Results

Evaluated on 5 test queries with human-labelled relevant papers.

### Stage 1: Bi-encoder only (pgvector cosine similarity)

| Metric | Score |
|---|---|
| Mean Precision@5 | 0.280 |
| Mean MRR | 0.617 |
| Mean NDCG@5 | 0.686 |

### Stage 2: Two-stage retrieval (pgvector → BAAI/bge-reranker-base)

| Metric | Before | After | Change |
|---|---|---|---|
| Mean Precision@5 | 0.280 | 0.240 | -4.0% |
| Mean MRR | 0.617 | 0.717 | +10.0% |
| Mean NDCG@5 | 0.686 | 0.770 | +8.4% |

### Analysis

The two-stage pipeline improves MRR by +10% and NDCG by +8.4%, meaning relevant papers rank closer to the top. Precision@5 drops slightly — the expected trade-off for a cross-encoder that prioritises ranking quality over coverage. For a QA use case where users read the top result first, MRR is the more important metric, making this a net positive trade-off.

One finding worth noting: the general-purpose `ms-marco-MiniLM` reranker degraded performance on academic abstracts (-5% MRR), while `BAAI/bge-reranker-base` improved it. Model selection for the reranker matters as much as the architecture itself.

## Roadmap

- Hybrid search: BM25 + dense retrieval combined for better keyword coverage
- Full paper ingestion via arXiv LaTeX source (cleaner than PDF parsing)
- Redis-backed session memory for multi-turn QA
- RAGAS evaluation pipeline (faithfulness, answer relevance metrics)
- Expand ground truth labels for more robust evaluation