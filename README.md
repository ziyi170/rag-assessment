# RAG Assessment — LEC AI

A production-ready RAG system built with Python, FastAPI, PostgreSQL + pgvector, and Claude.

## Architecture
User Query

↓

FastAPI /query endpoint

↓

Embed query (OpenAI text-embedding-3-small via OpenRouter)

↓

pgvector cosine similarity search (top-k documents)

↓

Claude claude-opus-4-6 answer generation with retrieved context

↓

Structured response with answer + sources + similarity scores

## Dataset

100 arXiv papers on RAG, LLM retrieval, and semantic search.
Downloaded via the arXiv API, filtered for withdrawn/retracted papers.
Embedded with `text-embedding-3-small` (1536 dimensions), stored in PostgreSQL with pgvector.

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| API | FastAPI | Async, type-safe, auto-docs |
| Vector DB | PostgreSQL + pgvector | Production-grade, SQL joins possible, scales horizontally |
| Embeddings | text-embedding-3-small | 1536-dim, strong semantic quality, cost-efficient |
| LLM | Claude claude-opus-4-6| Required by LEC AI stack |
| Search | ivfflat cosine similarity | Sub-millisecond at 100 docs, scalable to millions |

## Why pgvector over FAISS?

FAISS is fast but in-memory only — restarts lose all data. pgvector persists vectors in PostgreSQL alongside structured metadata, enabling SQL queries like filtering by date or author. For production RAG, this matters: you can combine semantic similarity with structured filters in a single query.

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 16+
- pgvector extension

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

# 5. Create table and download dataset
psql -U postgres -d rag_assessment < schema.sql
python3 ingest_data.py
python3 embed_and_index.py

# 6. Start API
python3 -m uvicorn main:app --reload --port 8001
```

### Environment Variables
DATABASE_URL=postgresql://postgres:postgres@localhost/rag_assessment

ANTHROPIC_API_KEY=your_claude_key

OPENAI_API_KEY=your_openrouter_key

## API Endpoints

### POST /query — Full RAG pipeline
```bash
curl -X POST http://localhost:8001/query \
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

| Metric | Score |
|---|---|
| Mean Precision@5 | 0.280 |
| Mean MRR | 0.617 |
| Mean NDCG@5 | 0.686 |

### Per-query breakdown

| Query | Precision@5 | MRR | NDCG@5 |
|---|---|---|---|
| How does RAG improve LLM accuracy? | 0.400 | 1.000 | 0.850 |
| Challenges in multi-hop QA with RAG? | 0.400 | 0.500 | 0.651 |
| Dense retrieval vector embeddings | 0.200 | 0.333 | 0.500 |
| Autoregressive image retrieval | 0.200 | 1.000 | 1.000 |
| Enterprise document QA | 0.200 | 0.250 | 0.431 |

### Analysis

MRR of 0.617 means the first relevant paper appears on average at rank 1.6 — the system reliably surfaces at least one good result near the top. NDCG of 0.686 confirms the overall ranking quality is strong.

Precision@5 of 0.280 reflects that the test queries were deliberately specific with narrow ground truth sets (1-3 relevant papers out of 100). In practice, broader queries score higher.

The weakest query is "enterprise document QA" — the relevant paper (T-RAG) ranked 4th because its abstract focuses on implementation details rather than the query terms. A hybrid search combining BM25 keyword matching with dense retrieval would improve this case.

## Roadmap

- Hybrid search: BM25 + dense retrieval combined
- Re-ranking with cross-encoder model
- Full paper ingestion via arXiv LaTeX source
- Redis-backed session memory for multi-turn QA
- RAGAS evaluation pipeline (faithfulness, answer relevance)
