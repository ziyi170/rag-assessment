"""
RAG Assessment API
FastAPI + pgvector + Claude
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from retrieval.search import semantic_search
from llm.claude import answer_with_claude

app = FastAPI(
    title="RAG Assessment API",
    description="Semantic search over 100 arXiv papers with Claude-powered QA",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    top_k: int


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/search")
async def search(request: QueryRequest):
    """Semantic search only — returns top-k documents without LLM generation."""
    try:
        results = await semantic_search(request.query, top_k=request.top_k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Full RAG pipeline — retrieve + generate answer with Claude."""
    try:
        # Retrieve relevant documents
        docs = await semantic_search(request.query, top_k=request.top_k)

        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        # Generate answer with Claude
        answer = answer_with_claude(request.query, docs)

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=[{
                "title": d["title"],
                "url": d["url"],
                "similarity": d["similarity"],
                "published": d["published"]
            } for d in docs],
            top_k=request.top_k
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


from retrieval.reranker import rerank


@app.post("/query/reranked")
async def query_reranked(request: QueryRequest):
    """
    Two-stage RAG pipeline:
    Stage 1: pgvector retrieves top-20 candidates
    Stage 2: cross-encoder reranks to top-k
    Stage 3: Claude generates answer
    """
    try:
        # Stage 1: retrieve more candidates
        candidates = await semantic_search(request.query, top_k=20)

        # Stage 2: rerank
        top_docs = rerank(request.query, candidates, top_k=request.top_k)

        # Stage 3: generate answer
        answer = answer_with_claude(request.query, top_docs)

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=[{
                "title": d["title"],
                "url": d["url"],
                "similarity": d.get("similarity", 0),
                "reranker_score": d.get("reranker_score", 0),
                "published": d["published"]
            } for d in top_docs],
            top_k=request.top_k
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
