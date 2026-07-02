"""
RAG Evaluation — Before vs After Reranking
Metrics: Precision@K, MRR, NDCG
"""

import asyncio
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retrieval.search import semantic_search
from retrieval.reranker import rerank

TEST_QUERIES = [
    {
        "query": "How does retrieval augmented generation improve LLM accuracy?",
        "relevant_titles": [
            "Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers",
            "T-RAG: Lessons from the LLM Trenches",
            "Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration"
        ]
    },
    {
        "query": "What are challenges in multi-hop question answering with RAG?",
        "relevant_titles": [
            "Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration",
            "Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers"
        ]
    },
    {
        "query": "dense retrieval vector embeddings semantic search",
        "relevant_titles": [
            "Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers",
            "Generative Retrieval as Dense Retrieval"
        ]
    },
    {
        "query": "autoregressive generation image retrieval augmentation",
        "relevant_titles": [
            "AR-RAG: Autoregressive Retrieval Augmentation for Image Generation"
        ]
    },
    {
        "query": "enterprise document question answering private knowledge base",
        "relevant_titles": [
            "T-RAG: Lessons from the LLM Trenches"
        ]
    }
]


def precision_at_k(retrieved_titles, relevant_titles, k):
    top_k = retrieved_titles[:k]
    hits = sum(1 for t in top_k if any(
        rel.lower() in t.lower() or t.lower() in rel.lower()
        for rel in relevant_titles))
    return hits / k


def mrr(retrieved_titles, relevant_titles):
    for i, title in enumerate(retrieved_titles):
        if any(rel.lower() in title.lower() or title.lower() in rel.lower()
               for rel in relevant_titles):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_titles, relevant_titles, k):
    def rel(title):
        return 1 if any(r.lower() in title.lower() or title.lower() in r.lower()
                        for r in relevant_titles) else 0
    dcg = sum(rel(retrieved_titles[i]) / math.log2(i + 2)
              for i in range(min(k, len(retrieved_titles))))
    ideal = sorted([rel(t) for t in retrieved_titles], reverse=True)
    idcg = sum(ideal[i] / math.log2(i + 2)
               for i in range(min(k, len(ideal))))
    return dcg / idcg if idcg > 0 else 0.0


async def run_eval(use_reranker: bool, top_k: int = 5):
    all_p, all_mrr, all_ndcg = [], [], []

    for test in TEST_QUERIES:
        query = test["query"]
        relevant = test["relevant_titles"]

        if use_reranker:
            candidates = await semantic_search(query, top_k=20)
            results = rerank(query, candidates, top_k=top_k)
        else:
            results = await semantic_search(query, top_k=top_k)

        titles = [r["title"] for r in results]
        all_p.append(precision_at_k(titles, relevant, top_k))
        all_mrr.append(mrr(titles, relevant))
        all_ndcg.append(ndcg_at_k(titles, relevant, top_k))

    return {
        "precision": round(sum(all_p) / len(all_p), 3),
        "mrr": round(sum(all_mrr) / len(all_mrr), 3),
        "ndcg": round(sum(all_ndcg) / len(all_ndcg), 3)
    }


async def evaluate():
    print("Running evaluation (this may take a minute)...\n")

    print("Stage 1: Bi-encoder only (pgvector cosine)")
    before = await run_eval(use_reranker=False)
    print(f"  Precision@5: {before['precision']}  |  MRR: {before['mrr']}  |  NDCG@5: {before['ndcg']}\n")

    print("Stage 2: Two-stage (pgvector → cross-encoder reranker)")
    after = await run_eval(use_reranker=True)
    print(f"  Precision@5: {after['precision']}  |  MRR: {after['mrr']}  |  NDCG@5: {after['ndcg']}\n")

    print("="*55)
    print("IMPROVEMENT")
    print("="*55)
    print(f"  Precision@5: {before['precision']} → {after['precision']}  ({'+' if after['precision'] >= before['precision'] else ''}{round((after['precision']-before['precision'])*100, 1)}%)")
    print(f"  MRR:         {before['mrr']} → {after['mrr']}  ({'+' if after['mrr'] >= before['mrr'] else ''}{round((after['mrr']-before['mrr'])*100, 1)}%)")
    print(f"  NDCG@5:      {before['ndcg']} → {after['ndcg']}  ({'+' if after['ndcg'] >= before['ndcg'] else ''}{round((after['ndcg']-before['ndcg'])*100, 1)}%)")

    return before, after


if __name__ == "__main__":
    asyncio.run(evaluate())
