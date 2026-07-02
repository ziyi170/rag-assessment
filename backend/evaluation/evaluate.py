"""
RAG Evaluation Script
Metrics: Precision@K, MRR, NDCG
5 test queries with expected relevant papers
"""

import asyncio
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retrieval.search import semantic_search

# 5 test queries with known relevant paper titles
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


def precision_at_k(retrieved_titles: list[str], relevant_titles: list[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    top_k = retrieved_titles[:k]
    hits = sum(1 for t in top_k if any(rel.lower() in t.lower() or t.lower() in rel.lower() 
                                        for rel in relevant_titles))
    return hits / k


def mrr(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    """Mean Reciprocal Rank — rank of first relevant result."""
    for i, title in enumerate(retrieved_titles):
        if any(rel.lower() in title.lower() or title.lower() in rel.lower() 
               for rel in relevant_titles):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_titles: list[str], relevant_titles: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    def relevance(title):
        return 1 if any(rel.lower() in title.lower() or title.lower() in rel.lower() 
                       for rel in relevant_titles) else 0

    dcg = sum(relevance(retrieved_titles[i]) / math.log2(i + 2) 
              for i in range(min(k, len(retrieved_titles))))

    ideal = sorted([relevance(t) for t in retrieved_titles], reverse=True)
    idcg = sum(ideal[i] / math.log2(i + 2) 
               for i in range(min(k, len(ideal))))

    return dcg / idcg if idcg > 0 else 0.0


async def evaluate(top_k: int = 5):
    print(f"{'='*60}")
    print(f"RAG Evaluation — top_k={top_k}")
    print(f"{'='*60}\n")

    all_p_at_k = []
    all_mrr = []
    all_ndcg = []

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        relevant = test["relevant_titles"]

        results = await semantic_search(query, top_k=top_k)
        retrieved_titles = [r["title"] for r in results]

        p = precision_at_k(retrieved_titles, relevant, top_k)
        m = mrr(retrieved_titles, relevant)
        n = ndcg_at_k(retrieved_titles, relevant, top_k)

        all_p_at_k.append(p)
        all_mrr.append(m)
        all_ndcg.append(n)

        print(f"Query {i+1}: {query[:55]}...")
        print(f"  Precision@{top_k}: {p:.3f}  |  MRR: {m:.3f}  |  NDCG@{top_k}: {n:.3f}")
        print(f"  Retrieved:")
        for j, r in enumerate(results):
            marker = "✅" if any(rel.lower() in r['title'].lower() or r['title'].lower() in rel.lower() 
                                  for rel in relevant) else "  "
            print(f"    {marker} [{r['similarity']:.3f}] {r['title'][:55]}")
        print()

    print(f"{'='*60}")
    print(f"AGGREGATE RESULTS (n={len(TEST_QUERIES)} queries, k={top_k})")
    print(f"{'='*60}")
    print(f"  Mean Precision@{top_k}: {sum(all_p_at_k)/len(all_p_at_k):.3f}")
    print(f"  Mean MRR:          {sum(all_mrr)/len(all_mrr):.3f}")
    print(f"  Mean NDCG@{top_k}:     {sum(all_ndcg)/len(all_ndcg):.3f}")


if __name__ == "__main__":
    asyncio.run(evaluate(top_k=5))
