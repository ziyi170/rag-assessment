"""
Cross-encoder reranker for two-stage retrieval.
Stage 1: pgvector cosine similarity (fast, top-20 candidates)
Stage 2: cross-encoder reranking (precise, top-k final)
"""

from sentence_transformers import CrossEncoder

# Lightweight but strong cross-encoder, no API cost
MODEL_NAME = "BAAI/bge-reranker-base"
_model = None


def get_reranker():
    global _model
    if _model is None:
        print(f"Loading reranker model: {MODEL_NAME}")
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates using cross-encoder.
    Input: query + list of docs with 'abstract' and 'title'
    Output: top_k docs sorted by cross-encoder score
    """
    model = get_reranker()

    # Cross-encoder scores query against title + abstract together
    pairs = [(query, f"{doc['title']}\n\n{doc['abstract']}") for doc in candidates]
    scores = model.predict(pairs)

    # Attach reranker score and sort
    for doc, score in zip(candidates, scores):
        doc["reranker_score"] = round(float(score), 4)

    ranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)
    return ranked[:top_k]
