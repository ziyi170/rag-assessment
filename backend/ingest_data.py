"""
Step 1: Download 100 arXiv papers on RAG and LLM retrieval.
Saves to dataset.json for later embedding and indexing.
"""

import arxiv
import json
import time

QUERIES = [
    "retrieval augmented generation language model",
    "RAG LLM question answering",
    "dense retrieval transformer embeddings",
    "knowledge grounded generation NLP",
    "semantic search vector embeddings"
]

WITHDRAWN_SIGNALS = [
    "this paper has been withdrawn",
    "this paper has been retracted",
    "duplicate of arxiv",
    "submitted under a pseudonym",
    "administratively withdrawn"
]

def download_papers(target=100):
    client = arxiv.Client()
    documents = []
    seen_ids = set()

    for query in QUERIES:
        if len(documents) >= target:
            break
        print(f"Searching: {query}")
        search = arxiv.Search(
            query=query,
            max_results=30,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for paper in client.results(search):
            if len(documents) >= target:
                break
            if paper.entry_id in seen_ids:
                continue
            abstract_lower = paper.summary.lower()
            if any(s in abstract_lower for s in WITHDRAWN_SIGNALS):
                continue
            documents.append({
                "id": paper.entry_id,
                "title": paper.title,
                "abstract": paper.summary,
                "authors": [a.name for a in paper.authors[:3]],
                "published": str(paper.published.date()),
                "url": paper.entry_id
            })
            seen_ids.add(paper.entry_id)
        time.sleep(2)

    print(f"✅ Downloaded {len(documents)} papers")
    return documents

if __name__ == "__main__":
    docs = download_papers(100)
    with open("dataset.json", "w") as f:
        json.dump(docs, f, indent=2)
    print("✅ Saved to dataset.json")
    print(f"Sample: {docs[0]['title']}")
