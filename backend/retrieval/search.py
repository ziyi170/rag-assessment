"""
Semantic search using pgvector cosine similarity.
"""

import os
import asyncio
import psycopg2
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

DB_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = "openai/text-embedding-3-small"


async def embed_query(text: str) -> list[float]:
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


async def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed query and retrieve top-k most similar documents from pgvector.
    Uses cosine similarity via <=> operator.
    """
    query_embedding = await embed_query(query)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            paper_id,
            title,
            abstract,
            authors,
            published::text,
            url,
            1 - (embedding <=> %s::vector) as similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (embedding_str, embedding_str, top_k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "paper_id": row[0],
            "title": row[1],
            "abstract": row[2],
            "authors": row[3],
            "published": row[4],
            "url": row[5],
            "similarity": round(float(row[6]), 4)
        })

    return results


if __name__ == "__main__":
    async def test():
        results = await semantic_search("retrieval augmented generation", top_k=3)
        for r in results:
            print(f"[{r['similarity']}] {r['title'][:60]}")

    asyncio.run(test())
