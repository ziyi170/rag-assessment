"""
Step 2: Embed abstracts and store in pgvector.
Uses OpenRouter for embeddings, stores in PostgreSQL.
"""

import os
import json
import asyncio
import psycopg2
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

DB_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = "openai/text-embedding-3-small"


async def embed_text(text: str) -> list[float]:
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000]  # truncate to avoid token limits
    )
    return response.data[0].embedding


async def index_documents():
    with open("dataset.json") as f:
        docs = json.load(f)

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    print(f"Indexing {len(docs)} documents...")
    success = 0

    for i, doc in enumerate(docs):
        try:
            # Embed title + abstract for richer representation
            text = f"{doc['title']}\n\n{doc['abstract']}"
            embedding = await embed_text(text)

            cur.execute("""
                INSERT INTO documents 
                    (paper_id, title, abstract, authors, published, url, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (paper_id) DO NOTHING
            """, (
                doc["id"],
                doc["title"],
                doc["abstract"],
                doc.get("authors", []),
                doc.get("published"),
                doc.get("url"),
                embedding
            ))
            conn.commit()
            success += 1

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(docs)} done...")

            await asyncio.sleep(0.3)  # rate limit

        except Exception as e:
            print(f"  Error on {doc['title'][:40]}: {e}")
            conn.rollback()

    cur.close()
    conn.close()
    print(f"✅ Indexed {success}/{len(docs)} documents")


if __name__ == "__main__":
    asyncio.run(index_documents())
