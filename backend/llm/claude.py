"""
LLM answer generation using Claude API.
Retrieved documents are injected as context (RAG pattern).
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a research assistant specialising in AI and machine learning papers.
When given context from retrieved academic papers, use it to answer the user's question accurately.
Always cite the paper titles you draw from in your answer.
If the context doesn't contain enough information, say so clearly."""


def answer_with_claude(query: str, retrieved_docs: list[dict]) -> str:
    """
    Generate answer using Claude with retrieved documents as context.
    """
    context = "\n\n---\n\n".join([
        f"Paper: {doc['title']}\nPublished: {doc['published']}\n\n{doc['abstract']}"
        for doc in retrieved_docs
    ])

    user_message = f"""Context from retrieved papers:

{context}

---

Question: {query}

Please answer based on the papers above. Cite specific papers by title."""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": user_message}
        ],
        system=SYSTEM_PROMPT
    )

    return message.content[0].text


if __name__ == "__main__":
    # Quick test
    test_docs = [{"title": "Test RAG Paper", "published": "2024-01-01",
                  "abstract": "RAG combines retrieval with generation to improve LLM accuracy."}]
    print(answer_with_claude("What is RAG?", test_docs))
