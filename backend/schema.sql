CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    paper_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT[],
    published DATE,
    url TEXT,
    embedding vector(1536)
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);
