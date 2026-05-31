"""
One-time script to index all Medium articles into Pinecone.

Usage:
    cd medium-rag
    python scripts/index_articles.py

Prerequisites:
    pip install -r requirements-dev.txt
    Copy .env.example to .env and fill in all credentials.

The CSV file is read from ../medium-english-50mb.csv relative to this script.
Upserts are idempotent — safe to re-run after a crash.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time
from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

from lib.rag_config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMS,
    CHUNK_SIZE,
    STRIDE,
    OVERLAP_TOKENS,
    EMBED_BATCH_SIZE,
    PINECONE_BATCH_SIZE,
    COST_PER_TOKEN,
)
from lib.chunker import chunk_text
from lib.embedder import get_embeddings_batch

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "medium-english-50mb.csv"
)


def ensure_index(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Waiting for index '{PINECONE_INDEX_NAME}' to become ready...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)
        print(f"Index created: {PINECONE_INDEX_NAME}")
    else:
        print(f"Index already exists: {PINECONE_INDEX_NAME}")
    return pc.Index(PINECONE_INDEX_NAME)


def build_chunk_records(row_idx: int, row: dict, total_chunks_for_row: int) -> list[dict]:
    """Convert one CSV row into a list of chunk records ready for embedding."""
    chunks = chunk_text(row.get("text", ""))
    records = []
    total = len(chunks)
    for chunk in chunks:
        vector_id = f"article_{row_idx}_chunk_{chunk['chunk_index']}"
        records.append({
            "id": vector_id,
            "text": chunk["text"],
            "metadata": {
                "article_id": str(row_idx),
                "title": (row.get("title") or "")[:500],
                "url": (row.get("url") or "")[:500],
                "authors": (row.get("authors") or "")[:300],
                "timestamp": (row.get("timestamp") or "")[:50],
                "tags": (row.get("tags") or "")[:300],
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["text"][:3000],
                "total_chunks": total,
            },
        })
    return records


def flush_batch(index, records: list[dict], stats: dict):
    """Embed and upsert a batch of chunk records."""
    texts = [r["text"] for r in records]
    embeddings = get_embeddings_batch(texts)

    # rough token estimate (4 chars ≈ 1 token)
    token_estimate = sum(len(t) // 4 for t in texts)
    stats["tokens"] += token_estimate
    stats["cost"] += token_estimate * COST_PER_TOKEN

    vectors = [
        (rec["id"], emb, rec["metadata"])
        for rec, emb in zip(records, embeddings)
    ]

    for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
        sub = vectors[i : i + PINECONE_BATCH_SIZE]
        index.upsert(vectors=sub)
        stats["vectors"] += len(sub)


def main():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = ensure_index(pc)

    print(f"Reading CSV from: {os.path.abspath(CSV_PATH)}")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Total articles: {len(rows)}")
    print(f"Chunk size: {CHUNK_SIZE} tokens, stride: {STRIDE}, overlap: {OVERLAP_TOKENS}")
    print()

    pending: list[dict] = []
    stats = {"vectors": 0, "tokens": 0, "cost": 0.0}

    for row_idx, row in enumerate(tqdm(rows, desc="Indexing")):
        text = row.get("text", "")
        if not text or not text.strip():
            continue

        records = build_chunk_records(row_idx, row, 0)
        pending.extend(records)

        while len(pending) >= EMBED_BATCH_SIZE:
            batch = pending[:EMBED_BATCH_SIZE]
            pending = pending[EMBED_BATCH_SIZE:]
            flush_batch(index, batch, stats)

    if pending:
        flush_batch(index, pending, stats)

    print()
    print("=" * 50)
    print(f"Total vectors upserted : {stats['vectors']:,}")
    print(f"Estimated tokens used  : {stats['tokens']:,}")
    print(f"Estimated cost         : ${stats['cost']:.4f}")
    print("=" * 50)
    print("Indexing complete. Run scripts/verify_index.py to confirm.")


if __name__ == "__main__":
    main()
