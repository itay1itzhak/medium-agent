"""
Stage 3 test: retrieval pipeline (embed query → Pinecone search).
Requires all credentials and a populated index.

Usage:
    cd medium-rag
    python scripts/test_retrieval.py

Tests one query per required query type and shows scored results.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dotenv import load_dotenv
load_dotenv(override=True)

from lib.embedder import get_embedding
from lib.pinecone_client import query_index
from lib.rag_config import TOP_K, PINECONE_INDEX_NAME
from pinecone import Pinecone
from lib.rag_config import PINECONE_API_KEY

QUERIES = [
    ("1 — Precise fact retrieval",
     "Which article explains how the attention mechanism works in transformers?"),
    ("2 — Multi-result topic listing",
     "Articles about productivity tips and time management"),
    ("3 — Key idea summary",
     "What are the main ideas about building machine learning models in production?"),
    ("4 — Recommendation with evidence",
     "Recommend an article about startup fundraising or venture capital"),
]


def divider(label: str):
    print("\n" + "─" * 60)
    print(f"  Query type {label}")
    print("─" * 60)


def main():
    # Show index stats first
    pc = Pinecone(api_key=PINECONE_API_KEY)
    stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
    total = stats.total_vector_count

    print("=" * 60)
    print("STAGE 3: RETRIEVAL")
    print("=" * 60)
    print(f"\nIndex  : {PINECONE_INDEX_NAME}")
    print(f"Vectors: {total:,}")
    print(f"Top-k  : {TOP_K}")

    if total == 0:
        print("\nERROR: Index is empty. Run scripts/index_articles.py first.")
        return

    for label, question in QUERIES:
        divider(label)
        print(f"  Q: {question}\n")

        t0 = time.perf_counter()
        vec = get_embedding(question)
        t_embed = time.perf_counter() - t0

        t0 = time.perf_counter()
        matches = query_index(vec, top_k=TOP_K)
        t_query = time.perf_counter() - t0

        print(f"  Embed: {t_embed*1000:.0f}ms   Pinecone query: {t_query*1000:.0f}ms\n")
        print(f"  {'#':<3} {'Score':<8} {'Chunk':<6} {'Title'}")
        print(f"  {'─'*3} {'─'*7} {'─'*5} {'─'*40}")

        seen_titles = set()
        for rank, m in enumerate(matches, 1):
            meta = m.metadata
            title = meta.get("title", "Unknown")[:50]
            chunk_idx = meta.get("chunk_index", "?")
            score = m.score
            print(f"  {rank:<3} {score:<8.4f} {chunk_idx:<6} {title}")

            # Print first 120 chars of chunk text for top 3
            if rank <= 3:
                snippet = meta.get("chunk_text", "")[:120].replace("\n", " ").strip()
                print(f"      \"{snippet}...\"")

            seen_titles.add(title)

        unique = len(seen_titles)
        print(f"\n  Unique articles in top-{TOP_K}: {unique}")

    print("\n\nRetrieval OK.")


if __name__ == "__main__":
    main()
