"""
Post-indexing sanity check.

Usage:
    cd medium-rag
    python scripts/verify_index.py

Confirms total vector count and runs 4 sample queries (one per required query type).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone
from lib.rag_config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from lib.embedder import get_embedding
from lib.pinecone_client import query_index

SAMPLE_QUERIES = [
    # (label, question)
    ("Precise fact retrieval", "Which article explains how neural networks learn?"),
    ("Multi-result topic listing", "Articles about productivity and time management"),
    ("Key idea summary", "What are the main ideas about machine learning in production?"),
    ("Recommendation", "Recommend an article about startup fundraising"),
]


def main():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    stats = index.describe_index_stats()
    total = stats.total_vector_count
    print(f"Total vectors in index: {total:,}")

    if total == 0:
        print("WARNING: Index is empty. Run scripts/index_articles.py first.")
        return

    print()
    for label, question in SAMPLE_QUERIES:
        print(f"--- {label} ---")
        print(f"Q: {question}")
        vec = get_embedding(question)
        matches = query_index(vec, top_k=3)
        for i, m in enumerate(matches, 1):
            title = m.metadata.get("title", "No title")[:70]
            print(f"  {i}. [{m.score:.4f}] {title}")
        print()

    print("Verification complete.")


if __name__ == "__main__":
    main()
