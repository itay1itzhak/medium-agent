"""
Stage 2 test: embedding API — requires OPENAI_API_KEY and OPENAI_BASE_URL.

Usage:
    cd medium-rag
    python scripts/test_embedder.py

Shows: vector dimensions, norm, and cosine similarity between sentence pairs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from dotenv import load_dotenv
load_dotenv()

from lib.embedder import get_embeddings_batch
from lib.rag_config import EMBEDDING_MODEL, EMBEDDING_DIMS

SENTENCES = [
    "Neural networks learn by adjusting their weights through backpropagation.",
    "Deep learning models are trained with gradient descent on large datasets.",
    "The best way to cook pasta is to use plenty of salted boiling water.",
]


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def vector_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def main():
    print("=" * 60)
    print("STAGE 2: EMBEDDER")
    print("=" * 60)
    print(f"\nModel  : {EMBEDDING_MODEL}")
    print(f"Dims   : {EMBEDDING_DIMS}")
    print(f"Calling API with {len(SENTENCES)} sentences in one batch...\n")

    embeddings = get_embeddings_batch(SENTENCES)

    for i, (sent, emb) in enumerate(zip(SENTENCES, embeddings)):
        norm = vector_norm(emb)
        print(f"  [{i}] \"{sent[:65]}\"")
        print(f"       dims={len(emb)}  norm={norm:.6f}")
        print(f"       first 5 values: {[round(x, 5) for x in emb[:5]]}")
        print()

    print("Cosine similarities:")
    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            sim = cosine_sim(embeddings[i], embeddings[j])
            label = "(similar)" if sim > 0.80 else "(dissimilar)"
            print(f"  [{i}] vs [{j}]  sim={sim:.4f}  {label}")
            print(f"       \"{SENTENCES[i][:45]}...\"")
            print(f"       \"{SENTENCES[j][:45]}...\"")
            print()

    print("Embedder OK.")


if __name__ == "__main__":
    main()
