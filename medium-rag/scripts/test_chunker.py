"""
Stage 1 test: chunker only — no API calls, no credentials needed.

Usage:
    cd medium-rag
    python scripts/test_chunker.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import textwrap
import tiktoken
from lib.chunker import chunk_text, get_tokenizer
from lib.rag_config import CHUNK_SIZE, OVERLAP_TOKENS, STRIDE

# ── Toy article ────────────────────────────────────────────────────────────────
TOY_ARTICLE = """
Machine learning has transformed how we build software. At its core, a neural network
is a function that maps inputs to outputs through layers of learned parameters. Each
layer performs a linear transformation followed by a non-linear activation, allowing
the network to approximate arbitrarily complex functions given enough data and compute.

Training a neural network means finding the parameters that minimize a loss function.
We do this with gradient descent: compute how much each parameter contributed to the
error, then nudge it in the opposite direction. The backpropagation algorithm makes
this efficient by reusing intermediate computations through the chain rule of calculus.

Attention mechanisms changed everything. Instead of compressing the entire input into
a fixed-size vector, attention lets the model look back at all previous tokens when
predicting the next one. Transformers stack many attention heads in parallel, each
learning different relationships. This design scales beautifully with more data and
compute, which is why large language models keep improving.

The practical challenge is inference speed. At deployment time, every token must be
generated one at a time, and the key-value cache grows with sequence length. Techniques
like quantization and speculative decoding make production deployment feasible without
sacrificing too much quality.

Data quality matters more than data quantity. A model trained on clean, diverse text
will consistently outperform one trained on ten times as much noisy data. This is why
careful curation — filtering, deduplication, and quality scoring — is now considered
a core part of the training pipeline.
""".strip()


def cosine_overlap(enc, tokens_a: list, tokens_b: list) -> float:
    """Count shared tokens at the tail of a and head of b (overlap window)."""
    tail = tokens_a[-OVERLAP_TOKENS:]
    head = tokens_b[:OVERLAP_TOKENS]
    shared = len(set(tail) & set(head))
    return shared / OVERLAP_TOKENS if OVERLAP_TOKENS else 0.0


def main():
    enc = get_tokenizer()
    total_tokens = len(enc.encode(TOY_ARTICLE))

    print("=" * 60)
    print("STAGE 1: CHUNKER")
    print("=" * 60)
    print(f"\nToy article: {len(TOY_ARTICLE)} chars, {total_tokens} tokens")
    print(f"Config: chunk_size={CHUNK_SIZE}, overlap={OVERLAP_TOKENS}, stride={STRIDE}")

    chunks = chunk_text(TOY_ARTICLE)
    print(f"\nProduced {len(chunks)} chunk(s)\n")

    for ch in chunks:
        tokens = enc.encode(ch["text"])
        print(f"  Chunk {ch['chunk_index']}  |  {len(tokens)} tokens  "
              f"|  token_start={ch['token_start']}")
        # Show first and last ~60 chars
        first = ch["text"][:60].replace("\n", " ")
        last  = ch["text"][-60:].replace("\n", " ")
        print(f"    start: \"{first}...\"")
        print(f"    end:   \"...{last}\"")
        print()

    # Show overlap between consecutive chunks
    if len(chunks) > 1:
        print("Overlap check (shared tokens between consecutive chunk tails/heads):")
        for i in range(len(chunks) - 1):
            ta = enc.encode(chunks[i]["text"])
            tb = enc.encode(chunks[i + 1]["text"])
            tail_text = enc.decode(ta[-OVERLAP_TOKENS:]).replace("\n", " ").strip()
            head_text = enc.decode(tb[:OVERLAP_TOKENS]).replace("\n", " ").strip()
            print(f"  Chunk {i} tail : \"{tail_text[:70]}\"")
            print(f"  Chunk {i+1} head: \"{head_text[:70]}\"")
            print()

    print("Chunker OK.")


if __name__ == "__main__":
    main()
