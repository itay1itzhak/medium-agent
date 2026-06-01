"""
Stage 4 test: full end-to-end pipeline (embed → retrieve → LLM → response).
Requires all credentials and a populated index.

Usage:
    cd medium-rag
    python scripts/test_pipeline.py [question]

If a question is passed as a command-line argument it runs that single question.
Otherwise it runs the four canonical test questions.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import textwrap
from dotenv import load_dotenv
load_dotenv(override=True)

from lib.embedder import get_embedding
from lib.pinecone_client import query_index
from lib.prompt_builder import (
    build_system_prompt,
    build_user_prompt,
    build_context_str,
    call_llm,
)
from lib.rag_config import TOP_K, EMBEDDING_MODEL, LLM_MODEL

DEFAULT_QUESTIONS = [
    "Which article explains how the attention mechanism works in transformers?",
    "List up to 3 articles about productivity and time management.",
    "Summarize the key ideas about building machine learning models in production.",
    "Recommend an article about startup fundraising and explain why.",
]

WRAP = 72  # text wrap width for display


def section(title: str):
    print(f"\n  ┌─ {title} {'─' * max(0, 50 - len(title))}┐")


def run_question(question: str, index: int):
    print("\n" + "=" * 60)
    print(f"  QUESTION {index}: {question}")
    print("=" * 60)

    # ── Stage A: Embed ────────────────────────────────────────────
    section("A. Embed question")
    t0 = time.perf_counter()
    query_vector = get_embedding(question)
    t_embed = time.perf_counter() - t0
    print(f"  Model : {EMBEDDING_MODEL}")
    print(f"  Dims  : {len(query_vector)}")
    print(f"  First 5 values: {[round(x, 5) for x in query_vector[:5]]}")
    print(f"  Time  : {t_embed*1000:.0f} ms")

    # ── Stage B: Retrieve ─────────────────────────────────────────
    section("B. Retrieve from Pinecone")
    t0 = time.perf_counter()
    matches = query_index(query_vector, top_k=TOP_K)
    t_retrieve = time.perf_counter() - t0
    print(f"  Top-k : {TOP_K}   retrieved: {len(matches)}")
    print(f"  Time  : {t_retrieve*1000:.0f} ms\n")
    for i, m in enumerate(matches, 1):
        title = m.metadata.get("title", "?")[:55]
        chunk_i = m.metadata.get("chunk_index", "?")
        print(f"  {i:>2}. score={m.score:.4f}  chunk={chunk_i}  \"{title}\"")

    # ── Stage C: Build prompts ────────────────────────────────────
    section("C. Build augmented prompt")
    context_str  = build_context_str(matches)
    system_prompt = build_system_prompt()
    user_prompt   = build_user_prompt(question, context_str)

    sys_tokens  = len(system_prompt.split())   # rough word count as proxy
    user_tokens = len(user_prompt.split())
    print(f"  System prompt : ~{sys_tokens} words")
    print(f"  User prompt   : ~{user_tokens} words  "
          f"({len(context_str)} chars of context + question)")
    # Show a truncated preview of the user prompt
    preview = user_prompt[:300].replace("\n", " ").strip()
    print(f"\n  User prompt preview:\n  \"{preview}...\"")

    # ── Stage D: LLM call ─────────────────────────────────────────
    section("D. LLM call")
    print(f"  Model : {LLM_MODEL}")
    t0 = time.perf_counter()
    response = call_llm(system_prompt, user_prompt)
    t_llm = time.perf_counter() - t0
    print(f"  Time  : {t_llm*1000:.0f} ms\n")

    # Wrap the response for readability
    for line in textwrap.wrap(response, width=WRAP):
        print(f"  {line}")

    # ── Summary ───────────────────────────────────────────────────
    total_ms = (t_embed + t_retrieve + t_llm) * 1000
    print(f"\n  Timing: embed={t_embed*1000:.0f}ms  "
          f"retrieve={t_retrieve*1000:.0f}ms  "
          f"llm={t_llm*1000:.0f}ms  "
          f"total={total_ms:.0f}ms")


def main():
    questions = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

    print("=" * 60)
    print("STAGE 4: FULL PIPELINE")
    print("=" * 60)
    print(f"\nRunning {len(questions)} question(s)...\n")

    for i, q in enumerate(questions, 1):
        run_question(q, i)

    print("\n\n" + "=" * 60)
    print("Pipeline OK.")
    print("=" * 60)


if __name__ == "__main__":
    main()
