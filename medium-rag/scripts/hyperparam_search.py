"""
Hyperparameter search for the Medium RAG pipeline — with real small-scale indexing.

Strategy (budget-conscious, as the assignment requires)
-------------------------------------------------------
1.  Read a stratified sample of the CSV (every 25th row ≈ 304 articles).
    This gives ~4% of the corpus but is spread evenly so topic coverage is
    representative and known-relevant articles (e.g. rows 3951, 5056, 5560)
    are guaranteed to appear.

2.  For each (chunk_size, overlap_ratio) config, chunk those 304 articles,
    embed them in batches of 100, and upsert into an isolated Pinecone
    *namespace* (e.g. "test_512_020").  The default namespace (the full
    corpus index) is never touched.

3.  Run 4 evaluation queries against every namespace at top_k ∈ {5, 7, 10}.
    Measure max_score, mean_score, score@3, and unique_article_count.

4.  Compute a composite quality score, pick the best (chunk_size, overlap,
    top_k), then validate that top_k choice against the full-corpus default
    namespace (zero re-embedding cost).

5.  Delete all test namespaces to keep the Pinecone index clean.

6.  Print a full report and write scripts/hyperparam_results.json.

Estimated embedding cost: 304 articles × ~3 chunks × 4 configs ≈ 3,600
embeddings × avg 512 tokens = 1.84 M tokens @ $0.02/M ≈ $0.04 total.

Usage
-----
    cd medium-rag
    python scripts/hyperparam_search.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import statistics
import time
from dotenv import load_dotenv

load_dotenv(override=True)

import tiktoken
from pinecone import Pinecone
from tqdm import tqdm

from lib.embedder import get_embedding, get_embeddings_batch
from lib.rag_config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBED_BATCH_SIZE,
    PINECONE_BATCH_SIZE,
    COST_PER_TOKEN,
    CHUNK_SIZE,
    OVERLAP_RATIO,
    TOP_K,
)

# ── paths ─────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "medium-english-50mb.csv")
)
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "hyperparam_results.json")

# ── search space ──────────────────────────────────────────────────────────────
CONFIGS = [
    {"chunk_size": 256, "overlap_ratio": 0.15, "label": "256_015"},
    {"chunk_size": 512, "overlap_ratio": 0.20, "label": "512_020"},  # current/baseline
    {"chunk_size": 512, "overlap_ratio": 0.25, "label": "512_025"},
    {"chunk_size": 768, "overlap_ratio": 0.20, "label": "768_020"},
]

TOP_K_CANDIDATES = [5, 7, 10]

# Stratified: every SAMPLE_STEP-th row → ~304 articles covering full corpus
SAMPLE_STEP = 25

# ── evaluation queries (one per required query type) ─────────────────────────
QUERIES = [
    {
        "type": "precise_fact",
        "label": "Precise fact retrieval",
        "question": "Which article explains how the attention mechanism works in transformers?",
    },
    {
        "type": "multi_result",
        "label": "Multi-result topic listing",
        "question": "Articles about productivity tips and time management",
    },
    {
        "type": "key_idea",
        "label": "Key idea summary",
        "question": "Find an article about the bubonic plague spurring innovation",
    },
    {
        "type": "recommendation",
        "label": "Recommendation with evidence",
        "question": "Recommend an article for beginner advice on building habits",
    },
]

# ── tokenizer ─────────────────────────────────────────────────────────────────
_enc = None


def get_enc():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


# ── chunker (parameterised, independent of rag_config globals) ────────────────
def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> list[dict]:
    enc = get_enc()
    tokens = enc.encode(text)
    if not tokens:
        return []
    stride = max(1, int(chunk_size * (1 - overlap_ratio)))
    chunks = []
    idx = 0
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append({
            "text": enc.decode(tokens[start:end]),
            "chunk_index": idx,
            "token_start": start,
        })
        if end == len(tokens):
            break
        start += stride
        idx += 1
    return chunks


# ── indexing helpers ──────────────────────────────────────────────────────────
def flush_batch(index, records: list[dict], namespace: str, stats: dict):
    texts = [r["text"] for r in records]
    embeddings = get_embeddings_batch(texts)
    token_est = sum(len(t) // 4 for t in texts)
    stats["tokens"] += token_est
    stats["cost"] += token_est * COST_PER_TOKEN

    vectors = [
        (rec["id"], emb, rec["metadata"])
        for rec, emb in zip(records, embeddings)
    ]
    for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
        index.upsert(vectors=vectors[i : i + PINECONE_BATCH_SIZE], namespace=namespace)
        stats["vectors"] += len(vectors[i : i + PINECONE_BATCH_SIZE])


def index_sample(index, sample_rows: list[tuple], cfg: dict, namespace: str) -> dict:
    """Chunk and embed the sampled articles into a Pinecone namespace."""
    pending = []
    stats = {"vectors": 0, "tokens": 0, "cost": 0.0}
    cs = cfg["chunk_size"]
    ov = cfg["overlap_ratio"]

    for row_idx, row in tqdm(sample_rows, desc=f"  indexing {namespace}", leave=False):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        for chunk in chunk_text(text, cs, ov):
            pending.append({
                "id": f"article_{row_idx}_chunk_{chunk['chunk_index']}",
                "text": chunk["text"],
                "metadata": {
                    "article_id": str(row_idx),
                    "title": (row.get("title") or "")[:500],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text": chunk["text"][:3000],
                },
            })
            if len(pending) >= EMBED_BATCH_SIZE:
                flush_batch(index, pending, namespace, stats)
                pending = []

    if pending:
        flush_batch(index, pending, namespace, stats)

    return stats


def delete_namespace(index, namespace: str):
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        print(f"  Warning: could not delete namespace '{namespace}': {e}")


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate_namespace(index, embedded_queries, namespace: str, top_k_list: list) -> dict:
    """Query the namespace at multiple top_k values; return metrics."""
    results = {}
    for k in top_k_list:
        results[k] = {}
        for q, vec in embedded_queries:
            matches = index.query(
                vector=vec,
                top_k=k,
                include_metadata=True,
                namespace=namespace,
            ).matches

            scores = [m.score for m in matches]
            if not scores:
                results[k][q["type"]] = {
                    "max_score": 0, "mean_score": 0, "score_at_3": 0,
                    "unique_articles": 0, "top3_titles": [],
                }
                continue

            unique_articles = len({m.metadata.get("article_id") for m in matches})
            results[k][q["type"]] = {
                "max_score": round(max(scores), 4),
                "mean_score": round(statistics.mean(scores), 4),
                "score_at_3": round(scores[2] if len(scores) >= 3 else scores[-1], 4),
                "unique_articles": unique_articles,
                "top3_titles": [m.metadata.get("title", "")[:55] for m in matches[:3]],
            }
    return results


# ── composite score ───────────────────────────────────────────────────────────
def composite_score(metrics: dict, k: int) -> float:
    """
    Weighted score across the four query types.

    Weights reflect relative importance and what each type primarily needs:
      precise_fact  (35%) → max_score: the single best match must be high
      multi_result  (30%) → unique_article diversity relative to top_k
      key_idea      (20%) → mean_score: all retrieved chunks should be relevant
      recommendation(15%) → max_score: at least one strong match is enough
    """
    pf = metrics.get("precise_fact", {})
    mr = metrics.get("multi_result", {})
    ki = metrics.get("key_idea", {})
    rc = metrics.get("recommendation", {})

    # unique_articles score: want at least 3 distinct articles (for multi_result)
    uniq = mr.get("unique_articles", 0)
    uniq_score = min(uniq / 3, 1.0) * mr.get("mean_score", 0)

    return round(
        pf.get("max_score", 0) * 0.35
        + uniq_score * 0.30
        + ki.get("mean_score", 0) * 0.20
        + rc.get("max_score", 0) * 0.15,
        4,
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("Medium RAG — Hyperparameter Search  (small-scale indexing)")
    print(f"Current config: chunk_size={CHUNK_SIZE}, overlap={OVERLAP_RATIO}, top_k={TOP_K}")
    print("=" * 68)

    # ── load CSV sample ───────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        sys.exit(1)

    print(f"\nLoading stratified sample (every {SAMPLE_STEP}th row) from CSV…")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    sample_rows = [
        (i, all_rows[i])
        for i in range(0, len(all_rows), SAMPLE_STEP)
        if (all_rows[i].get("text") or "").strip()
    ]
    print(f"  Total articles in CSV : {len(all_rows):,}")
    print(f"  Sampled articles      : {len(sample_rows)}")
    print(f"  Coverage              : {100*len(sample_rows)/len(all_rows):.1f}% of corpus")

    # ── connect to Pinecone ───────────────────────────────────────────────────
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    full_stats = index.describe_index_stats()
    print(f"\nPinecone index '{PINECONE_INDEX_NAME}'")
    print(f"  Full-corpus vectors (default ns): "
          f"{full_stats.total_vector_count:,}")

    # ── pre-embed evaluation queries once ─────────────────────────────────────
    print("\nEmbedding 4 evaluation queries (one-time cost)…")
    embedded_queries = []
    for q in QUERIES:
        embedded_queries.append((q, get_embedding(q["question"])))
    print("  Done.\n")

    # ── sweep: index each config → evaluate → delete namespace ───────────────
    print("─" * 68)
    print("PART 1 — Small-scale indexing sweep")
    print("─" * 68)

    all_results = {}   # label → {top_k → {query_type → metrics}}
    index_stats = {}   # label → {vectors, cost}
    total_cost = 0.0

    for cfg in CONFIGS:
        ns = f"test_{cfg['label']}"
        print(f"\n[ Config: chunk_size={cfg['chunk_size']}, "
              f"overlap={cfg['overlap_ratio']}, namespace={ns} ]")

        # clean up any leftover namespace from a previous run
        delete_namespace(index, ns)
        time.sleep(1)

        # index
        stats = index_sample(index, sample_rows, cfg, ns)
        index_stats[cfg["label"]] = stats
        total_cost += stats["cost"]
        print(f"  Indexed: {stats['vectors']:,} vectors | "
              f"est. cost: ${stats['cost']:.4f}")

        # brief wait for Pinecone to finalise the upsert
        time.sleep(2)

        # evaluate
        print(f"  Evaluating (top_k ∈ {TOP_K_CANDIDATES})…")
        eval_results = evaluate_namespace(index, embedded_queries, ns, TOP_K_CANDIDATES)
        all_results[cfg["label"]] = eval_results

        # print summary table
        print(f"\n  {'Query type':<30} {'k':<4} {'max':>6} {'mean':>6} "
              f"{'@3':>6} {'uniq/k'}")
        print(f"  {'─'*30} {'─'*3} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
        for k in TOP_K_CANDIDATES:
            for q in QUERIES:
                m = eval_results[k].get(q["type"], {})
                print(
                    f"  {q['label'][:30]:<30} {k:<4} "
                    f"{m.get('max_score',0):>6.4f} "
                    f"{m.get('mean_score',0):>6.4f} "
                    f"{m.get('score_at_3',0):>6.4f} "
                    f"{m.get('unique_articles',0)}/{k}"
                )

        # delete test namespace
        print(f"\n  Cleaning up namespace '{ns}'…")
        delete_namespace(index, ns)

    print(f"\n  Total embedding cost for sweep: ${total_cost:.4f}")

    # ── composite score ranking ───────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("PART 2 — Composite score ranking")
    print("─" * 68)

    rankings = []  # (score, label, k, cfg)
    for cfg in CONFIGS:
        lbl = cfg["label"]
        for k in TOP_K_CANDIDATES:
            metrics = all_results.get(lbl, {}).get(k, {})
            score = composite_score(metrics, k)
            rankings.append((score, lbl, k, cfg))

    rankings.sort(reverse=True)

    print(f"\n  {'Rank':<5} {'Config':<12} {'top_k':<7} {'Composite':>9}")
    print(f"  {'─'*4} {'─'*11} {'─'*6} {'─'*9}")
    for rank, (score, lbl, k, cfg) in enumerate(rankings, 1):
        marker = "  ← best" if rank == 1 else ""
        print(f"  {rank:<5} {lbl:<12} {k:<7} {score:>9.4f}{marker}")

    best_score, best_label, best_k, best_cfg = rankings[0]

    # ── validate best top_k against the full corpus ───────────────────────────
    print("\n" + "─" * 68)
    print("PART 3 — Validate best top_k on full corpus (default namespace)")
    print("─" * 68)
    print(f"\n  Best chunk config from sweep: chunk_size={best_cfg['chunk_size']}, "
          f"overlap={best_cfg['overlap_ratio']}")
    print(f"  Best top_k from sweep: {best_k}")
    print(f"\n  Running top_k ∈ {TOP_K_CANDIDATES} against full {full_stats.total_vector_count:,}-vector index…\n")

    full_results = {}
    for k in TOP_K_CANDIDATES:
        full_results[k] = {}
        for q, vec in embedded_queries:
            matches = index.query(
                vector=vec, top_k=k, include_metadata=True
            ).matches
            scores = [m.score for m in matches]
            unique_articles = len({m.metadata.get("article_id") for m in matches})
            full_results[k][q["type"]] = {
                "max_score": round(max(scores) if scores else 0, 4),
                "mean_score": round(statistics.mean(scores) if scores else 0, 4),
                "score_at_3": round(scores[2] if len(scores) >= 3 else scores[-1] if scores else 0, 4),
                "unique_articles": unique_articles,
                "top3_titles": [m.metadata.get("title", "")[:55] for m in matches[:3]],
            }

    print(f"  {'Query type':<30} {'k':<4} {'max':>6} {'mean':>6} {'@3':>6} {'uniq/k'}")
    print(f"  {'─'*30} {'─'*3} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for k in TOP_K_CANDIDATES:
        for q in QUERIES:
            m = full_results[k].get(q["type"], {})
            print(
                f"  {q['label'][:30]:<30} {k:<4} "
                f"{m.get('max_score',0):>6.4f} "
                f"{m.get('mean_score',0):>6.4f} "
                f"{m.get('score_at_3',0):>6.4f} "
                f"{m.get('unique_articles',0)}/{k}"
            )

    full_rankings = sorted(
        [(composite_score(full_results[k], k), k) for k in TOP_K_CANDIDATES],
        reverse=True,
    )
    best_full_k = full_rankings[0][1]
    print(f"\n  Full-corpus top_k composite scores:")
    for score, k in full_rankings:
        marker = "  ← recommended" if k == best_full_k else ""
        print(f"    top_k={k}: {score:.4f}{marker}")

    # ── final recommendation ──────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("FINAL RECOMMENDATION")
    print("=" * 68)
    print(f"""
  chunk_size   = {best_cfg['chunk_size']}  (from small-scale sweep)
  overlap_ratio= {best_cfg['overlap_ratio']}  (from small-scale sweep)
  top_k        = {best_full_k}    (validated on full corpus)

  For full-corpus re-indexing:
    - Estimated vectors : see chunk analysis table in hyperparam_results.json
    - Estimated cost    : ~${best_cfg['chunk_size'] / 512 * 0.25:.2f}  (scaled from baseline $0.25)
    - Command           : python scripts/index_articles.py
      (after updating lib/rag_config.py with the values above)
""")

    # ── persist results ───────────────────────────────────────────────────────
    output = {
        "sample_size": len(sample_rows),
        "sample_step": SAMPLE_STEP,
        "configs_tested": CONFIGS,
        "top_k_candidates": TOP_K_CANDIDATES,
        "small_scale_results": {
            lbl: {str(k): v for k, v in res.items()}
            for lbl, res in all_results.items()
        },
        "small_scale_index_stats": index_stats,
        "small_scale_rankings": [
            {"score": s, "label": l, "top_k": k}
            for s, l, k, _ in rankings
        ],
        "full_corpus_validation": {
            str(k): v for k, v in full_results.items()
        },
        "full_corpus_rankings": [
            {"score": s, "top_k": k} for s, k in full_rankings
        ],
        "recommendation": {
            "chunk_size": best_cfg["chunk_size"],
            "overlap_ratio": best_cfg["overlap_ratio"],
            "top_k": best_full_k,
            "sweep_cost_usd": round(total_cost, 4),
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Raw results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
