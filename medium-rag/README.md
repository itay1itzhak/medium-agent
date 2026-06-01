# Medium Article RAG System

Retrieval-Augmented Generation (RAG) API that answers natural-language questions about Medium articles. Indexes ~7,600 English articles into a Pinecone vector database and serves answers via two HTTP endpoints deployed on Vercel.

**Course assignment — Individual RAG Assignment**

---

## Live Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/prompt` | Answer a question using RAG |
| `GET` | `/api/stats` | Return the current RAG configuration |

---

## Architecture

```
User question
      │
      ▼
get_embedding(question)          ← OpenAI-compatible embedding API
      │  1536-dim vector
      ▼
query_index(vector, top_k=5)     ← Pinecone cosine similarity search
      │  top-5 scored chunks + metadata
      ▼
build_user_prompt(question, ctx)
call_llm(system_prompt, user_prompt)  ← OpenAI-compatible chat API
      │
      ▼
JSON response  { response, context, Augmented_prompt }
```

**Offline indexing (run once):**
```
CSV row → chunk_text() → get_embeddings_batch() → pinecone.upsert()
```

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | **512** tokens | Paragraph granularity; best balance across all 4 query types |
| `overlap_ratio` | **0.2** (20 %) | 102-token overlap preserves context at chunk boundaries |
| `top_k` | **5** | Best composite score on the full corpus; 4+ distinct articles for multi-result queries |

These values were selected through a two-stage hyperparameter search (see [HYPERPARAMETER_REPORT.md](HYPERPARAMETER_REPORT.md)):

1. **Small-scale sweep** — 4 chunk configurations × 3 top_k values on a 308-article stratified sample (~$0.05).
2. **Full-corpus validation** — top_k re-evaluated on the live 27,426-vector index at no additional embedding cost.

`chunk_size=512` outperforms 256 (fragmented concepts) and 768 (diluted mean score). `top_k=5` beats 7 and 10 because extra lower-quality chunks dilute the LLM context without improving answers.

---

## API Reference

### `POST /api/prompt`

**Request:**
```json
{ "question": "Which article explains how the attention mechanism works in transformers?" }
```

**Response:**
```json
{
  "response": "Based on the retrieved articles...",
  "context": [
    {
      "article_id": "42",
      "title": "The Illustrated Transformer",
      "chunk": "The attention mechanism allows...",
      "score": 0.8912
    }
  ],
  "Augmented_prompt": {
    "System": "You are a Medium-article assistant...",
    "User": "Context from Medium articles:\n\n[Article: ...]...\n\nQuestion: ..."
  }
}
```

### `GET /api/stats`

**Response:**
```json
{ "chunk_size": 512, "overlap_ratio": 0.2, "top_k": 5 }
```

---

## Supported Query Types

| Type | Example |
|------|---------|
| Precise fact retrieval | "Which article explains how the attention mechanism works in transformers?" |
| Multi-result topic listing | "List up to 3 articles about productivity and time management" |
| Key idea summary | "Summarize the key ideas about building ML models in production" |
| Recommendation with evidence | "Recommend an article about startup fundraising and explain why" |

---

## Project Structure

```
medium-rag/
├── api/
│   ├── prompt.py          # POST /api/prompt — full RAG pipeline (Vercel function)
│   └── stats.py           # GET /api/stats — returns hyperparameters (Vercel function)
├── lib/
│   ├── rag_config.py      # Single source of truth: hyperparameters + env var reads
│   ├── chunker.py         # Token-based sliding-window chunker (tiktoken)
│   ├── embedder.py        # OpenAI embedding wrapper with module-level client caching
│   ├── pinecone_client.py # Pinecone query wrapper with module-level index caching
│   └── prompt_builder.py  # System prompt, user prompt builder, LLM call
├── scripts/
│   ├── index_articles.py  # One-time offline indexing: CSV → chunks → embed → upsert
│   ├── verify_index.py    # Post-index sanity check: vector count + 4 sample queries
│   ├── hyperparam_search.py  # Automated hyperparameter sweep (see HYPERPARAMETER_REPORT.md)
│   ├── test_query.py      # Test the deployed app with a single user-provided question
│   ├── test_vercel.py     # Full endpoint test suite: stats + 4 canonical questions
│   ├── debug_500.py       # Diagnose Vercel 500 errors with detailed request/response dump
│   ├── test_pipeline.py   # Local end-to-end pipeline test (no Vercel)
│   ├── test_retrieval.py  # Test Pinecone retrieval in isolation
│   ├── test_chunker.py    # Unit tests for the chunker
│   └── test_embedder.py   # Unit tests for the embedder
├── vercel.json            # Vercel build + routing config
├── requirements.txt       # Runtime dependencies (Vercel)
├── requirements-dev.txt   # Local/indexing dependencies
├── .env.example           # Template for credentials
├── CONTEXT.md             # Architecture reference
└── HYPERPARAMETER_REPORT.md  # Full hyperparameter search results and analysis
```

---

## Setup

### 1. Install dependencies

```bash
cd medium-rag
pip install -r requirements-dev.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://your-custom-endpoint/v1

EMBEDDING_MODEL=4UHRUIN-text-embedding-3-small
LLM_MODEL=4UHRUIN-gpt-5-mini

PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=medium-articles
```

> The course proxy models (`4UHRUIN-*`) are OpenAI-compatible. Set `OPENAI_BASE_URL` to the course-provided endpoint.

### 3. Index the articles (one-time, ~15 minutes)

Place `medium-english-50mb.csv` in the parent directory (`../medium-english-50mb.csv`), then run:

```bash
python scripts/index_articles.py
```

This reads every article, chunks with a 512-token sliding window (stride 410), embeds in batches of 100, and upserts into Pinecone. Upserts are idempotent — safe to re-run if interrupted. Expected output: ~27,000 vectors.

### 4. Verify the index

```bash
python scripts/verify_index.py
```

Prints total vector count and runs 4 sample queries to confirm retrieval is working.

---

## Deploy to Vercel

```bash
# Commit the code (CSV and .env are gitignored)
git add . && git commit -m "initial commit"

# Deploy (or import the repo at vercel.com/new)
vercel --prod
```

Set all six environment variables in the Vercel dashboard under **Settings → Environment Variables** before deploying. The `vercel.json` configures 60-second function timeout for `/api/prompt` (embedding + retrieval + LLM can take 5–20 s).

---

## Testing

### Test a single query against the deployed app

```bash
# Basic usage
python scripts/test_query.py https://your-app.vercel.app "Your question here"

# Also print the full raw JSON response
python scripts/test_query.py https://your-app.vercel.app "Your question here" --json

# Use VERCEL_URL env var instead of a positional argument
VERCEL_URL=https://your-app.vercel.app python scripts/test_query.py "Your question"

# Interactive mode (prompts for URL and question)
python scripts/test_query.py
```

Output is divided into five stages:

| Stage | What it shows |
|-------|--------------|
| 1 · App Status | `GET /api/stats` result; validates `chunk_size`, `overlap_ratio`, `top_k` |
| 2 · Sending Request | HTTP status and latency for `POST /api/prompt` |
| 3 · Retrieved Context | All chunks with score, title, and text preview |
| 4 · Augmented Prompt | System + user prompt word counts and first 300 characters |
| 5 · Final Answer | Full LLM response |

With `--json`, the complete raw API response is printed at the end for inspection.

### Run the full test suite (4 canonical questions)

```bash
python scripts/test_vercel.py https://your-app.vercel.app
# or
VERCEL_URL=https://your-app.vercel.app python scripts/test_vercel.py
```

Tests `GET /api/stats` and `POST /api/prompt` with all four required query types, then prints a summary table.

### Diagnose a 500 error

```bash
python scripts/debug_500.py https://your-app.vercel.app
```

Dumps the full response headers and body, classifies whether the error is Vercel-internal (with a request ID to look up in function logs) or application-level, and pinpoints the failing step.

### Local pipeline tests (no Vercel)

```bash
# Full RAG pipeline locally (requires .env)
python scripts/test_pipeline.py

# Pinecone retrieval only
python scripts/test_retrieval.py

# Chunker unit tests
python scripts/test_chunker.py

# Embedder unit tests
python scripts/test_embedder.py
```

### Quick curl tests

```bash
# POST /api/prompt
curl -s -X POST https://your-app.vercel.app/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the attention mechanism?"}' | python -m json.tool

# GET /api/stats
curl -s https://your-app.vercel.app/api/stats | python -m json.tool
```

---

## Cost Estimate

| Phase | Cost |
|-------|------|
| Full-corpus indexing (~12.5M tokens, one-time) | ~$0.25 |
| Hyperparameter sweep (308-article sample) | ~$0.05 |
| Full-corpus top_k validation (no new embeddings) | ~$0.00002 |
| **Total development cost** | **~$0.30** |
| Per query (~4,000 input + 400 output tokens) | ~$0.0009 |
| Remaining budget for queries | **~$4.70 (~5,000+ queries)** |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for the OpenAI-compatible endpoint |
| `OPENAI_BASE_URL` | Base URL of the API (e.g., `https://your-endpoint/v1`) |
| `EMBEDDING_MODEL` | Embedding model name (e.g., `4UHRUIN-text-embedding-3-small`) |
| `LLM_MODEL` | Chat model name (e.g., `4UHRUIN-gpt-5-mini`) |
| `PINECONE_API_KEY` | Pinecone API key from the Pinecone console |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `medium-articles`) |

All variables must be set both in `.env` (for local scripts) and in the Vercel dashboard (for deployed functions).
