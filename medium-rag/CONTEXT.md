# Medium Article RAG Assistant

## Overview

A Retrieval-Augmented Generation (RAG) API that answers questions about Medium articles, deployed as Python serverless functions on Vercel. The system indexes ~7,600 English Medium articles into a Pinecone vector database and answers queries strictly from the retrieved content.

## Assignment Objectives

- Build a RAG pipeline over `medium-english-50mb.csv` (~7,600 articles)
- Expose two API endpoints: `POST /api/prompt` and `GET /api/stats`
- Answer four query categories: precise fact retrieval, topic listing, key idea summarization, and recommendation with evidence
- Stay within a $5 API budget across development and testing
- Deploy publicly on Vercel; keep Pinecone index active until grading

## Architecture

```
Query
  │
  ▼
get_embedding(question)           ← OpenAI-compatible API
  │  1536-dim vector
  ▼
query_index(vector, top_k=7)      ← Pinecone cosine search
  │  top 7 scored chunks + metadata
  ▼
build_context_str(matches)
build_user_prompt(question, ctx)
call_llm(system_prompt, user_prompt)  ← OpenAI-compatible chat API
  │
  ▼
JSON response
```

The offline indexing pipeline (run once locally):

```
CSV row → chunk_text() → get_embeddings_batch() → pinecone.upsert()
```

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 512 tokens | Covers a full paragraph; well under the 8,191-token embedding limit |
| `overlap_ratio` | 0.2 (20%) | 102-token overlap preserves context across chunk boundaries |
| `top_k` | 7 | Enough diversity (~3,500 retrieved tokens); keeps LLM input cost low |

These are reported by `GET /api/stats` and defined in `lib/rag_config.py`.

## Component Descriptions

| File | Purpose |
|------|---------|
| `lib/rag_config.py` | Single source of truth: all hyperparameters + env var reads |
| `lib/chunker.py` | Token-based sliding-window chunker using `tiktoken`; local only |
| `lib/embedder.py` | OpenAI embedding wrapper with module-level client caching |
| `lib/pinecone_client.py` | Pinecone query wrapper with module-level index caching |
| `lib/prompt_builder.py` | Required system prompt, user prompt builder, LLM call |
| `api/prompt.py` | POST /api/prompt — full RAG pipeline handler (Vercel function) |
| `api/stats.py` | GET /api/stats — returns current hyperparameters (Vercel function) |
| `scripts/index_articles.py` | One-time offline indexing: reads CSV → chunks → embeds → upserts |
| `scripts/verify_index.py` | Post-indexing sanity check: counts vectors + runs 4 sample queries |

## API Reference

### POST /api/prompt

**Request:**
```json
{ "question": "Which article explains the attention mechanism in transformers?" }
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

### GET /api/stats

**Response:**
```json
{ "chunk_size": 512, "overlap_ratio": 0.2, "top_k": 7 }
```

## Dependencies

**Runtime (Vercel — `requirements.txt`):**
- `openai>=1.0.0` — embedding + chat completion calls
- `pinecone-client>=3.0.0` — vector store queries

**Local/indexing only (`requirements-dev.txt`):**
- `tiktoken>=0.5.0` — token counting for chunker
- `pandas>=2.0.0` — optional CSV inspection
- `tqdm>=4.65.0` — progress bar during indexing
- `python-dotenv>=1.0.0` — loads `.env` for local scripts

## Environment Variables

Copy `.env.example` to `.env` and fill in your credentials.

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for the course-provided OpenAI-compatible endpoint |
| `OPENAI_BASE_URL` | Base URL of the custom API endpoint (e.g., `https://...`) |
| `EMBEDDING_MODEL` | `4UHRUIN-text-embedding-3-small` |
| `LLM_MODEL` | `4UHRUIN-gpt-5-mini` |
| `PINECONE_API_KEY` | Pinecone API key (from pinecone.io console) |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `medium-articles`) |

Set the same variables in the Vercel dashboard (Settings → Environment Variables) before deploying.

## How to Run

### Step 1: Install local dependencies

```bash
cd medium-rag
pip install -r requirements-dev.txt
```

### Step 2: Configure credentials

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### Step 3: Index the articles (one-time, ~15 minutes)

```bash
python scripts/index_articles.py
```

This reads `../medium-english-50mb.csv`, chunks all articles, embeds them in batches of 100, and upserts into Pinecone. Upserts are idempotent — safe to re-run if interrupted.

### Step 4: Verify the index

```bash
python scripts/verify_index.py
```

Prints total vector count (~30,000 expected) and runs 4 sample queries.

### Step 5: Deploy to Vercel

```bash
# Push to GitHub first (the .gitignore excludes .env and *.csv)
git init && git add . && git commit -m "initial commit"
# Then import the repo at vercel.com/new
# Set all env vars in the Vercel dashboard
vercel --prod
```

### Step 6: Test deployed endpoints

```bash
# Test POST /api/prompt
curl -X POST https://your-app.vercel.app/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "What is transformer architecture?"}'

# Test GET /api/stats
curl https://your-app.vercel.app/api/stats
```

## Cost Estimate

| Phase | Estimated Cost |
|-------|---------------|
| Indexing (~12.5M tokens @ $0.02/1M) | ~$0.25 |
| Pinecone serverless free tier | $0.00 |
| Per query (~4,000 input + 400 output tokens) | ~$0.0009 |
| Budget remaining for queries | ~$4.75 (~5,000+ queries) |

## Assumptions

1. The `4UHRUIN-*` models are served via an OpenAI-compatible API (same SDK interface, different `base_url`).
2. The Pinecone free tier (serverless, 100K vectors, AWS us-east-1) is sufficient for ~30,000 chunks from the dataset.
3. Article text fields are in English and well-formed; minimal preprocessing is needed.
4. The `cl100k_base` tiktoken encoding matches the tokenizer used by `4UHRUIN-text-embedding-3-small` (standard for text-embedding-3-small).
5. Articles with empty or missing `text` fields are skipped during indexing.

## Supported Query Types

The same pipeline handles all four required query categories:

| Type | Example | How it works |
|------|---------|-------------|
| Precise fact retrieval | "Which article discusses the impact of isolation on the brain?" | Top-scored chunk is from the target article; LLM cites it specifically |
| Multi-result topic listing | "List articles about meditation" | 7 retrieved chunks span multiple articles; LLM enumerates titles from context |
| Key idea summary | "What are the main ideas about remote work?" | Multiple thematically related chunks retrieved; LLM synthesizes across them |
| Recommendation with evidence | "Recommend an article on startup fundraising" | Top chunk becomes the recommendation; LLM uses chunk text as justification |
