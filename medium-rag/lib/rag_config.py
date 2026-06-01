import os

# --- RAG Hyperparameters ---
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.25
OVERLAP_TOKENS = int(CHUNK_SIZE * OVERLAP_RATIO)  # 128
STRIDE = CHUNK_SIZE - OVERLAP_TOKENS              # 384
TOP_K = 5

# --- Model Names ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small")
LLM_MODEL = os.environ.get("LLM_MODEL", "4UHRUIN-gpt-5-mini")
EMBEDDING_DIMS = 1536

# --- API Credentials ---
# When using the course proxy models (4UHRUIN-*), UHRUIN_API_KEY /
# UHRUIN_BASE_URL take precedence so the standard OPENAI_* vars can keep
# pointing at api.openai.com without breaking anything.
_is_uhruin = "4UHRUIN" in EMBEDDING_MODEL or "4UHRUIN" in LLM_MODEL

OPENAI_API_KEY = (
    os.environ.get("UHRUIN_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if _is_uhruin
    else os.environ.get("OPENAI_API_KEY", "")
)
OPENAI_BASE_URL = (
    os.environ.get("UHRUIN_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if _is_uhruin
    else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-articles")

# --- Indexing Batch Sizes ---
EMBED_BATCH_SIZE = 100
PINECONE_BATCH_SIZE = 100

# text-embedding-3-small: $0.02 / 1M tokens
COST_PER_TOKEN = 0.02 / 1_000_000
