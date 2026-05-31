import os

# --- RAG Hyperparameters ---
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
OVERLAP_TOKENS = int(CHUNK_SIZE * OVERLAP_RATIO)  # 102
STRIDE = CHUNK_SIZE - OVERLAP_TOKENS              # 410
TOP_K = 7

# --- Model Names ---
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small")
LLM_MODEL = os.environ.get("LLM_MODEL", "4UHRUIN-gpt-5-mini")
EMBEDDING_DIMS = 1536

# --- API Credentials ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-articles")

# --- Indexing Batch Sizes ---
EMBED_BATCH_SIZE = 100
PINECONE_BATCH_SIZE = 100

# text-embedding-3-small: $0.02 / 1M tokens
COST_PER_TOKEN = 0.02 / 1_000_000
