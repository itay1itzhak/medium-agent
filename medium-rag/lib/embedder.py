"""OpenAI embedding wrapper with module-level client caching."""

from openai import OpenAI
from lib.rag_config import OPENAI_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def get_embedding(text: str) -> list[float]:
    """Embed a single string. Returns a 1536-dim vector."""
    resp = get_client().embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings (max 100). Returns list of 1536-dim vectors."""
    resp = get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
