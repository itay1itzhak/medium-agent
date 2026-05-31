"""Pinecone query wrapper with module-level index caching."""

from pinecone import Pinecone
from lib.rag_config import PINECONE_API_KEY, PINECONE_INDEX_NAME, TOP_K

_index = None


def get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


def query_index(vector: list[float], top_k: int = TOP_K) -> list:
    """
    Query Pinecone for nearest neighbors.

    Returns a list of ScoredVector objects, each with:
        .id, .score, .metadata (dict with title, chunk_text, article_id, etc.)
    """
    result = get_index().query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
    )
    return result.matches
