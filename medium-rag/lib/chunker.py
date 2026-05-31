"""Token-based sliding-window chunker. Used only by scripts/, never deployed."""

import tiktoken
from lib.rag_config import CHUNK_SIZE, STRIDE

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def chunk_text(text: str) -> list[dict]:
    """
    Split text into overlapping token windows.

    Returns a list of dicts:
        {"text": str, "chunk_index": int, "token_start": int}
    """
    enc = get_tokenizer()
    tokens = enc.encode(text)

    if not tokens:
        return []

    chunks = []
    chunk_index = 0
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append({
            "text": enc.decode(chunk_tokens),
            "chunk_index": chunk_index,
            "token_start": start,
        })
        if end == len(tokens):
            break
        start += STRIDE
        chunk_index += 1

    return chunks
