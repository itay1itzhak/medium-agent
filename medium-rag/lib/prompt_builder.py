"""System/user prompt construction and LLM call."""

from lib.embedder import get_client
from lib.rag_config import LLM_MODEL

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    "'I don't know based on the provided Medium articles data.' "
    "Always explain your answer using the given context, quoting or paraphrasing "
    "the relevant article passage or metadata when helpful."
)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_prompt(question: str, context_str: str) -> str:
    return (
        f"Context from Medium articles:\n\n{context_str}\n\n"
        f"Question: {question}"
    )


def build_context_str(matches: list) -> str:
    parts = []
    for match in matches:
        meta = match.metadata
        title = meta.get("title", "Unknown")
        chunk_text = meta.get("chunk_text", "")
        parts.append(f"[Article: {title}]\n{chunk_text}")
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    resp = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=600,
    )
    return resp.choices[0].message.content
