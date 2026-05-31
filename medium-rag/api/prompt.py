import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from http.server import BaseHTTPRequestHandler
import json

try:
    from lib.rag_config import TOP_K, LLM_MODEL, OPENAI_BASE_URL, EMBEDDING_MODEL
    from lib.embedder import get_embedding
    from lib.pinecone_client import query_index
    from lib.prompt_builder import (
        build_system_prompt,
        build_user_prompt,
        build_context_str,
        call_llm,
    )
    IMPORT_ERROR = None
except Exception as e:
    IMPORT_ERROR = traceback.format_exc()

# ── startup diagnostic (visible in Vercel function logs) ─────────────────────
if IMPORT_ERROR:
    print(f"[prompt] STARTUP: import FAILED:\n{IMPORT_ERROR}", flush=True)
else:
    _key_hint = OPENAI_BASE_URL[:40] if OPENAI_BASE_URL else "(not set)"
    print(
        f"[prompt] STARTUP OK | "
        f"LLM_MODEL={LLM_MODEL!r} | "
        f"EMBEDDING_MODEL={EMBEDDING_MODEL!r} | "
        f"TOP_K={TOP_K} | "
        f"base_url={_key_hint!r}",
        flush=True,
    )

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_POST(self):
        # Outer catch-all: prevents Vercel from receiving an unhandled exception
        # (which would produce its own opaque 500 instead of our diagnostic 5xx).
        try:
            self._handle_post()
        except BaseException as exc:
            print(
                f"[prompt] FATAL unhandled {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            # Attempt to tell the caller something; if this fails we at least
            # logged above so the Vercel function log has the full traceback.
            try:
                self._json(500, {"error": f"Fatal: {type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def _handle_post(self):
        if IMPORT_ERROR:
            print(f"[prompt] import error path hit", flush=True)
            self._json(500, {"error": "Import failed", "detail": IMPORT_ERROR})
            return

        # ── parse request ─────────────────────────────────────────────────────
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "Invalid JSON body"})
            return

        question = body.get("question", "").strip()
        if not question:
            self._json(400, {"error": '"question" field is required and must not be empty'})
            return

        print(f"[prompt] question={question[:80]!r}", flush=True)

        # ── Step 1: embed ─────────────────────────────────────────────────────
        try:
            print("[prompt] step=1 embedding...", flush=True)
            t0 = time.time()
            query_vector = get_embedding(question)
            ms = (time.time() - t0) * 1000
            print(f"[prompt] step=1 OK embed_ms={ms:.0f}", flush=True)
        except Exception as e:
            print(f"[prompt] step=1 FAIL embed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            self._json(503, {"error": f"Embedding failed: {e}"})
            return

        # ── Step 2: retrieve ──────────────────────────────────────────────────
        try:
            print("[prompt] step=2 pinecone query...", flush=True)
            t0 = time.time()
            matches = query_index(query_vector, top_k=TOP_K)
            ms = (time.time() - t0) * 1000
            top_score = matches[0].score if matches else float("nan")
            print(
                f"[prompt] step=2 OK chunks={len(matches)} ms={ms:.0f} top_score={top_score:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[prompt] step=2 FAIL pinecone: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            self._json(503, {"error": f"Retrieval failed: {e}"})
            return

        # ── Build context items for the response ──────────────────────────────
        context_items = []
        for match in matches:
            meta = match.metadata
            context_items.append({
                "article_id": meta.get("article_id", match.id),
                "title": meta.get("title", ""),
                "chunk": meta.get("chunk_text", ""),
                "score": round(float(match.score), 4),
            })

        context_str = build_context_str(matches)
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(question, context_str)

        # ── Step 3: LLM call ──────────────────────────────────────────────────
        try:
            print(
                f"[prompt] step=3 llm call model={LLM_MODEL!r} ...",
                flush=True,
            )
            t0 = time.time()
            response_text = call_llm(system_prompt, user_prompt)
            ms = (time.time() - t0) * 1000
            print(
                f"[prompt] step=3 OK llm_ms={ms:.0f} response_len={len(response_text)}",
                flush=True,
            )
        except Exception as e:
            print(f"[prompt] step=3 FAIL llm: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            self._json(503, {"error": f"LLM call failed: {e}"})
            return

        # ── Step 4: send response ─────────────────────────────────────────────
        print("[prompt] step=4 sending 200 response...", flush=True)
        result = {
            "response": response_text,
            "context": context_items,
            "Augmented_prompt": {
                "System": system_prompt,
                "User": user_prompt,
            },
        }
        self._json(200, result)
        print("[prompt] step=4 OK response sent", flush=True)

    def log_message(self, format, *args):
        pass  # suppress default access log noise

    def _json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
