import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from http.server import BaseHTTPRequestHandler
import json

try:
    from lib.rag_config import TOP_K
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
        if IMPORT_ERROR:
            self._json(500, {"error": "Import failed", "detail": IMPORT_ERROR})
            return

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

        print(f"[prompt] question={question[:80]!r}")

        try:
            t0 = time.time()
            query_vector = get_embedding(question)
            ms = (time.time() - t0) * 1000
            print(f"[prompt] embedded in {ms:.0f}ms")

            t0 = time.time()
            matches = query_index(query_vector, top_k=TOP_K)
            ms = (time.time() - t0) * 1000
            top_score = matches[0].score if matches else float("nan")
            print(f"[prompt] retrieved {len(matches)} chunks in {ms:.0f}ms, top score={top_score:.4f}")

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

            t0 = time.time()
            response_text = call_llm(system_prompt, user_prompt)
            ms = (time.time() - t0) * 1000
            print(f"[prompt] llm responded in {ms:.0f}ms, len={len(response_text)}")

            result = {
                "response": response_text,
                "context": context_items,
                "Augmented_prompt": {
                    "System": system_prompt,
                    "User": user_prompt,
                },
            }
            self._json(200, result)

        except Exception as e:
            print(f"[prompt] ERROR: {e}", flush=True)
            traceback.print_exc()
            self._json(503, {"error": f"Service error: {str(e)}"})

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
