import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler
import json

from lib.rag_config import TOP_K
from lib.embedder import get_embedding
from lib.pinecone_client import query_index
from lib.prompt_builder import (
    build_system_prompt,
    build_user_prompt,
    build_context_str,
    call_llm,
)


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
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

        try:
            query_vector = get_embedding(question)
            matches = query_index(query_vector, top_k=TOP_K)

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
            response_text = call_llm(system_prompt, user_prompt)

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
            self._json(503, {"error": f"Service error: {str(e)}"})

    def log_message(self, format, *args):
        pass  # suppress default access log noise

    def _json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
