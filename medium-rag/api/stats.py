import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler
import json

from lib.rag_config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        result = {
            "chunk_size": CHUNK_SIZE,
            "overlap_ratio": OVERLAP_RATIO,
            "top_k": TOP_K,
        }
        body = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass
