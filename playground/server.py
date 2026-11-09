import argparse
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from playground.tokenizers_engine import AVAILABLE_MODELS, tokenize_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# In-memory LRU cache: (model_id, text) -> result dict
_CACHE = {}
_CACHE_MAX = 512


def cached_tokenize(model_id: str, text: str) -> dict:
    key = (model_id, text)
    if key in _CACHE:
        return _CACHE[key]
    res = tokenize_text(model_id, text)
    if len(_CACHE) >= _CACHE_MAX:
        # Evict oldest entry
        first_key = next(iter(_CACHE))
        _CACHE.pop(first_key, None)
    _CACHE[key] = res
    return res


class PlaygroundHandler(SimpleHTTPRequestHandler):
    """HTTP Request handler for tokenizer playground."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        # Concise server logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def end_headers(self):
        # Enable CORS for local dev
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/models":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            payload = json.dumps({"models": AVAILABLE_MODELS}, ensure_ascii=False)
            self.wfile.write(payload.encode("utf-8"))
            return

        # Fallback to static file serving
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/tokenize":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            try:
                data = json.loads(body_bytes.decode("utf-8"))
            except Exception as e:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON payload: {e}")
                return

            text = data.get("text", "")
            model_id = data.get("model_id")
            model_ids = data.get("model_ids")

            if not model_id and not model_ids:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing model_id or model_ids")
                return

            targets = model_ids if model_ids else [model_id]

            results = {}
            for mid in targets:
                try:
                    results[mid] = cached_tokenize(mid, text)
                except Exception as ex:
                    results[mid] = {
                        "error": str(ex),
                        "model_id": mid
                    }

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            response_payload = json.dumps({
                "text": text,
                "results": results
            }, ensure_ascii=False)
            self.wfile.write(response_payload.encode("utf-8"))
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Path not found")


def run(host: str = "127.0.0.1", port: int = 7860):
    candidate_ports = [port, 7860, 8765, 5173, 8000, 3000, 8088]
    httpd = None
    bound_port = None

    for p in candidate_ports:
        try:
            server_address = (host, p)
            httpd = ThreadingHTTPServer(server_address, PlaygroundHandler)
            bound_port = p
            break
        except (PermissionError, OSError) as e:
            continue

    if httpd is None:
        sys.exit("Failed to bind server to any candidate port.")

    print(f"Tamil Tokenizer Playground running at http://{host}:{bound_port}/")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tamil Tokenizer Playground Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (default: 7860)")
    args = parser.parse_args()
    run(host=args.host, port=args.port)
