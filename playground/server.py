import argparse
import json
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

STATIC_DIR = os.path.join(BASE_DIR, "static")


class PlaygroundHandler(SimpleHTTPRequestHandler):
    """HTTP Request handler for tokenizer playground."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def end_headers(self):
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
                    results[mid] = tokenize_text(mid, text)
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
    try:
        server_address = (host, port)
        httpd = ThreadingHTTPServer(server_address, PlaygroundHandler)
        bound_port = port
    except (PermissionError, OSError) as e:
        candidate_ports = [7860, 8765, 5173, 8000, 3000, 8088]
        httpd = None
        bound_port = None
        for p in candidate_ports:
            try:
                server_address = (host, p)
                httpd = ThreadingHTTPServer(server_address, PlaygroundHandler)
                bound_port = p
                break
            except (PermissionError, OSError):
                continue
        if httpd is None:
            sys.exit(f"Failed to bind server to {host}:{port}: {e}")

    print(f"Tamil Tokenizer Playground running at http://{host}:{bound_port}/")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    env_port = int(os.environ.get("PORT", 7860))
    env_host = os.environ.get("HOST", "0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
    parser = argparse.ArgumentParser(description="Tamil Tokenizer Playground Web Server")
    parser.add_argument("--host", default=env_host, help="Host address to bind")
    parser.add_argument("--port", type=int, default=env_port, help="Port to bind")
    args = parser.parse_args()
    run(host=args.host, port=args.port)
