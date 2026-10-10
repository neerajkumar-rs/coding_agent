"""Offline end-to-end test for the complete pipeline using local Ollama mock."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from ollama_client import OllamaEngine
from orchestrator import VibeOrchestrator


class FakeOllama(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = {"models": [{"name": "deepseek-r1:1.5b"}, {"name": "qwen2.5-coder:3b"}]}
        else:
            body = {"status": "ok"}
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        prompt = payload.get("prompt", "")
        if "validation" in prompt.lower():
            content = "PASS\nGenerated files satisfy the specification."
        elif "pure code generator" in prompt.lower():
            content = "```FILE: generated.py\n\ndef greet(name: str) -> str:\n    return f'Hello, {name}!'\n```"
        else:
            content = "## TASK 1: create greeting\n- files: generated.py\n- logic: def greet(name: str) -> str"
        response = json.dumps({"response": content}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with TemporaryDirectory() as workspace:
            engine = OllamaEngine(base_url=base_url, workspace_root=workspace)
            result = VibeOrchestrator(engine=engine, workspace=workspace).run(
                "Create a greeting function", "generated.py"
            )
            target = Path(workspace) / "generated.py"
            summary = Path(workspace) / "summary.txt"
            assert result.status == "completed", result
            assert target.exists()
            assert summary.exists()
            assert "generated.py" in summary.read_text(encoding="utf-8")
        print("E2E smoke test passed: Ollama thinker -> coder -> write -> compile")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
