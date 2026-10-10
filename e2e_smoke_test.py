"""Offline end-to-end test for the complete pipeline using local Ollama mock."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from ollama_client import OllamaEngine
from orchestrator import RUNS_DIR_NAME, VibeOrchestrator


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
        lower = prompt.lower()
        if "auditor" in lower:
            content = "<think>checking spec compliance</think>PASS\nGenerated files satisfy the specification."
        elif "pure code generator" in lower or "precise code editor" in lower:
            content = "```FILE: generated.py\n\ndef greet(name: str) -> str:\n    return f'Hello, {name}!'\n```"
        else:
            content = "<think>planning</think>## TASK 1: create greeting\n- files: generated.py\n- logic: def greet(name: str) -> str"
        response = json.dumps(
            {
                "response": content,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 128,
                "eval_count": 64,
                "total_duration": 1_000_000_000,
            }
        ).encode()
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
            workspace_root = Path(workspace)
            target = workspace_root / "generated.py"

            assert result.status == "completed", result
            assert result.run_id, "run must carry a run_id"
            assert target.exists()

            # Junk-file regression: no fake-role artifacts in the workspace root
            assert not (workspace_root / "summary.txt").exists()
            assert not (workspace_root / "tasks.md").exists()
            assert not list(workspace_root.glob("task*_code.txt"))

            # Run metadata lives under .agent_runs/<run_id>/
            run_dir = workspace_root / RUNS_DIR_NAME / result.run_id
            assert (run_dir / "state.json").exists()
            assert (run_dir / "events.jsonl").exists()
            assert (run_dir / "tasks.md").exists()
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            assert state["status"] == "completed", state
            assert state["llm_calls"] > 0

            # Only the requested target plus benign git-ignored infrastructure:
            # .agent (config), .agent_runs/ (run metadata), .agent_staging/ and
            # .agent_logs/ (emptied by deploy), __pycache__/ (py_compile).
            # NO role junk files anywhere in the root.
            root_entries = {p.name for p in workspace_root.iterdir()}
            assert root_entries <= {
                "generated.py", ".agent", RUNS_DIR_NAME,
                ".agent_staging", ".agent_logs", "__pycache__",
            }, root_entries
            # Deploy flushed the staging cache: no generated files left behind.
            staging_leftovers = [
                p for p in (workspace_root / ".agent_staging").rglob("*") if p.is_file()
            ]
            assert not staging_leftovers, staging_leftovers

            # Think blocks were stripped before validation/parsing
            assert "think" not in result.llm_validation.lower()[:200] or result.llm_validation.startswith("PASS")
        print("E2E smoke test passed: Ollama thinker -> coder -> write -> compile")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
