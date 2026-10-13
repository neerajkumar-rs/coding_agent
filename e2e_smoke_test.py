"""Offline end-to-end test for the complete run_pipeline using a local Ollama mock."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_core.run_state import RUNS_DIR_NAME
from orchestrator import run_pipeline


class FakeOllama(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = {"models": [{"name": "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"}, {"name": "qwen2.5-coder:3b-instruct"}]}
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
        if "auditor" in lower or "satisfies_request" in lower:
            content = "<think>checking spec compliance</think>PASS\nGenerated files satisfy the specification."
        elif "pure code generator" in lower or "precise code editor" in lower or "target file:" in lower:
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
        with TemporaryDirectory() as workspace:
            result = asyncio.run(
                run_pipeline(
                    user_request="Create a greeting function in generated.py",
                    workspace=workspace,
                    deploy=True,
                )
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
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            assert state["status"] == "completed", state

            # Only the requested target plus benign git-ignored infrastructure:
            # .agent (config), .agent_runs/ (run metadata), .agent_staging/
            # (emptied by deploy), __pycache__/ (py_compile).
            # NO role junk files anywhere in the root.
            root_entries = {p.name for p in workspace_root.iterdir()}
            assert root_entries <= {
                "generated.py", ".agent", RUNS_DIR_NAME,
                ".agent_staging", "__pycache__",
            }, root_entries
            # Deploy flushed the staging cache: no generated files left behind.
            staging_dir = workspace_root / ".agent_staging"
            if staging_dir.exists():
                staging_leftovers = [
                    p for p in staging_dir.rglob("*") if p.is_file()
                ]
                assert not staging_leftovers, staging_leftovers
        print("E2E smoke test passed: route -> plan -> code -> check -> validate -> deploy")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
