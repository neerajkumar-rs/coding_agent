"""Offline end-to-end test for the complete run_pipeline using a local Ollama mock.

Runs ROUTE -> CLASSIFY -> VERIFY -> PLAN -> CODE -> CHECK -> VALIDATE ->
SUMMARY against an in-process fake Ollama, then deploys via the staging
manager (the pipeline itself never deploys). Also asserts the same-ctx rule:
the coder model loads exactly once (a num_ctx change would force a reload).
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_core.run_state import RUNS_DIR_NAME
from orchestrator import run_pipeline

CODER = "qwen2.5-coder:3b-instruct"
THINKER = "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"


class FakeOllama(BaseHTTPRequestHandler):
    resident: dict[str, int] = {}
    loads: list[tuple[str, int]] = []

    @classmethod
    def reset(cls) -> None:
        cls.resident = {}
        cls.loads = []

    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = {"models": [{"name": CODER}, {"name": THINKER}]}
        elif self.path.endswith("/api/ps"):
            body = {"models": [
                {"name": m, "size": 1000, "size_vram": 1000} for m in self.resident
            ]}
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
        model = payload.get("model", "")
        if payload.get("keep_alive") == 0:
            self.resident.pop(model, None)
            self._json({})
            return
        ctx = int((payload.get("options") or {}).get("num_ctx", 0))
        if self.resident.get(model) != ctx:
            type(self).loads.append((model, ctx))
            type(self).resident = {model: ctx}
        prompt = payload.get("prompt", "")
        if "CLASSIFY MODE:" in prompt:
            content = "code"
        elif "VERIFY SIDE EFFECT:" in prompt:
            content = "yes"
        elif "PLAN TASKS JSON:" in prompt:
            content = json.dumps({"tasks": [{
                "path": "generated.py", "action": "create",
                "description": "Python module with a greet function returning a greeting",
            }], "assumptions": []})
        elif "TARGET FILE:" in prompt:
            content = ("```python\ndef greet(name: str) -> str:\n"
                       "    return f'Hello, {name}!'\n```")
        elif "satisfies_request" in prompt:
            content = json.dumps(
                {"satisfies_request": True, "problems": [], "summary": "A greet module."})
        elif "Describe only features visible" in prompt:
            content = "Defines a greet function returning a greeting."
        else:
            content = "ok"
        self._json({
            "response": content, "done": True, "done_reason": "stop",
            "prompt_eval_count": 128, "eval_count": 64,
            "total_duration": 1_000_000_000, "eval_duration": 500_000_000,
        })

    def _json(self, body: dict) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        return


def main() -> None:
    FakeOllama.reset()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with TemporaryDirectory() as workspace:
            from agent_config import AgentConfig
            from ollama_client import AgentGateway, OllamaSupervisorClient

            base = f"http://127.0.0.1:{server.server_address[1]}"
            config = AgentConfig()
            gateway = AgentGateway(
                OllamaSupervisorClient(base_url=base), config, watchdog=False)

            async def generate(model, prompt, system="", temperature=0.2, **kwargs):
                role = kwargs.pop("role", None) or (
                    "coder" if model == CODER else "thinker")
                return await gateway.generate(
                    role, prompt, system=system, temperature=temperature, **kwargs)

            result = asyncio.run(run_pipeline(
                "Create a greeting function in generated.py",
                workspace,
                generate=generate,
                planner_model=THINKER,
                coder_model=CODER,
                understand_model=CODER,
            ))
            workspace_root = Path(workspace)
            run_id = result["run_id"]

            assert result.get("status") in (None, "completed"), result
            assert run_id, "run must carry a run_id"
            assert result["staged_files"] == ["generated.py"], result

            # Deploy is explicit (the pipeline never deploys itself).
            from workspace_staging import WorkspaceStagingManager

            outcome = WorkspaceStagingManager(workspace_root).deploy_staged_files()
            assert outcome["status"] == "success"
            target = workspace_root / "generated.py"
            assert target.exists()

            # Junk-file regression: no fake-role artifacts in the workspace root
            assert not (workspace_root / "summary.txt").exists()
            assert not (workspace_root / "tasks.md").exists()
            assert not list(workspace_root.glob("task*_code.txt"))

            # Run metadata lives under .agent_runs/<run_id>/
            run_dir = workspace_root / RUNS_DIR_NAME / run_id
            assert (run_dir / "state.json").exists()
            assert (run_dir / "events.jsonl").exists()
            state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            assert state["status"] == "completed", state
            # Per-call records exist for the debug export
            assert sorted((run_dir / "calls").glob("*.json")), "no calls/*.json written"

            # Same-ctx rule: ONE load per model per turn (no ctx-change reload).
            coder_loads = [c for m, c in FakeOllama.loads if m == CODER]
            thinker_loads = [c for m, c in FakeOllama.loads if m == THINKER]
            assert len(coder_loads) == 1, FakeOllama.loads
            assert len(set(thinker_loads)) <= 1, FakeOllama.loads

            # Only the requested target plus benign git-ignored infrastructure.
            root_entries = {p.name for p in workspace_root.iterdir()}
            assert root_entries <= {
                "generated.py", ".agent", RUNS_DIR_NAME,
                ".agent_staging", "__pycache__",
            }, root_entries
            staging_dir = workspace_root / ".agent_staging"
            if staging_dir.exists():
                leftovers = [p for p in staging_dir.rglob("*") if p.is_file()]
                assert not leftovers, leftovers
        print("E2E smoke test passed: classify -> verify -> plan -> code -> check "
              "-> validate -> deploy (one load per model)")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
