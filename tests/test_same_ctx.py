"""Same-ctx rule (offline, fake Ollama HTTP server): roles that resolve to the
same model send the SAME num_ctx on every call, so a full code turn loads each
model exactly once (a ctx change would force a reload)."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_config import (
    AgentConfig,
    effective_ctx_for_model,
)

CODER = "qwen2.5-coder:3b-instruct"
THINKER = "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"


# --- unit: the effective-ctx rule itself ----------------------------------------

def test_effective_ctx_same_model_one_ctx() -> None:
    """coder=4096 + understand=6144 on the SAME model -> both send 6144
    (the legacy 4096 leak is overridden by the per-model rule)."""
    cfg = AgentConfig(roles={
        "coder": {"model": CODER, "num_ctx": 4096},
        "understand": {"model": CODER, "num_ctx": 6144},
    })
    assert effective_ctx_for_model(cfg, CODER) == 6144
    # thinker/validator/writer share the thinker model at 8192 by default
    assert effective_ctx_for_model(cfg, THINKER) == 8192


def test_effective_ctx_clamped_by_ceiling() -> None:
    cfg = AgentConfig(roles={
        "coder": {"model": CODER, "num_ctx": 16384},
        "understand": {"model": CODER, "num_ctx": 16384},
    })
    # coder/understand ceilings are 12288 -> clamped there
    assert effective_ctx_for_model(cfg, CODER) == 12288


def test_effective_ctx_unknown_model_falls_back() -> None:
    cfg = AgentConfig()
    assert effective_ctx_for_model(cfg, "no-such-model:9b") == 16384


def test_legacy_max_context_tokens_ignored() -> None:
    """Legacy max_context_tokens: 4096 is ignored; role defaults stand
    (thinker 8192, coder 6144, validator follows thinker)."""
    cfg = AgentConfig(max_context_tokens=4096)
    assert int(cfg.roles["thinker"]["num_ctx"]) == 8192
    assert int(cfg.roles["coder"]["num_ctx"]) == 6144
    assert cfg.roles["validator"]["model"] == cfg.roles["thinker"]["model"]
    assert int(cfg.roles["validator"]["num_ctx"]) == 8192


# --- integration: one load per model per code turn (fake Ollama) -----------------

class _CountingOllama(BaseHTTPRequestHandler):
    """Fake Ollama that counts a LOAD whenever a generate arrives with a
    num_ctx different from the model's currently-loaded ctx."""

    resident: dict[str, int] = {}      # model -> num_ctx currently loaded
    loads: list[tuple[str, int]] = []  # (model, num_ctx) per actual load
    calls: list[tuple[str, int]] = []  # (model, num_ctx) per generate call
    loop_n: int = 0                    # agent-loop call counter (per test)

    @classmethod
    def reset(cls) -> None:
        cls.resident = {}
        cls.loads = []
        cls.calls = []
        cls.loop_n = 0

    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = {"models": [{"name": CODER}, {"name": THINKER}]}
        elif self.path.endswith("/api/ps"):
            body = {"models": [
                {"name": m, "size": 1000, "size_vram": 1000}
                for m in self.resident
            ]}
        else:
            body = {"status": "ok"}
        self._json(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        model = payload.get("model", "")
        if payload.get("keep_alive") == 0:  # unload
            self.resident.pop(model, None)
            self._json({})
            return
        ctx = int((payload.get("options") or {}).get("num_ctx", 0))
        type(self).calls.append((model, ctx))
        if self.resident.get(model) != ctx:
            type(self).loads.append((model, ctx))
            type(self).resident = {model: ctx}  # one model resident at a time
        prompt = payload.get("prompt", "")
        if "AGENT ACTION JSON:" in prompt:
            # The loop: write notes.md, then finish. Drive by call count per
            # model so the write's coder call (same model) never confuses it.
            type(self).loop_n = getattr(type(self), "loop_n", 0) + 1
            actions = [
                {"action": "write_file", "path": "notes.md",
                 "text": "Markdown notes file with a short todo list inside"},
                {"action": "finish"},
            ]
            text = json.dumps(actions[min(type(self).loop_n - 1, len(actions) - 1)])
        elif "TARGET FILE:" in prompt or "Write notes.md." in prompt:
            text = "```markdown\n# Notes\n\n- [ ] thing\n```"
        elif "satisfies_request" in prompt:
            text = json.dumps(
                {"satisfies_request": True, "problems": [], "summary": "ok"})
        else:
            text = "ok"
        self._json({
            "response": text, "done": True, "done_reason": "stop",
            "prompt_eval_count": 64, "eval_count": 16,
            "total_duration": 100_000_000, "eval_duration": 50_000_000,
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


@pytest.fixture()
def fake_ollama():
    _CountingOllama.reset()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CountingOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_one_load_per_model_per_code_turn(fake_ollama: str) -> None:
    """classify + verify + plan + code all run on the coder model; validate +
    summary on the thinker. With the same-ctx rule, each model loads ONCE."""
    from ollama_client import AgentGateway, OllamaSupervisorClient
    from orchestrator import run_pipeline

    with TemporaryDirectory() as tmp:
        config = AgentConfig(roles={
            # the defective legacy pair: coder 4096 vs understand 6144
            "coder": {"model": CODER, "num_ctx": 4096},
            "understand": {"model": CODER, "num_ctx": 6144},
        })

        supervisor = OllamaSupervisorClient(base_url=fake_ollama)
        gateway = AgentGateway(supervisor, config, watchdog=False)

        async def generate(model, prompt, system="", temperature=0.2, **kwargs):
            role = kwargs.pop("role", None) or (
                "coder" if model == CODER else "thinker")
            return await gateway.generate(
                role, prompt, system=system, temperature=temperature, **kwargs)

        result = asyncio.run(run_pipeline(
            "make a notes.md with my todo list",
            tmp,
            generate=generate,
            planner_model=THINKER,
            coder_model=CODER,
            understand_model=CODER,
            num_ctx=8192,
        ))
        assert result["staged_files"] == ["notes.md"]

    # Every load of the coder model used the SAME num_ctx -> exactly one load.
    coder_loads = [ctx for m, ctx in _CountingOllama.loads if m == CODER]
    thinker_loads = [ctx for m, ctx in _CountingOllama.loads if m == THINKER]
    assert len(coder_loads) == 1, _CountingOllama.loads
    assert coder_loads[0] == 6144  # the per-model effective ctx (max wins)
    assert len(set(thinker_loads)) <= 1
    models_loaded = {m for m, _ in _CountingOllama.loads}
    assert models_loaded == {CODER, THINKER}
    # ...and EVERY call to the coder model carried that same num_ctx on the wire.
    coder_call_ctxs = {ctx for m, ctx in _CountingOllama.calls if m == CODER}
    assert coder_call_ctxs == {6144}, coder_call_ctxs
