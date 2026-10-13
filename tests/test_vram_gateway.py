"""Phase 3 tests: single-lock AgentGateway + VRAM guard.

FakeOllama here tracks loaded models in memory: /api/ps reflects what is
resident and reports size/size_vram so the GPU-fraction guard can be driven.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_config import AgentConfig
from ollama_client import (
    AgentGateway,
    OllamaSupervisorClient,
    VramGuardError,
)

MODEL = "qwen2.5-coder:3b-instruct"
THINKER = "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"


class FakeOllama(BaseHTTPRequestHandler):
    """In-memory residency model. `loaded` = {name: {"size":, "size_vram":}}."""

    loaded: dict[str, dict] = {}
    # Behavior knobs, set per-test:
    # how much of a freshly loaded model lands on GPU (0..1); callable(num_ctx)
    gpu_fraction = 1.0
    fail_times: int = 0          # fail the next N generates with HTTP 500
    oom_until_ctx: int = 0       # 500-OOM when num_ctx > this (0 = off)
    requests: list[dict] = []    # observed payloads

    MODEL_SIZES = {MODEL: 1_900_000_000, THINKER: 1_100_000_000}

    @classmethod
    def reset(cls) -> None:
        cls.loaded = {}
        cls.gpu_fraction = 1.0
        cls.fail_times = 0
        cls.oom_until_ctx = 0
        cls.requests = []

    # -- helpers -------------------------------------------------------------

    @classmethod
    def _frac(cls, num_ctx: int) -> float:
        return cls.gpu_fraction(num_ctx) if callable(cls.gpu_fraction) else cls.gpu_fraction

    def do_GET(self) -> None:
        if self.path.endswith("/api/tags"):
            body = json.dumps({"models": [{"name": m} for m in type(self).MODEL_SIZES]}).encode()
        elif self.path.endswith("/api/ps"):
            models = [{"name": n, **v} for n, v in type(self).loaded.items()]
            body = json.dumps({"models": models}).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self._send(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(payload)
        model = payload.get("model", "")

        if payload.get("keep_alive") == 0:  # unload
            type(self).loaded.pop(model, None)
            self._send(b"{}"); return

        # a generate: simulate a load (evicting nothing; Ollama would, but our
        # gateway unloads first) then maybe fail
        num_ctx = payload.get("options", {}).get("num_ctx", 4096)
        if type(self).fail_times > 0:
            type(self).fail_times -= 1
            type(self).loaded[model] = {
                "size": self.MODEL_SIZES[model],
                "size_vram": int(self.MODEL_SIZES[model] * self._frac(num_ctx)),
            }
            self._send(json.dumps({"error": "out of memory (runner stopped)"}).encode(), 500)
            return
        if type(self).oom_until_ctx and num_ctx > type(self).oom_until_ctx:
            type(self).loaded[model] = {
                "size": self.MODEL_SIZES[model],
                "size_vram": int(self.MODEL_SIZES[model] * self._frac(num_ctx)),
            }
            self._send(json.dumps({"error": "out of memory"}).encode(), 500)
            return

        type(self).loaded[model] = {
            "size": self.MODEL_SIZES[model],
            "size_vram": int(self.MODEL_SIZES[model] * self._frac(num_ctx)),
        }
        body = json.dumps({
            "response": "OK", "done": True, "done_reason": "stop",
            "prompt_eval_count": 10, "eval_count": 2, "total_duration": 1_000_000,
        }).encode()
        self._send(body)

    def _send(self, body: bytes, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a: object) -> None:
        return


def _serve() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _gateway(base: str, watchdog: bool = False) -> AgentGateway:
    sup = OllamaSupervisorClient(base_url=base)
    cfg = AgentConfig()
    return AgentGateway(sup, cfg, watchdog=watchdog)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset():
    FakeOllama.reset()
    yield
    FakeOllama.reset()


# --- 1. one resident model at a time + swap logging ---------------------------

def test_swap_unloads_other_model_first() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        # thinker resident, then generate with coder -> must evict thinker
        FakeOllama.loaded[THINKER] = {"size": 1_100_000_000, "size_vram": 1_100_000_000}
        out = _run(gw.generate("coder", "ping", keep_warm=False))
        assert out == "OK"
        # after generation (not keep_warm), nothing resident
        assert FakeOllama.loaded == {}
        # the coder was loaded at some point: a generate request went out
        assert any(r.get("model") == MODEL and r.get("prompt") == "ping"
                   for r in FakeOllama.requests)
        # and an unload of the thinker happened
        assert any(r.get("model") == THINKER and r.get("keep_alive") == 0
                   for r in FakeOllama.requests)
    finally:
        server.shutdown(); server.server_close()


# --- 2. 0.97 enforcement + step-down ------------------------------------------

def test_gpu_fraction_enforced_with_stepdown() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)

        # coder num_ctx=6144: passes only when ctx small enough
        def frac(num_ctx: int) -> float:
            return 0.99 if num_ctx <= 4096 else 0.60  # spills above 4096

        FakeOllama.gpu_fraction = frac
        out = _run(gw.generate("coder", "ping"))
        assert out == "OK"
        # Final successful generate used num_ctx 4096 (the passing rung)
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "ping"]
        assert gen[-1]["options"]["num_ctx"] == 4096
    finally:
        server.shutdown(); server.server_close()


def test_vram_guard_raises_at_lowest_rung() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        FakeOllama.gpu_fraction = 0.5  # never fits, even at 4096
        with pytest.raises(VramGuardError):
            _run(gw.generate("coder", "ping"))
    finally:
        server.shutdown(); server.server_close()


# --- 3. OOM / runner-stopped -> same step-down retry --------------------------

def test_oom_triggers_stepdown_retry() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        # OOM whenever ctx > 4096; fits at 4096
        FakeOllama.oom_until_ctx = 4096
        out = _run(gw.generate("coder", "ping"))
        assert out == "OK"
        gen = [r for r in FakeOllama.requests if r.get("prompt") == "ping"]
        assert gen[-1]["options"]["num_ctx"] == 4096
        # at least two generate attempts happened (6144 OOM, then 4096 ok)
        assert len(gen) >= 2
    finally:
        server.shutdown(); server.server_close()


# --- 4. keep_warm batching: no unload between same-model calls -----------------

def test_keep_warm_skips_unload_between_calls() -> None:
    server, base = _serve()
    try:
        gw = _gateway(base)
        _run(gw.generate("coder", "one", keep_warm=True))
        # still resident after keep_warm
        assert MODEL in FakeOllama.loaded
        _run(gw.generate("coder", "two", keep_warm=True))
        assert MODEL in FakeOllama.loaded
        # final non-warm call unloads
        _run(gw.generate("coder", "three", keep_warm=False))
        assert FakeOllama.loaded == {}
    finally:
        server.shutdown(); server.server_close()


# --- 5. cancel unloads (cancel event set mid-call) -----------------------------

def test_cancel_unloads_and_stops() -> None:
    import threading as th

    server, base = _serve()
    try:
        gw = _gateway(base)
        cancel = th.Event()
        cancel.set()  # set before the call
        from ollama_client import OllamaCancelledError
        with pytest.raises(OllamaCancelledError):
            _run(gw.generate("coder", "ping", cancel_event=cancel))
        # nothing left resident
        assert FakeOllama.loaded == {}
    finally:
        server.shutdown(); server.server_close()


# --- 6. single shared lock between gateway and supervisor ---------------------

def test_gateway_and_supervisor_share_one_lock() -> None:
    server, base = _serve()
    try:
        sup = OllamaSupervisorClient(base_url=base)
        gw = AgentGateway(sup, AgentConfig(), watchdog=False)
        assert gw._lock is sup._vram_mutex  # exactly ONE lock in the process
    finally:
        server.shutdown(); server.server_close()


# --- 7. structured output goes through the same lock + guard -----------------

def test_structured_generate_uses_format_and_guard() -> None:
    from agent_core.schemas import ValidateResult

    server, base = _serve()
    try:
        # fake returns a JSON object for structured prompts
        class StructFake(FakeOllama):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                model = payload.get("model", "")
                if payload.get("keep_alive") == 0:
                    FakeOllama.loaded.pop(model, None)
                    self._send(b"{}"); return
                FakeOllama.loaded[model] = {
                    "size": self.MODEL_SIZES[model], "size_vram": self.MODEL_SIZES[model],
                }
                assert "format" in payload  # schema-constrained
                body = json.dumps({
                    "response": '{"satisfies_request": true, "problems": [], "summary": "ok"}',
                    "done": True, "done_reason": "stop",
                }).encode()
                self._send(body)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), StructFake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            gw = _gateway(f"http://127.0.0.1:{srv.server_port}")
            out = _run(gw.structured("validator", "review", ValidateResult))
            assert out.satisfies_request is True
            # structured goes through generate(keep_warm default True) -> stays warm
            assert THINKER in FakeOllama.loaded
        finally:
            srv.shutdown(); srv.server_close()
    finally:
        server.shutdown(); server.server_close()


# --- 8. phase batching: thinker phase -> coder phase -> thinker ---------------

def test_phase_batched_swaps_logged() -> None:
    """route/plan (thinker) -> code (coder) -> validate (thinker): each phase
    keeps its model warm; exactly two swaps (thinker->coder, coder->thinker),
    each logged with a duration; same-model calls never reload."""
    import logging

    server, base = _serve()
    try:
        gw = _gateway(base)
        records = []
        h = logging.Handler()
        h.emit = lambda r: records.append(r.getMessage())
        logging.getLogger("ollama_client").addHandler(h)
        logging.getLogger("ollama_client").setLevel(logging.INFO)

        async def go():
            # thinker phase (two calls, no reload between)
            await gw.generate("thinker", "route")
            await gw.generate("thinker", "plan")
            # coder phase (two calls, no reload between)
            await gw.generate("coder", "code1")
            await gw.generate("coder", "code2")
            # thinker phase again
            await gw.generate("thinker", "validate")

        _run(go())

        swaps = [m for m in records if "swap" in m and "took" in m]
        # exactly two model swaps: thinker->coder and coder->thinker
        assert len(swaps) == 2, swaps
        # the swap targets are correct
        assert "-> " + MODEL in swaps[0]
        assert "-> " + THINKER in swaps[1]
        # same-model second calls (plan, code2) did NOT trigger a swap: only 2
        gens = [r for r in FakeOllama.requests if r.get("prompt") in
                ("route", "plan", "code1", "code2", "validate")]
        assert len(gens) == 5  # each prompt generated once
    finally:
        server.shutdown(); server.server_close()
