"""High-performance asynchronous Ollama API client with strict VRAM fencing.

Post-processing guarantees (applied in THIS layer, before any caller parses):
- ``<think>...</think>`` blocks (including an unterminated trailing block) are
  stripped from every response and returned separately as *reasoning*.
- A response whose ``done_reason`` is ``"length"`` was cut off. It is never
  handed back for judgement: :class:`OllamaTruncatedError` (retryable) is
  raised so callers can retry with an adjusted budget instead of failing.

Ollama payload rules:
- ``keep_alive`` is a *top-level* request field (never inside ``options``).
- ``options`` carries only: ``num_ctx``, ``num_predict``, ``temperature``, ``seed``.
  ``flash_attention`` is NOT a request option — it is the *server* env var
  ``OLLAMA_FLASH_ATTENTION=1``. ``OLLAMA_MAX_LOADED_MODELS=1`` is also a server
  env var. ``tui.py`` sets both when it spawns ``ollama serve``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx

from agent_config import AgentConfig, load_agent_config

logger = logging.getLogger("ollama_client")

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

# Generation budgets -----------------------------------------------------------
# Reasoning models (deepseek-r1 style) need extra room for <think> + answer.
DEFAULT_NUM_PREDICT = 2048
REASONING_NUM_PREDICT = 3072
MAX_NUM_PREDICT = 4096
TEMPERATURE_CODE = 0.1
TEMPERATURE_PLAN = 0.2
DEFAULT_SEED = 42

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>(.*)$", re.DOTALL | re.IGNORECASE)


class OllamaError(RuntimeError):
    """Raised when Ollama inference fails or returns an error status."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        done_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.done_reason = done_reason


class ModelNotInstalled(OllamaError):
    """The exact configured model is not in /api/tags. Never substitute."""

    def __init__(self, requested: str, installed: list[str] | None = None) -> None:
        self.requested = requested
        self.installed = installed or []
        hint = f"ollama pull {requested}"
        detail = f"model not installed: {requested!r}. Run: {hint}"
        if self.installed:
            detail += f"\nInstalled: {', '.join(self.installed)}"
        super().__init__(detail)


class OllamaCancelledError(OllamaError):
    """The in-flight request was aborted because the run was cancelled."""


class OllamaTruncatedError(OllamaError):
    """Response was cut off (``done_reason == "length"``). Always retryable."""

    def __init__(self, message: str = "response truncated (done_reason=length)") -> None:
        super().__init__(message, retryable=True, done_reason="length")


class VramGuardError(OllamaError):
    """A model could not be loaded with >= min_gpu_fraction of its weights on
    the GPU even at the lowest context rung. Never fall back to CPU silently."""


# Context rungs for the VRAM guard's step-down retry.
CTX_RUNGS = [4096, 6144, 8192, 12288, 16384]


def _gpu_memory_mb() -> tuple[int, int] | None:
    """nvidia-smi (memory.used, memory.total) in MB, or None when unavailable."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            used_s, total_s = out.stdout.strip().splitlines()[0].split(",")
            return int(float(used_s)), int(float(total_s))
    except Exception:
        pass
    return None


def _log_vram(tag: str) -> None:
    """nvidia-smi watchdog: log memory.used, warn when free < 300 MB."""
    mem = _gpu_memory_mb()
    if mem is None:
        return  # silent when nvidia-smi is absent
    used, total = mem
    free = total - used
    logger.info(f"[nvidia-smi] memory.used {tag}: {used} MB (free {free} MB)")
    if free < 300:
        logger.warning(f"[nvidia-smi] low free VRAM after {tag}: {free} MB (< 300 MB)")


def _print_sysmem_hint_once() -> None:
    """One-time Windows hint: prefer 'No Sysmem Fallback' for CUDA."""
    import sys

    if getattr(_print_sysmem_hint_once, "_done", False):
        return
    if sys.platform == "win32":
        logger.info(
            "Windows: set NVIDIA Control Panel > CUDA - Sysmem Fallback Policy "
            "to 'Prefer No Sysmem Fallback' so a spilled model never silently "
            "runs on system RAM."
        )
    _print_sysmem_hint_once._done = True


def strip_think_blocks(raw_text: str) -> tuple[str, str]:
    """Remove ``<think>...</think>`` blocks; return ``(reasoning, answer)``.

    Handles an unterminated trailing ``<think>`` block (common when a reasoning
    model is cut off mid-thought): its content is treated as reasoning and
    removed from the answer.
    """
    if not raw_text:
        return "", ""
    reasoning_parts = [m.strip() for m in _THINK_BLOCK_RE.findall(raw_text)]
    text = _THINK_BLOCK_RE.sub("", raw_text)
    open_match = _THINK_OPEN_RE.search(text)
    if open_match:
        reasoning_parts.append(open_match.group(1).strip())
        text = text[: open_match.start()]
    reasoning = "\n".join(part for part in reasoning_parts if part)
    return reasoning, text.strip()


def is_reasoning_model(model_name: str) -> bool:
    """Heuristic: models that emit ``<think>`` blocks by default."""
    lowered = model_name.lower()
    return "deepseek-r1" in lowered or "qwq" in lowered or lowered.startswith("r1")


@dataclass
class LLMResponse:
    """Fully post-processed model response."""

    text: str
    reasoning: str
    model: str
    done_reason: str = "stop"
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def _default_num_predict(model_name: str) -> int:
    return REASONING_NUM_PREDICT if is_reasoning_model(model_name) else DEFAULT_NUM_PREDICT


class OllamaSupervisorClient:
    """Pure Python deterministic supervisor enforcing sequential model execution.

    Operates at 0MB VRAM footprint. Models are called strictly sequentially as
    stateless functions via an async mutex to prevent memory exhaustion on 4GB VRAM.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_HOST,
        timeout: float = 600.0,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"
        self.timeout = timeout
        # The single process-wide lock. When an AgentGateway wraps this client
        # it injects its own lock here so there is exactly ONE.
        self._vram_mutex = lock if lock is not None else asyncio.Lock()
        self._cached_installed_models: list[str] | None = None

    async def list_installed_models(self) -> list[str]:
        """Fetch list of local models available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(self.tags_url)
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    self._cached_installed_models = models
                    return models
        except Exception as exc:
            logger.warning(f"Could not query Ollama tags: {exc}")
        return self._cached_installed_models or []

    async def resolve_model_name(self, requested: str) -> str:
        """Exact-match the configured model against /api/tags.

        No fuzzy substitution, ever. The only normalization is Ollama's own:
        an untagged name matches its ``:latest`` tag. Otherwise raise
        :class:`ModelNotInstalled` with the exact ``ollama pull <name>``.
        """
        installed = await self.list_installed_models()
        if not installed:
            # Cannot verify (offline/tags failed): pass the name through and let
            # the generate call surface any real error.
            return requested
        # Exact match.
        if requested in installed:
            return requested
        # Ollama's untagged -> :latest normalization.
        if ":" not in requested and f"{requested}:latest" in installed:
            return f"{requested}:latest"
        raise ModelNotInstalled(requested, installed)

    def build_payload(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        config: AgentConfig | None = None,
        stream: bool = False,
        num_predict: int | None = None,
        temperature: float = TEMPERATURE_CODE,
        seed: int = DEFAULT_SEED,
    ) -> dict[str, Any]:
        """Build a valid Ollama /api/generate payload.

        ``keep_alive`` is top-level; ``options`` carries only num_ctx /
        num_predict / temperature / seed.
        """
        cfg = config or load_agent_config()
        return {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "keep_alive": cfg.keep_alive,
            "options": {
                "num_ctx": cfg.max_context_tokens,
                "num_predict": num_predict if num_predict is not None else _default_num_predict(model),
                "temperature": temperature,
                "seed": seed,
            },
        }

    async def _post_cancellable(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        cancel_event: Any | None,
    ) -> dict[str, Any]:
        """POST /api/generate, aborting the in-flight request the moment the
        workspace's cancel event is set. Closing the client drops the HTTP
        connection, which makes Ollama stop the runner promptly."""
        if cancel_event is None:
            response = await client.post(self.generate_url, json=payload)
            if response.status_code != 200:
                raise OllamaError(
                    f"Ollama generation failed ({response.status_code}): {response.text}"
                )
            return response.json()

        async def _watch() -> None:
            while not cancel_event.is_set():
                await asyncio.sleep(0.1)

        post_task = asyncio.ensure_future(client.post(self.generate_url, json=payload))
        watch_task = asyncio.ensure_future(_watch())
        try:
            done, _pending = await asyncio.wait(
                {post_task, watch_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if watch_task in done and cancel_event.is_set():
                post_task.cancel()
                await client.aclose()  # drop the connection: aborts the runner
                raise OllamaCancelledError("run cancelled by user")
            watch_task.cancel()
            response = post_task.result()
            if response.status_code != 200:
                raise OllamaError(
                    f"Ollama generation failed ({response.status_code}): {response.text}"
                )
            return response.json()
        finally:
            for task in (post_task, watch_task):
                if not task.done():
                    task.cancel()

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str = "",
        model_name: str | None = None,
        config: AgentConfig | None = None,
        num_predict: int | None = None,
        temperature: float = TEMPERATURE_PLAN,
        seed: int = DEFAULT_SEED,
        cancel_event: Any | None = None,
    ) -> LLMResponse:
        """Sequential stateless generation with think-stripping and truncation guard.

        Never returns a cut-off response for judgement: if Ollama reports
        ``done_reason == "length"`` and the output hit the ``num_predict``
        ceiling, we retry once with a larger budget (bounded by the remaining
        context window); otherwise :class:`OllamaTruncatedError` is raised so
        the caller can shrink the prompt.
        """
        cfg = config or load_agent_config()
        target_model = model_name or cfg.planner_model
        resolved = await self.resolve_model_name(target_model)
        effective_predict = num_predict if num_predict is not None else _default_num_predict(resolved)

        for attempt in range(2):
            payload = self.build_payload(
                model=resolved,
                prompt=prompt,
                system_prompt=system_prompt,
                config=cfg,
                stream=False,
                num_predict=effective_predict,
                temperature=temperature,
                seed=seed,
            )

            async with self._vram_mutex:
                data = await self._generate_transport(
                    resolved, payload, cfg, effective_predict, temperature, cancel_event
                )

            raw_text = data.get("response", "")
            done_reason = data.get("done_reason") or "stop"
            prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
            output_tokens = int(data.get("eval_count", 0) or 0)
            seconds = round(float(data.get("total_duration", 0) or 0) / 1e9, 3)

            if done_reason == "length":
                ctx_headroom = cfg.max_context_tokens - prompt_tokens
                bigger = min(effective_predict * 2, MAX_NUM_PREDICT, max(ctx_headroom, 0))
                if output_tokens >= effective_predict and bigger > effective_predict and attempt == 0:
                    logger.warning(
                        f"Response from {resolved} hit num_predict={effective_predict}; "
                        f"retrying once with num_predict={bigger}"
                    )
                    effective_predict = bigger
                    continue
                raise OllamaTruncatedError(
                    f"response from {resolved} was cut off at the context window "
                    f"(prompt={prompt_tokens} tok, output={output_tokens} tok, "
                    f"num_ctx={cfg.max_context_tokens}); retry with a smaller prompt"
                )

            reasoning, answer = strip_think_blocks(raw_text)
            return LLMResponse(
                text=answer,
                reasoning=reasoning,
                model=resolved,
                done_reason=done_reason,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                seconds=seconds,
            )

        raise OllamaTruncatedError(  # unreachable, but keeps type-checkers honest
            f"response from {resolved} was cut off repeatedly"
        )

    async def _generate_transport(
        self,
        resolved: str,
        payload: dict[str, Any],
        cfg: AgentConfig,
        num_predict: int,
        temperature: float,
        cancel_event: Any | None,
    ) -> dict[str, Any]:
        """Raw generate POST. Caller must hold ``self._vram_mutex``."""
        logger.info(
            f"[Supervisor Lock Acquired] Running {resolved} "
            f"(num_ctx={cfg.max_context_tokens}, num_predict={num_predict}, "
            f"temp={temperature}, keep_alive={cfg.keep_alive})"
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._post_cancellable(client, payload, cancel_event)

    async def generate_unlocked(
        self,
        prompt: str,
        system_prompt: str = "",
        model_name: str | None = None,
        config: AgentConfig | None = None,
        num_predict: int | None = None,
        temperature: float = TEMPERATURE_PLAN,
        cancel_event: Any | None = None,
        format_schema: dict[str, Any] | None = None,
    ) -> str:
        """Lock-free generate for a caller (AgentGateway) that already holds
        the single process lock. Identical to ``generate`` minus the mutex.
        ``format_schema`` adds Ollama structured output (format=<json schema>)."""
        cfg = config or load_agent_config()
        resolved = await self.resolve_model_name(model_name or cfg.planner_model)
        effective_predict = num_predict if num_predict is not None else _default_num_predict(resolved)
        payload = self.build_payload(
            model=resolved, prompt=prompt, system_prompt=system_prompt, config=cfg,
            stream=False, num_predict=effective_predict, temperature=temperature,
        )
        if format_schema is not None:
            payload["format"] = format_schema
        data = await self._generate_transport(
            resolved, payload, cfg, effective_predict, temperature, cancel_event
        )
        raw_text = data.get("response", "")
        if (data.get("done_reason") or "stop") == "length":
            raise OllamaTruncatedError(
                f"response from {resolved} was cut off (done_reason=length); retry with a smaller prompt"
            )
        _, answer = strip_think_blocks(raw_text)
        return answer

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model_name: str | None = None,
        config: AgentConfig | None = None,
        num_predict: int | None = None,
        temperature: float = TEMPERATURE_PLAN,
        cancel_event: Any | None = None,
    ) -> str:
        """Compatibility wrapper: return only the think-stripped answer text."""
        response = await self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            model_name=model_name,
            config=config,
            num_predict=num_predict,
            temperature=temperature,
            cancel_event=cancel_event,
        )
        return response.text

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model_name: str | None = None,
        config: AgentConfig | None = None,
        num_predict: int | None = None,
        temperature: float = TEMPERATURE_CODE,
    ) -> AsyncGenerator[str, None]:
        """Stream raw generated tokens while holding the VRAM supervisor lock.

        Raw transport: callers that parse the joined output must run it through
        :func:`strip_think_blocks` first. Raises :class:`OllamaTruncatedError`
        at end-of-stream if the response was cut off.
        """
        cfg = config or load_agent_config()
        target_model = model_name or cfg.coder_model
        resolved = await self.resolve_model_name(target_model)
        payload = self.build_payload(
            model=resolved,
            prompt=prompt,
            system_prompt=system_prompt,
            config=cfg,
            stream=True,
            num_predict=num_predict,
            temperature=temperature,
        )

        async with self._vram_mutex:
            logger.info(
                f"[Supervisor Lock Acquired for Stream] Running {resolved} "
                f"(num_ctx={cfg.max_context_tokens}, keep_alive={cfg.keep_alive})"
            )
            done_reason = "stop"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.generate_url, json=payload) as resp:
                    if resp.status_code != 200:
                        content = await resp.aread()
                        raise OllamaError(
                            f"Ollama stream failed ({resp.status_code}): {content.decode('utf-8', errors='replace')}"
                        )
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("done"):
                            done_reason = chunk.get("done_reason") or "stop"
                        token = chunk.get("response", "")
                        if token:
                            yield token
            if done_reason == "length":
                raise OllamaTruncatedError(
                    f"streamed response from {resolved} was cut off (done_reason=length)"
                )


class AgentGateway:
    """THE single process-wide LLM gateway: one lock, VRAM guard, role config.

    generate(role, prompt, ...) reads model/num_ctx/num_predict/temperature/
    keep_alive from the role config. Every generation in the process funnels
    through here under ONE asyncio.Lock — the only lock on the hot path.

    VRAM guard (per load of model M):
      1. /api/ps: unload every resident model != M (keep_alive=0), poll until
         empty (15 s timeout, else error).
      2. Generate. Then /api/ps again: require size_vram/size >= min_gpu_fraction
         (default 0.97). If not, unload M, step num_ctx down one rung of
         CTX_RUNGS and retry; at the lowest rung raise VramGuardError.
      OOM / "runner stopped" / HTTP 500 => same step-down retry.
      Never fall back to CPU silently.

    Phase batching: callers pass keep_warm=True to skip the unload between
    same-model calls (route+plan, then all-code, etc.); a swap is logged with
    its duration. nvidia-smi watchdog logs memory.used per call (silent when
    nvidia-smi is absent; warns when free < 300 MB).
    """

    def __init__(
        self,
        supervisor: OllamaSupervisorClient,
        config: AgentConfig | None = None,
        min_gpu_fraction: float = 0.97,
        watchdog: bool = True,
    ) -> None:
        self.supervisor = supervisor
        self.config = config or load_agent_config()
        self.min_gpu_fraction = min_gpu_fraction
        self.watchdog = watchdog
        # THE one process-wide lock. Injected into the supervisor so the whole
        # process shares exactly one lock on the hot path.
        self._lock = asyncio.Lock()
        self.supervisor._vram_mutex = self._lock
        self._resident_model: str | None = None
        _print_sysmem_hint_once()

    @property
    def base_url(self) -> str:
        return self.supervisor.base_url

    async def _ps(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        try:
            resp = await client.get(f"{self.base_url}/api/ps")
            if resp.status_code == 200:
                return resp.json().get("models", [])
        except httpx.HTTPError:
            pass
        return []

    async def _unload(self, client: httpx.AsyncClient, model: str) -> None:
        try:
            await client.post(
                self.supervisor.generate_url,
                json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            )
        except httpx.HTTPError as exc:
            logger.warning(f"unload {model} failed: {exc}")

    async def _ensure_only_resident(
        self, client: httpx.AsyncClient, target: str
    ) -> None:
        """Before loading `target`: unload every other resident model and poll
        /api/ps until only `target` (or nothing) remains. 15 s timeout."""
        import time as _time

        deadline = _time.monotonic() + 15.0
        while True:
            resident = [m.get("name", "") for m in await self._ps(client)]
            others = [m for m in resident if m != target]
            if not others:
                if len(resident) > 1:
                    raise VramGuardError(f"two models resident: {resident} (expected at most one)")
                self._resident_model = resident[0] if resident else None
                return
            for m in others:
                t0 = _time.monotonic()
                logger.info(f"[VRAM guard] unloading {m} to make room for {target}")
                await self._unload(client, m)
                logger.info(f"[VRAM guard] swap {m} -> {target} took {_time.monotonic()-t0:.2f}s")
            if _time.monotonic() > deadline:
                raise VramGuardError(f"could not unload resident models {others} within 15s")
            await asyncio.sleep(0.25)

    async def generate(
        self,
        role: str,
        prompt: str,
        *,
        system: str = "",
        temperature: float | None = None,
        num_predict: int | None = None,
        cancel_event: Any | None = None,
        keep_warm: bool = True,
        _format_schema: dict[str, Any] | None = None,
    ) -> str:
        """Role-based generation under the single lock, with the VRAM guard.

        ``keep_warm`` (default True) leaves the model resident after the call so
        consecutive same-model phases (route+plan, all-code, ...) skip the
        reload; a different model triggers a logged swap in _ensure_only_resident.
        """
        rc = self.config.get_role_config(role)
        model = str(rc["model"])
        resolved = await self.supervisor.resolve_model_name(model)
        temp = rc["temperature"] if temperature is None else temperature
        rungs = [r for r in CTX_RUNGS if r <= int(rc["num_ctx"])]
        if not rungs:
            rungs = [CTX_RUNGS[0]]
        ctx = rungs[-1]  # start at the role's configured rung

        async with self._lock:  # THE lock: load + generate are atomic
            last_exc: Exception | None = None
            for attempt_ctx in reversed(rungs):
                ctx = attempt_ctx
                try:
                    return await self._generate_guarded(
                        resolved, prompt, system=system, num_ctx=ctx,
                        num_predict=num_predict or int(rc["num_predict"]),
                        temperature=temp, cancel_event=cancel_event,
                        keep_warm=keep_warm, format_schema=_format_schema,
                    )
                except (VramGuardError,) as exc:
                    last_exc = exc
                    logger.warning(f"[VRAM guard] {exc}; stepping num_ctx down to next rung")
                except OllamaError as exc:
                    # OOM / runner stopped / HTTP 500 -> same step-down retry
                    if "memory" in str(exc).lower() or "runner stopped" in str(exc).lower() or "500" in str(exc):
                        last_exc = exc
                        logger.warning(f"[VRAM guard] {exc}; stepping num_ctx down to next rung")
                    else:
                        raise
            raise VramGuardError(
                f"{resolved} never reached GPU fraction >= {self.min_gpu_fraction} "
                f"even at num_ctx={CTX_RUNGS[0]}: {last_exc}"
            )

    async def structured(
        self,
        role: str,
        prompt: str,
        schema: Any,
        *,
        cancel_event: Any | None = None,
    ) -> Any:
        """Schema-constrained generation through the single lock + VRAM guard.
        One self-correct retry; raises on failure so callers fall back to the
        JSON-regex prompt path."""
        import json as _json

        last_error: str | None = None
        for _attempt in range(2):
            effective = prompt
            if last_error:
                effective = (
                    f"{prompt}\n\nYour previous reply failed validation with:\n"
                    f"{last_error}\nReturn ONLY a JSON object matching the schema."
                )
            text = await self.generate(
                role, effective,
                system="You output only strict JSON matching the schema.",
                cancel_event=cancel_event,
                _format_schema=schema.model_json_schema(),
            )
            try:
                return schema.model_validate(_json.loads(text))
            except Exception as exc:
                last_error = str(exc)[:400]
        raise OllamaError(f"structured output failed validation twice: {last_error}")

    async def _generate_guarded(
        self,
        resolved: str,
        prompt: str,
        *,
        system: str,
        num_ctx: int,
        num_predict: int,
        temperature: float,
        cancel_event: Any | None,
        keep_warm: bool,
        format_schema: dict[str, Any] | None = None,
    ) -> str:
        async with httpx.AsyncClient(timeout=self.supervisor.timeout) as client:
            # 1) only `resolved` may be resident before we load
            await self._ensure_only_resident(client, resolved)
            if self.watchdog:
                _log_vram("before load")
            # 2) generate (load+run; we already hold the single lock, so use
            #    the lock-free supervisor path to avoid re-entrant deadlock)
            try:
                text = await self.supervisor.generate_unlocked(
                    prompt=prompt,
                    system_prompt=system,
                    model_name=resolved,
                    config=self._ctx_config(num_ctx),
                    num_predict=num_predict,
                    temperature=temperature,
                    cancel_event=cancel_event,
                    format_schema=format_schema,
                )
            except OllamaCancelledError:
                # Cancel: unload whatever is resident so nothing lingers on GPU.
                await self._unload(client, resolved)
                self._resident_model = None
                raise
            # 3) verify GPU residency
            resident = await self._ps(client)
            entry = next((m for m in resident if m.get("name") == resolved), None)
            if entry is None:
                raise VramGuardError(f"{resolved} not resident after generate")
            size = int(entry.get("size", 0) or 0)
            size_vram = int(entry.get("size_vram", 0) or 0)
            fraction = (size_vram / size) if size else 0.0
            if self.watchdog:
                _log_vram("after load")
            if fraction < self.min_gpu_fraction:
                await self._unload(client, resolved)
                self._resident_model = None
                raise VramGuardError(
                    f"{resolved} GPU fraction {fraction:.2f} < {self.min_gpu_fraction} "
                    f"(size_vram={size_vram}, size={size}) at num_ctx={num_ctx}"
                )
            if not keep_warm:
                await self._unload(client, resolved)
                self._resident_model = None
            else:
                self._resident_model = resolved
            return text

    def _ctx_config(self, num_ctx: int) -> AgentConfig:
        """Config copy with num_ctx overridden for this rung."""
        import copy as _copy

        cfg = _copy.deepcopy(self.config)
        for role_cfg in cfg.roles.values():
            role_cfg["num_ctx"] = num_ctx
        return cfg


class OllamaEngine:
    """Synchronous Ollama engine adapter used by the CLI orchestrator pipeline."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_HOST,
        config: AgentConfig | None = None,
        workspace_root: Path | str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
        self.config = config or load_agent_config(self.workspace_root)
        self._sync_lock = threading.Lock()

    @property
    def active_model(self) -> str | None:
        return self.config.coder_model

    def _resolve_model_name(self, requested: str) -> str:
        """Exact-match against /api/tags; never fuzzy-substitute.

        Falls back to the requested name only when the tags endpoint is
        unreachable (the generate call then surfaces any real error).
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(self.tags_url)
                if res.status_code == 200:
                    installed = [m.get("name", "") for m in res.json().get("models", [])]
                    if requested in installed:
                        return requested
                    if ":" not in requested and f"{requested}:latest" in installed:
                        return f"{requested}:latest"
                    raise ModelNotInstalled(requested, installed)
        except ModelNotInstalled:
            raise
        except Exception:
            pass
        return requested

    def chat_response(
        self,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = TEMPERATURE_CODE,
        seed: int = DEFAULT_SEED,
    ) -> LLMResponse:
        """Synchronous generation with think-stripping and truncation guard.

        A response cut off at the ``num_predict`` ceiling is retried ONCE with a
        doubled budget (bounded by remaining context); a response cut off by the
        context window itself raises :class:`OllamaTruncatedError` so the caller
        can retry with a smaller prompt instead of judging a partial answer.
        """
        with self._sync_lock:
            resolved = self._resolve_model_name(model)
            num_predict = max_tokens if max_tokens else _default_num_predict(resolved)

            for attempt in range(2):
                payload: dict[str, Any] = {
                    "model": resolved,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "keep_alive": self.config.keep_alive,
                    "options": {
                        "num_ctx": self.config.max_context_tokens,
                        "num_predict": num_predict,
                        "temperature": temperature,
                        "seed": seed,
                    },
                }
                with httpx.Client(timeout=600.0) as client:
                    resp = client.post(self.generate_url, json=payload)
                    if resp.status_code != 200:
                        raise OllamaError(
                            f"Ollama generation failed ({resp.status_code}): {resp.text}"
                        )
                    data = resp.json()

                raw_text = data.get("response", "")
                done_reason = data.get("done_reason") or "stop"
                prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
                output_tokens = int(data.get("eval_count", 0) or 0)
                seconds = round(float(data.get("total_duration", 0) or 0) / 1e9, 3)

                if done_reason == "length":
                    ctx_headroom = self.config.max_context_tokens - prompt_tokens
                    bigger = min(num_predict * 2, MAX_NUM_PREDICT, max(ctx_headroom, 0))
                    if output_tokens >= num_predict and bigger > num_predict and attempt == 0:
                        logger.warning(
                            f"Response from {resolved} hit num_predict={num_predict}; "
                            f"retrying once with num_predict={bigger}"
                        )
                        num_predict = bigger
                        continue
                    raise OllamaTruncatedError(
                        f"response from {resolved} was cut off at the context window "
                        f"(prompt={prompt_tokens} tok, output={output_tokens} tok, "
                        f"num_ctx={self.config.max_context_tokens}); retry with a smaller prompt"
                    )

                reasoning, answer = strip_think_blocks(raw_text)
                return LLMResponse(
                    text=answer,
                    reasoning=reasoning,
                    model=resolved,
                    done_reason=done_reason,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    seconds=seconds,
                )

        raise OllamaTruncatedError(  # unreachable, but keeps type-checkers honest
            f"response from {resolved} was cut off repeatedly"
        )

    def chat(
        self,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = TEMPERATURE_CODE,
    ) -> str:
        """Compatibility wrapper: return only the think-stripped answer text."""
        return self.chat_response(
            model, prompt, system=system, max_tokens=max_tokens, temperature=temperature
        ).text
