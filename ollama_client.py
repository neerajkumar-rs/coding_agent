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


class OllamaCancelledError(OllamaError):
    """The in-flight request was aborted because the run was cancelled."""


class OllamaTruncatedError(OllamaError):
    """Response was cut off (``done_reason == "length"``). Always retryable."""

    def __init__(self, message: str = "response truncated (done_reason=length)") -> None:
        super().__init__(message, retryable=True, done_reason="length")


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"
        self.timeout = timeout
        self._vram_mutex = asyncio.Lock()
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
        """Resolve requested model name against locally pulled Ollama models."""
        installed = await self.list_installed_models()
        if not installed:
            return requested

        # Direct match
        for m in installed:
            if m == requested or m.startswith(f"{requested}:"):
                return m

        # Fuzzy match for deepseek 1.5b
        req_lower = requested.lower()
        if "deepseek" in req_lower and "1.5b" in req_lower:
            for m in installed:
                m_lower = m.lower()
                if "deepseek" in m_lower and "1.5b" in m_lower:
                    logger.info(f"Resolved '{requested}' to locally installed '{m}'")
                    return m

        # Fuzzy match for qwen coder
        if "coder" in req_lower:
            size = "3b" if "3b" in req_lower else ("1.5b" if "1.5b" in req_lower else "")
            for m in installed:
                m_lower = m.lower()
                if "coder" in m_lower and (not size or size in m_lower):
                    logger.info(f"Resolved '{requested}' to locally installed '{m}'")
                    return m

        return requested

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
                logger.info(
                    f"[Supervisor Lock Acquired] Running {resolved} "
                    f"(num_ctx={cfg.max_context_tokens}, num_predict={effective_predict}, "
                    f"temp={temperature}, keep_alive={cfg.keep_alive})"
                )
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    data = await self._post_cancellable(client, payload, cancel_event)

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
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(self.tags_url)
                if res.status_code == 200:
                    installed = [m.get("name", "") for m in res.json().get("models", [])]
                    for m in installed:
                        if m == requested or m.startswith(f"{requested}:"):
                            return m
                    req_lower = requested.lower()
                    for m in installed:
                        m_lower = m.lower()
                        if "deepseek" in req_lower and "1.5b" in req_lower and "deepseek" in m_lower and "1.5b" in m_lower:
                            return m
                        if "coder" in req_lower and "coder" in m_lower:
                            return m
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
