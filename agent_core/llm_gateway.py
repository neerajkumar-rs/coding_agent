"""Single process-wide LLM gateway.

Constraints honoured here:
- ONE process-wide asyncio.Lock: only one generation touches the GPU at a
  time (RTX 3050 4GB cannot hold two models in flight).
- Structured outputs: Ollama's ``format`` parameter is fed a JSON schema and
  the reply is validated with pydantic. One validation retry, then fail.
- Think-stripping and truncation retry live in ``ollama_client``; this layer
  adds schema discipline and the global concurrency bound.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ollama_client import (
    DEFAULT_SEED,
    OllamaEngine,
    OllamaError,
    OllamaTruncatedError,
)

T = TypeVar("T", bound=BaseModel)

# One generation at a time, process-wide. Created lazily so importing this
# module never binds to a specific event loop.
_process_lock: asyncio.Lock | None = None


def get_process_lock() -> asyncio.Lock:
    global _process_lock
    if _process_lock is None:
        _process_lock = asyncio.Lock()
    return _process_lock


class StructuredOutputError(OllamaError):
    """Model output failed pydantic validation twice (initial + one retry)."""


class LegacyLLMAdapter:
    """Adapt any object exposing ``chat(model, prompt, ...) -> str`` to the
    gateway protocol (``.text`` / ``.structured``).

    Used so the Engine can run against test stubs and the old OllamaEngine
    chat surface. ``supports_structured`` is False: the planner skips the
    schema-constrained call and goes straight to markdown prompting, so a
    scripted stub's response sequence is consumed exactly as before.
    """

    supports_structured = False

    def __init__(self, chat_engine: Any) -> None:
        self._engine = chat_engine

    def text(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.1,
        seed: int = DEFAULT_SEED,
    ) -> str:
        return self._engine.chat(
            model, prompt, system=system, max_tokens=max_tokens, temperature=temperature
        )

    def structured(
        self,
        model: str,
        prompt: str,
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        raise StructuredOutputError(
            "legacy chat engine does not support schema-constrained decoding",
            retryable=False,
        )


class LLMGateway:
    """Wraps OllamaEngine with schema-validated calls and the global lock."""

    supports_structured = True

    def __init__(self, engine: OllamaEngine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # Plain text generation (sync, lock-guarded via the engine's own lock)
    # ------------------------------------------------------------------

    def text(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.1,
        seed: int = DEFAULT_SEED,
    ) -> str:
        """Think-stripped, truncation-guarded plain text generation."""
        return self.engine.chat_response(
            model, prompt, system=system, max_tokens=max_tokens,
            temperature=temperature, seed=seed,
        ).text

    # ------------------------------------------------------------------
    # Structured generation: JSON schema + pydantic validation + one retry
    # ------------------------------------------------------------------

    def structured(
        self,
        model: str,
        prompt: str,
        schema: type[T],
        *,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float = 0.1,
        seed: int = DEFAULT_SEED,
    ) -> T:
        """Generate a pydantic-validated structured response.

        Uses Ollama's ``format=<json schema>`` to constrain decoding, then
        validates with pydantic. On validation failure, retries ONCE with the
        validation error appended so the model can self-correct.
        """
        format_schema = schema.model_json_schema()
        last_error: str | None = None

        for attempt in range(2):
            effective_prompt = prompt
            if last_error:
                effective_prompt = (
                    f"{prompt}\n\nYour previous reply failed validation with:\n"
                    f"{last_error}\nReturn ONLY a JSON object matching the schema."
                )
            raw = self._generate_with_format(
                model, effective_prompt, format_schema,
                system=system, max_tokens=max_tokens,
                temperature=temperature, seed=seed,
            )
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON: {exc}; excerpt: {raw[:200]!r}"
                continue
            try:
                return schema.model_validate(data)
            except ValidationError as exc:
                # Compact error list; full repr is too long for small contexts.
                last_error = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                    for e in exc.errors()[:5]
                )
                continue

        raise StructuredOutputError(
            f"{model} could not produce a valid {schema.__name__} "
            f"after one retry: {last_error}",
            retryable=False,
        )

    def _generate_with_format(
        self,
        model: str,
        prompt: str,
        format_schema: dict[str, Any],
        *,
        system: str,
        max_tokens: int | None,
        temperature: float,
        seed: int,
    ) -> str:
        """Raw /api/generate call carrying a ``format`` JSON schema.

        Reuses the engine's model resolution, keep_alive, num_ctx and sync
        lock so this stays the single concurrency path.
        """
        import httpx

        from ollama_client import _default_num_predict, strip_think_blocks

        with self.engine._sync_lock:
            resolved = self.engine._resolve_model_name(model)
            num_predict = max_tokens if max_tokens else _default_num_predict(resolved)
            payload: dict[str, Any] = {
                "model": resolved,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "keep_alive": self.engine.config.keep_alive,
                "format": format_schema,
                "options": {
                    "num_ctx": self.engine.config.max_context_tokens,
                    "num_predict": num_predict,
                    "temperature": temperature,
                    "seed": seed,
                },
            }
            with httpx.Client(timeout=600.0) as client:
                resp = client.post(self.engine.generate_url, json=payload)
                if resp.status_code != 200:
                    raise OllamaError(
                        f"Ollama generation failed ({resp.status_code}): {resp.text}"
                    )
                data = resp.json()

            if (data.get("done_reason") or "stop") == "length":
                raise OllamaTruncatedError(
                    f"structured response from {resolved} was cut off "
                    f"(num_predict={num_predict}); retry with a smaller prompt"
                )
            _, answer = strip_think_blocks(data.get("response", ""))
            return answer
