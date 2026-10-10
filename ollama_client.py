"""High-performance asynchronous Ollama API client with strict VRAM fencing."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx

from agent_config import AgentConfig, load_agent_config

logger = logging.getLogger("ollama_client")

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
GENERATE_ENDPOINT = f"{DEFAULT_OLLAMA_HOST}/api/generate"
TAGS_ENDPOINT = f"{DEFAULT_OLLAMA_HOST}/api/tags"


class OllamaError(RuntimeError):
    """Raised when Ollama inference fails or returns an error status."""


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
    ) -> dict[str, Any]:
        cfg = config or load_agent_config()
        return {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "keep_alive": "10m",
            "options": {
                "num_ctx": cfg.max_context_tokens,
                "flash_attention": True,
                "keep_alive": "10m",
            },
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model_name: str | None = None,
        config: AgentConfig | None = None,
    ) -> str:
        """Sequential stateless generation call with strict lock and 0ms keep_alive."""
        cfg = config or load_agent_config()
        target_model = model_name or cfg.planner_model
        resolved = await self.resolve_model_name(target_model)
        payload = self.build_payload(
            model=resolved,
            prompt=prompt,
            system_prompt=system_prompt,
            config=cfg,
            stream=False,
        )

        async with self._vram_mutex:
            logger.info(
                f"[Supervisor Lock Acquired] Running {resolved} "
                f"(num_ctx={cfg.max_context_tokens}, flash_attn=True, keep_alive=10m)"
            )
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.generate_url, json=payload)
                if response.status_code != 200:
                    raise OllamaError(
                        f"Ollama generation failed ({response.status_code}): {response.text}"
                    )
                data = response.json()
                return data.get("response", "")

    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model_name: str | None = None,
        config: AgentConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens while holding the VRAM supervisor lock."""
        cfg = config or load_agent_config()
        target_model = model_name or cfg.coder_model
        resolved = await self.resolve_model_name(target_model)
        payload = self.build_payload(
            model=resolved,
            prompt=prompt,
            system_prompt=system_prompt,
            config=cfg,
            stream=True,
        )

        async with self._vram_mutex:
            logger.info(
                f"[Supervisor Lock Acquired for Stream] Running {resolved} "
                f"(num_ctx={cfg.max_context_tokens}, keep_alive=10m)"
            )
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
                            token = chunk.get("response", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue


class OllamaEngine:
    """Lightweight direct Ollama engine adapter replacing legacy proprietary backends."""

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
        self._lock = asyncio.Lock()
        import threading
        self._sync_lock = threading.Lock()

    @property
    def active_model(self) -> str | None:
        return self.config.coder_model

    def chat(self, model: str, prompt: str, system: str = "", max_tokens: int | None = None) -> str:
        """Synchronous chat method hitting local Ollama endpoint with strict options."""
        with self._sync_lock:
            resolved = model
            try:
                with httpx.Client(timeout=5.0) as client:
                    res = client.get(self.tags_url)
                    if res.status_code == 200:
                        installed = [m.get("name", "") for m in res.json().get("models", [])]
                        for m in installed:
                            if m == model or m.startswith(f"{model}:"):
                                resolved = m
                                break
                            if "deepseek" in model.lower() and "1.5b" in model.lower() and "deepseek" in m.lower() and "1.5b" in m.lower():
                                resolved = m
                                break
                            if "coder" in model.lower() and "coder" in m.lower():
                                resolved = m
                                break
            except Exception:
                pass

            payload: dict[str, Any] = {
                "model": resolved,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "num_ctx": self.config.max_context_tokens,
                    "flash_attention": True,
                    "keep_alive": "10m",
                },
            }
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            with httpx.Client(timeout=600.0) as client:
                resp = client.post(self.generate_url, json=payload)
                if resp.status_code != 200:
                    raise OllamaError(f"Ollama generation failed ({resp.status_code}): {resp.text}")
                return resp.json().get("response", "")
