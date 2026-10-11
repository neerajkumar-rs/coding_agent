"""FastAPI Master Controller for Local LLM Orchestration & Agentic Workflows.

Deterministic supervisor (0MB VRAM footprint) managing local Ollama endpoints,
deferred staging isolation, cross-platform sandbox diagnostics, and difflib reviews.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent_config import (
    AgentConfig,
    benchmark_and_calculate_tokens,
    load_agent_config,
    parse_and_recalculate,
    save_agent_config,
)
from ollama_client import (
    TEMPERATURE_CODE,
    TEMPERATURE_PLAN,
    OllamaError,
    OllamaSupervisorClient,
    OllamaTruncatedError,
    strip_think_blocks,
)
from sandbox import ExecutionSandbox
from scanner import WorkspaceScanner
from workspace_staging import WorkspaceStagingManager

# Set up controller logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("master_controller")

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".txt": "text",
}

app = FastAPI(
    title="Local LLM Master Controller",
    description="High-performance deterministic supervisor for local Ollama token-efficient agentic workflows.",
    version="2.0.0",
)

# Shared in-memory controller state
class ControllerState:
    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.config: AgentConfig = load_agent_config(self.workspace_root)
        self.staging = WorkspaceStagingManager(self.workspace_root)
        self.sandbox = ExecutionSandbox(self.workspace_root, self.staging)
        self.scanner = WorkspaceScanner(self.workspace_root)
        self.ollama = OllamaSupervisorClient()

    def rebind_workspace(self, root_path: str | Path) -> None:
        self.workspace_root = Path(root_path).resolve()
        self.config = load_agent_config(self.workspace_root)
        self.staging = WorkspaceStagingManager(self.workspace_root)
        self.sandbox = ExecutionSandbox(self.workspace_root, self.staging)
        self.scanner = WorkspaceScanner(self.workspace_root)
        logger.info(f"Workspace rebound to: {self.workspace_root}")

    def ensure_workspace(self, root_path: str | Path | None) -> None:
        """Rebind ONLY when the resolved path actually differs — sending the
        same workspace_dir on consecutive calls must NOT wipe the task queue."""
        if root_path is None:
            return
        resolved = Path(str(root_path).strip().strip('"')).resolve()
        if resolved != self.workspace_root:
            self.rebind_workspace(resolved)


state = ControllerState()


# -----------------------------------------------------------------------------
# Pydantic Request / Response Schemas
# -----------------------------------------------------------------------------

class InitRequest(BaseModel):
    root_path: str = Field(default=".", description="Target workspace root path")
    max_context_tokens: int | None = Field(default=None, description="Optional manual token limit override")
    coder_model: str | None = Field(default=None, description="Coder model override")
    planner_model: str | None = Field(default=None, description="Planner model override")


class StatusRequest(BaseModel):
    workspace_dir: str | None = Field(default=None, description="Optional workspace override")


class DeployRequest(BaseModel):
    workspace_dir: str | None = Field(default=None, description="Optional workspace override")


class RunRequestPayload(BaseModel):
    prompt: str = Field(min_length=1, description="Full user request for an autonomous run")
    workspace_dir: str | None = Field(default=None, description="Optional workspace override")
    auto_deploy: bool = Field(default=True, description="Deploy staged files on success")


class CancelRequest(BaseModel):
    workspace_dir: str | None = Field(default=None, description="Workspace whose run to cancel")


# -----------------------------------------------------------------------------
# Core Controller Endpoints
# -----------------------------------------------------------------------------

def _build_id() -> str:
    """Hash of main.py's mtime+size — changes whenever the controller file
    changes, so clients can detect a stale (pre-fix) server process."""
    try:
        st = Path(__file__).stat()
        raw = f"{Path(__file__).name}:{st.st_mtime_ns}:{st.st_size}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
    except OSError:
        return "unknown"


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Controller health check and VRAM supervisor status."""
    installed = await state.ollama.list_installed_models()
    return {
        "status": "online",
        "build_id": _build_id(),
        "supervisor_vram": "0MB (Pure Python host)",
        "active_workspace": str(state.workspace_root),
        "agent_profile": state.config.to_dict(),
        "ollama_connected": len(installed) > 0,
        "available_models": installed,
    }


def _base_model_warning(name: str, quant: str) -> str | None:
    """Heuristic: a name with no instruct/chat/code suffix, or a base quant,
    is likely a base (non-instruction-tuned) model — warn for the thinker."""
    lowered = name.lower()
    if any(tag in lowered for tag in ("instruct", "chat", "coder", "-it", "r1")):
        return None
    return f"'{name}' looks like a BASE model (no instruct/chat/coder tag); it may follow instructions poorly"


@app.get("/models")
async def list_models() -> dict[str, Any]:
    """Per-role model card: model, digest (12 chars), size, quant.

    Reads /api/tags (digest, size) and /api/show (quantization) for each
    configured role, plus which model is currently resident (from /api/ps).
    """
    tags: dict[str, dict[str, Any]] = {}
    resident: list[str] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(state.ollama.tags_url)
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    tags[m.get("name", "")] = m
        except httpx.HTTPError:
            pass
        try:
            ps = await client.get(f"{state.ollama.base_url}/api/ps")
            if ps.status_code == 200:
                resident = [m.get("name", "") for m in ps.json().get("models", [])]
        except httpx.HTTPError:
            pass
        # /api/show per configured role for the quantization level
        quants: dict[str, str] = {}
        for role in ("thinker", "coder", "validator"):
            name = str(state.config.get_role_config(role)["model"])
            if name in quants:
                continue
            try:
                show = await client.post(
                    f"{state.ollama.base_url}/api/show", json={"name": name}
                )
                if show.status_code == 200:
                    details = show.json().get("details", {})
                    quants[name] = details.get("quantization_level", "")
            except httpx.HTTPError:
                pass

    roles: dict[str, Any] = {}
    for role in ("thinker", "coder", "validator"):
        cfg = state.config.get_role_config(role)
        name = str(cfg["model"])
        tag = tags.get(name, {})
        raw_digest = str(tag.get("digest", ""))
        digest = raw_digest.removeprefix("sha256:")[:12]
        size_gb = round(int(tag.get("size", 0)) / (1024**3), 2) if tag.get("size") else None
        quant = quants.get(name, "")
        roles[role] = {
            "model": name,
            "digest": digest,
            "size_gb": size_gb,
            "quant": quant,
            "num_ctx": cfg.get("num_ctx"),
            "installed": name in tags,
            "resident": name in resident,
            "warning": _base_model_warning(name, quant),
        }
    return {"workspace": str(state.workspace_root), "roles": roles, "resident": resident}


@app.post("/workspace/init")
async def workspace_init(payload: InitRequest) -> dict[str, Any]:
    """1. POST /workspace/init
    
    Accepts root path. Initializes or parses .agent configuration, builds
    staging (.agent_staging/), and creates logging structures (.agent_logs/).
    """
    target_root = Path(payload.root_path).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    state.rebind_workspace(target_root)

    # Check for overrides. Copy-on-write: never rebuild AgentConfig from
    # scratch (that drops roles/keep_alive/allowed_commands/extra).
    if payload.coder_model or payload.planner_model or payload.max_context_tokens:
        import copy as _copy

        updated = _copy.deepcopy(state.config)
        if payload.planner_model:
            updated.roles["thinker"]["model"] = payload.planner_model
            updated.roles["validator"]["model"] = payload.planner_model
        if payload.coder_model:
            updated.roles["coder"]["model"] = payload.coder_model
        if payload.max_context_tokens or payload.coder_model or payload.planner_model:
            tokens = benchmark_and_calculate_tokens(
                coder_model=payload.coder_model or state.config.coder_model,
                manual_context=payload.max_context_tokens,
            )
            for role, cfg in updated.roles.items():
                cfg["num_ctx"] = min(tokens, 16384)
        state.config = updated
        save_agent_config(state.config, state.workspace_root)

    # Scan project structure for awareness
    awareness = state.scanner.build_system_awareness()

    logger.info(f"Initialized workspace at {state.workspace_root}")
    return {
        "status": "initialized",
        "workspace_root": str(state.workspace_root),
        "agent_config": state.config.to_dict(),
        "staging_dir": str(state.staging.staging_dir),
        "logs_dir": str(state.staging.logs_dir),
        "system_awareness": awareness["summary"],
    }


@app.post("/agent/run")
async def agent_run(payload: RunRequestPayload) -> dict[str, Any]:
    """THE request pipeline: ROUTE -> PLAN -> CODE -> CHECK -> VALIDATE -> SUMMARY.

    Returns {summary, staged_files, failed_tasks, trace} for code tasks or
    {answer, trace} for questions. Never auto-deploys: the TUI shows diffs
    and asks before deploying (only files that passed CHECK are staged).
    """
    from orchestrator import run_pipeline

    from agent_core.engine import get_engine

    state.ensure_workspace(payload.workspace_dir)
    # One cancel event per workspace: /agent/cancel sets it on this same engine.
    cancel_event = get_engine(state.workspace_root)._cancel_event
    cancel_event.clear()  # a fresh run starts un-cancelled

    async def _generate(
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        role: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        return await state.ollama.generate(
            prompt=prompt, system_prompt=system, model_name=model,
            config=state.config, temperature=temperature,
            cancel_event=cancel_event,
        )

    async def _unload(model: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{state.ollama.base_url}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
                )
        except Exception as exc:
            logger.warning(f"model unload failed for {model}: {exc}")

    async def _structured_generate(model: str, prompt: str, schema: Any) -> Any:
        """Schema-constrained generation: Ollama `format` + pydantic validation,
        one self-correct retry. Aborts promptly on cancel. Raises on failure so
        the caller falls back to the JSON-regex prompt path."""
        last_error: str | None = None
        for _attempt in range(2):
            effective = prompt
            if last_error:
                effective = (
                    f"{prompt}\n\nYour previous reply failed validation with:\n"
                    f"{last_error}\nReturn ONLY a JSON object matching the schema."
                )
            payload = {
                "model": model,
                "prompt": effective,
                "system": "You output only strict JSON matching the schema.",
                "stream": False,
                "keep_alive": state.config.keep_alive,
                "format": schema.model_json_schema(),
                "options": {
                    "num_ctx": state.config.max_context_tokens,
                    "num_predict": 1024,
                    "temperature": TEMPERATURE_PLAN,
                },
            }
            if cancel_event.is_set():
                raise RuntimeError("run cancelled")
            async with httpx.AsyncClient(timeout=state.ollama.timeout) as client:
                resp = await client.post(state.ollama.generate_url, json=payload)
                if resp.status_code != 200:
                    raise OllamaError(f"structured generate failed ({resp.status_code})")
                raw = resp.json().get("response", "")
            _, clean = strip_think_blocks(raw)
            try:
                return schema.model_validate(json.loads(clean))
            except Exception as exc:
                last_error = str(exc)[:400]
        raise OllamaError(f"structured output failed validation twice: {last_error}")

    try:
        result = await run_pipeline(
            payload.prompt,
            state.workspace_root,
            generate=_generate,
            planner_model=state.config.planner_model,
            coder_model=state.config.coder_model,
            unload=_unload,
            cancel_event=cancel_event,
            num_ctx=state.config.max_context_tokens,
            structured_generate=_structured_generate,
        )
        if result.get("status") == "cancelled":
            # Cancel is a normal outcome, not a 500: unload whatever is resident.
            await _unload(state.config.coder_model)
            await _unload(state.config.planner_model)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"agent run failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent/cancel")
async def agent_cancel(payload: CancelRequest | None = None) -> dict[str, Any]:
    """Request cancellation of the run in progress for a workspace."""
    from agent_core.engine import get_engine

    workspace = payload.workspace_dir if payload and payload.workspace_dir else state.workspace_root
    engine = get_engine(workspace)
    engine.cancel()
    return {"status": "cancel_requested", "workspace": str(workspace)}


@app.get("/agent/events")
async def agent_events(workspace_dir: str | None = None, run_id: str | None = None, follow: bool = False):
    """Server-sent events stream of a run's events.jsonl.

    ``follow=true`` tails the file (for live TUI updates); otherwise the
    current contents are returned and the stream closes.
    """
    import asyncio as _asyncio

    from fastapi.responses import StreamingResponse

    from agent_core.run_state import RUNS_DIR_NAME

    workspace = Path(workspace_dir).resolve() if workspace_dir else state.workspace_root
    runs_dir = workspace / RUNS_DIR_NAME
    if not runs_dir.is_dir():
        raise HTTPException(status_code=404, detail="No runs yet for this workspace")

    if run_id:
        run_dir = runs_dir / run_id
    else:
        candidates = sorted((d for d in runs_dir.iterdir() if d.is_dir()), reverse=True)
        if not candidates:
            raise HTTPException(status_code=404, detail="No runs yet for this workspace")
        run_dir = candidates[0]
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        raise HTTPException(status_code=404, detail=f"No events for run {run_dir.name}")

    async def event_stream():
        sent = 0
        # Replay existing events
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield f"data: {line.strip()}\n\n"
                sent += 1
        if not follow:
            return
        # Tail for new events while the run is live
        idle = 0
        while idle < 300:  # ~5 minutes max tail
            lines = events_path.read_text(encoding="utf-8").splitlines()
            new = [ln for ln in lines[sent:] if ln.strip()]
            for line in new:
                yield f"data: {line.strip()}\n\n"
            sent += len(new)
            if not new:
                idle += 1
                yield ": keepalive\n\n"
            else:
                idle = 0
            await _asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/workspace/status")
async def workspace_status(payload: StatusRequest | None = None) -> dict[str, Any]:
    """4. POST /workspace/status
    
    Uses Python's native difflib to compare .agent_staging/ against the active
    workspace root. Returns a line-by-line git-style diff object ready for TUI/GUI change previews.
    """
    if payload:
        state.ensure_workspace(payload.workspace_dir)

    diff_data = state.staging.compute_staging_diff()
    return {
        "status": "ok",
        "workspace_root": str(state.workspace_root),
        "has_changes": diff_data["has_changes"],
        "staged_count": diff_data["staged_count"],
        "total_additions": diff_data["total_additions"],
        "total_deletions": diff_data["total_deletions"],
        "diffs": diff_data["diffs"],
    }


@app.post("/agent/deploy")
async def agent_deploy(payload: DeployRequest | None = None) -> dict[str, Any]:
    """5. POST /agent/deploy

    Triggers atomic file rearrangements from staging to production paths and
    flushes the staging cache. Overwritten files are backed up under
    .agent_runs/<latest_run>/backup/ for /agent/rollback.
    """
    if payload:
        state.ensure_workspace(payload.workspace_dir)

    from agent_core.run_state import RUNS_DIR_NAME

    runs_dir = state.workspace_root / RUNS_DIR_NAME
    backup_dir = None
    if runs_dir.is_dir():
        run_dirs = sorted((d for d in runs_dir.iterdir() if d.is_dir()), reverse=True)
        if run_dirs:
            backup_dir = run_dirs[0] / "backup"

    from agent_core.staging import StagingArea

    staging = StagingArea(state.workspace_root)
    # Reuse the controller's staging manager state: deploy whatever is staged.
    staged_files = staging.manager.list_staged_files()
    if not staged_files:
        return {"status": "noop", "message": "No staged files to deploy", "deployed_files": []}
    from agent_core.staging import safe_target

    for rel in staged_files:
        safe_target(state.workspace_root, rel)
    if backup_dir is not None:
        for rel in staged_files:
            existing = state.workspace_root / rel
            if existing.is_file():
                dest = backup_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(existing.read_bytes())
    result = state.staging.deploy_staged_files()
    result["backup_dir"] = str(backup_dir) if backup_dir else None
    return result


class RollbackRequest(BaseModel):
    workspace_dir: str | None = Field(default=None, description="Optional workspace override")
    run_id: str | None = Field(default=None, description="Run whose backup to restore (default: latest)")


@app.post("/agent/rollback")
async def agent_rollback(payload: RollbackRequest | None = None) -> dict[str, Any]:
    """Restore files backed up during a previous deploy.

    Picks the latest run's backup under .agent_runs/ unless run_id is given.
    """
    from agent_core.run_state import RUNS_DIR_NAME
    from agent_core.staging import StagingArea

    workspace = Path(payload.workspace_dir).resolve() if payload and payload.workspace_dir else state.workspace_root
    runs_dir = workspace / RUNS_DIR_NAME
    if not runs_dir.is_dir():
        raise HTTPException(status_code=404, detail="No .agent_runs directory found")

    run_dirs = sorted((d for d in runs_dir.iterdir() if d.is_dir()), reverse=True)
    chosen = None
    if payload and payload.run_id:
        chosen = runs_dir / payload.run_id
        if not chosen.is_dir():
            raise HTTPException(status_code=404, detail=f"run not found: {payload.run_id}")
    else:
        chosen = next((d for d in run_dirs if (d / "backup").is_dir()), None)
        if chosen is None:
            raise HTTPException(status_code=404, detail="No run backups available to restore")

    restored = StagingArea.rollback(workspace, chosen / "backup")
    return {
        "status": "restored" if restored else "noop",
        "run_id": chosen.name,
        "restored_files": restored,
    }


# -----------------------------------------------------------------------------
# CLI Entrypoint & Runner
# -----------------------------------------------------------------------------

def run_cli() -> None:
    """Parse sys.argv and handle --recalculate or start uvicorn server."""
    # Check if --recalculate flag was passed
    recalculated = parse_and_recalculate(sys.argv[1:])
    if "--recalculate" in sys.argv and "--serve" not in sys.argv:
        # Processed recalculate and exit
        sys.exit(0)

    # Otherwise run uvicorn server
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")  # localhost by default; set HOST=0.0.0.0 to expose
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting Local LLM Master Controller on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_cli()
