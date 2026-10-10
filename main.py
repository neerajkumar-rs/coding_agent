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
from typing import Any, Literal

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
        self.task_queue: list[dict[str, Any]] = []
        self.executed_task_ids: set[int] = set()
        self.last_planner_reasoning: str = ""

    def rebind_workspace(self, root_path: str | Path) -> None:
        self.workspace_root = Path(root_path).resolve()
        self.config = load_agent_config(self.workspace_root)
        self.staging = WorkspaceStagingManager(self.workspace_root)
        self.sandbox = ExecutionSandbox(self.workspace_root, self.staging)
        self.scanner = WorkspaceScanner(self.workspace_root)
        self.task_queue.clear()
        self.executed_task_ids.clear()
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


class TaskItem(BaseModel):
    id: int
    file: str
    action: Literal["create", "edit"]
    description: str


class PlanRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Feature or refactor instruction")
    workspace_dir: str | None = Field(default=None, description="Optional workspace override")


class PlanResponse(BaseModel):
    tasks: list[TaskItem]
    raw_reasoning_length: int
    model_used: str


class StepRequest(BaseModel):
    task_id: int | None = Field(default=None, description="Specific task ID to execute (or next pending)")
    task: TaskItem | None = Field(default=None, description="Direct task specification override")
    workspace_dir: str | None = Field(default=None, description="Optional workspace override")
    custom_command: str | None = Field(default=None, description="Optional custom test/diagnostic script")


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
# Parsing Helpers
# -----------------------------------------------------------------------------

def extract_think_and_content(raw_text: str) -> tuple[str, str]:
    """Separate <think>...</think> chain-of-thought blocks from final payload.

    Delegates to the LLM layer's stripper, which also handles unterminated
    trailing <think> blocks. Returns (reasoning, answer).
    """
    return strip_think_blocks(raw_text)


SOURCE_SUFFIXES = {".py", ".html", ".css", ".js", ".json", ".md"}


def parse_tasks_json(raw_text: str) -> list[dict[str, Any]]:
    """Extract and strictly validate task queue matching the required schema.

    Tasks whose target is not a source file type (tasks.md, summary.txt,
    taskN_code.txt, images, etc.) are rejected — the planner never plans
    around leftover junk artifacts.
    """
    _, content = extract_think_and_content(raw_text)

    # Search for json block in markdown
    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidate_json = json_block.group(1) if json_block else content

    def _valid_target(file_path: str) -> bool:
        from scanner import is_junk_file

        if is_junk_file(Path(file_path).name):
            return False
        return Path(file_path).suffix.lower() in SOURCE_SUFFIXES

    try:
        data = json.loads(candidate_json)
        if isinstance(data, dict) and "tasks" in data and isinstance(data["tasks"], list):
            tasks = []
            for item in data["tasks"]:
                task_id = int(item.get("id", len(tasks) + 1))
                file_path = str(item.get("file", "app.py")).strip()
                action = str(item.get("action", "edit")).lower()
                if action not in ("create", "edit"):
                    action = "edit"
                desc = str(item.get("description", "")).strip()
                if not _valid_target(file_path):
                    logger.warning(f"Rejecting plan task with non-source target: {file_path}")
                    continue
                tasks.append({"id": task_id, "file": file_path, "action": action, "description": desc})
            if tasks:
                return tasks
    except Exception:
        pass

    # Regex fallback search for JSON-like task objects
    matches = re.findall(
        r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"file"\s*:\s*"([^"]+)"\s*,\s*"action"\s*:\s*"(create|edit)"\s*,\s*"description"\s*:\s*"([^"]+)"\s*\}',
        content,
    )
    if matches:
        return [
            {"id": int(m[0]), "file": m[1], "action": m[2], "description": m[3]}
            for m in matches
        ]

    # Minimal fallback
    logger.warning("Could not parse explicit tasks array, creating default single task")
    return [{"id": 1, "file": "app.py", "action": "edit", "description": content[:200]}]


def extract_code_block(raw_text: str, target_file: str) -> str:
    """Extract code block generated by coder model."""
    # Look for markdown code fence
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    if matches:
        # Choose the longest code block
        longest = max(matches, key=len)
        return longest.rstrip() + "\n"

    # If no fences, return stripped raw text
    return raw_text.strip() + "\n"


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


@app.post("/workspace/init")
async def workspace_init(payload: InitRequest) -> dict[str, Any]:
    """1. POST /workspace/init
    
    Accepts root path. Initializes or parses .agent configuration, builds
    staging (.agent_staging/), and creates logging structures (.agent_logs/).
    """
    target_root = Path(payload.root_path).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    state.rebind_workspace(target_root)

    # Check for overrides
    if payload.coder_model or payload.planner_model or payload.max_context_tokens:
        updated_tokens = benchmark_and_calculate_tokens(
            coder_model=payload.coder_model or state.config.coder_model,
            manual_context=payload.max_context_tokens,
        )
        state.config = AgentConfig(
            planner_model=payload.planner_model or state.config.planner_model,
            coder_5b_or_3b=payload.coder_model or state.config.coder_5b_or_3b,
            max_context_tokens=updated_tokens,
        )
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


@app.post("/agent/plan", response_model=PlanResponse)
async def agent_plan(payload: PlanRequest) -> PlanResponse:
    """2. POST /agent/plan
    
    Evaluates structure map, queries the Thinker model (deepseek-r1:1.5b), logs raw
    reasoning to planner.log, and outputs strict JSON task queue matching:
    {"tasks": [{"id": 1, "file": "app.py", "action": "create"|"edit", "description": "task snippet"}]}
    """
    state.ensure_workspace(payload.workspace_dir)

    awareness = state.scanner.build_system_awareness()

    planner_prompt = (
        f"{awareness['prompt_context']}\n\n"
        f"=== USER INSTRUCTION ===\n{payload.prompt}\n\n"
        f"=== STRICT SPECIFICATION ===\n"
        f"You are the Lead Architectural Planner and Auditor. Analyze the workspace and user request.\n"
        f"Output MUST contain a JSON task queue array matching EXACTLY this schema:\n"
        f'{{"tasks": [{{"id": 1, "file": "path/to/file.py", "action": "create"|"edit", "description": "detailed task instructions"}}]}}\n'
        f"Do not output any instructions to apply code yet. Only plan the atomic sequential tasks."
    )

    system_prompt = (
        "You are an expert software architect operating in a resource-constrained deterministic pipeline. "
        "Your output must provide clean reasoning followed by a strict JSON object with a 'tasks' list."
    )

    try:
        raw_output = await state.ollama.generate(
            prompt=planner_prompt,
            system_prompt=system_prompt,
            model_name=state.config.planner_model,
            config=state.config,
            temperature=TEMPERATURE_PLAN,
        )
    except OllamaTruncatedError as exc:
        logger.error(f"Planning generation truncated: {exc}")
        raise HTTPException(status_code=502, detail=f"Planner response truncated (retryable): {exc}")
    except Exception as exc:
        logger.error(f"Planning generation failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Planner model failed: {exc}")

    reasoning, clean_text = extract_think_and_content(raw_output)
    task_queue = parse_tasks_json(raw_output)

    # Save to discrete planner.log matrix
    state.staging.log_planner(
        reasoning=reasoning if reasoning else raw_output,
        task_queue=task_queue,
    )

    # Store task queue in memory state
    state.task_queue = list(task_queue)
    state.executed_task_ids.clear()
    state.last_planner_reasoning = reasoning

    task_items = [TaskItem(**item) for item in task_queue]
    return PlanResponse(
        tasks=task_items,
        raw_reasoning_length=len(reasoning),
        model_used=state.config.planner_model,
    )


@app.post("/agent/step")
async def agent_step(payload: StepRequest) -> dict[str, Any]:
    """3. POST /agent/step
    
    Pulls next item from task queue. Injects specific code context into Coder model via Ollama.
    Streams patch into .agent_staging/, runs OS-specific diagnostic script, and returns result string.
    """
    state.ensure_workspace(payload.workspace_dir)

    # Resolve target task
    task_to_run: dict[str, Any] | None = None
    if payload.task:
        task_to_run = payload.task.model_dump()
    elif payload.task_id is not None:
        for t in state.task_queue:
            if t["id"] == payload.task_id:
                task_to_run = t
                break
        if task_to_run is None:
            # Clear, specific error: is the queue empty or the id wrong?
            if not state.task_queue:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Task {payload.task_id} not found: the queue for "
                        f"{state.workspace_root} is EMPTY. Call /agent/plan first, "
                        "or send the full task object in the request."
                    ),
                )
            known_ids = [t["id"] for t in state.task_queue]
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Task {payload.task_id} not found in the queue "
                    f"(known ids: {known_ids}). It may already be executed or the "
                    "queue was replanned."
                ),
            )
    else:
        # Pull next unexecuted task
        for t in state.task_queue:
            if t["id"] not in state.executed_task_ids:
                task_to_run = t
                break

    if not task_to_run:
        raise HTTPException(
            status_code=400,
            detail="No pending tasks found in queue. Call /agent/plan or provide a task payload.",
        )

    target_file = task_to_run["file"]
    action = task_to_run["action"]
    desc = task_to_run["description"]

    # Read existing context if editing
    existing_code = ""
    active_target = state.workspace_root / target_file
    staged_target = state.staging.get_staged_file_path(target_file)

    if staged_target.exists():
        existing_code = staged_target.read_text(encoding="utf-8", errors="replace")
    elif active_target.exists():
        existing_code = active_target.read_text(encoding="utf-8", errors="replace")

    # Build coder prompt — language derived from the target's real extension.
    awareness = state.scanner.build_system_awareness()
    language = LANGUAGE_BY_SUFFIX.get(Path(target_file).suffix.lower(), "text")
    coder_prompt = (
        f"{awareness['prompt_context']}\n\n"
        f"=== CURRENT TASK ===\n"
        f"File: {target_file}\n"
        f"Action: {action.upper()}\n"
        f"Description: {desc}\n\n"
    )

    if existing_code:
        coder_prompt += f"=== EXISTING FILE CONTENTS ({target_file}) ===\n```{language}\n{existing_code}\n```\n\n"

    coder_prompt += (
        f"=== INSTRUCTIONS ===\n"
        f"Write complete, production-ready {language} implementation for {target_file}.\n"
        f"Wrap the code inside a single markdown code fence (```{language} ... ```).\n"
        f"Do not truncate or omit lines."
    )

    system_prompt = (
        "You are an expert autonomous software engineer. "
        f"Output ONLY the complete {language} source enclosed in markdown fences."
    )

    # Generate code via Coder model (streamed token-by-token internally)
    generated_chunks: list[str] = []
    try:
        async for chunk in state.ollama.stream_generate(
            prompt=coder_prompt,
            system_prompt=system_prompt,
            model_name=state.config.coder_model,
            config=state.config,
            temperature=TEMPERATURE_CODE,
        ):
            generated_chunks.append(chunk)
    except OllamaTruncatedError as exc:
        logger.error(f"Coder generation truncated: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Coder response truncated (retryable, use a smaller scope): {exc}",
        )
    except Exception as exc:
        logger.error(f"Coder generation failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Coder model failed: {exc}")

    raw_code = "".join(generated_chunks)
    _, clean_code = strip_think_blocks(raw_code)
    extracted_code = extract_code_block(clean_code, target_file)

    # Deferred Staging: write exclusively into .agent_staging/
    staged_path = state.staging.stage_file(target_file, extracted_code)

    # Log to coder.log
    state.staging.log_coder(
        file_path=target_file,
        code_block=extracted_code,
        task_description=desc,
    )

    # OS-Aware Cross-Platform Sandbox Diagnostic
    diagnostic = await state.sandbox.execute_validation(
        target_rel_path=target_file,
        custom_command=payload.custom_command,
    )

    # Mark task executed
    state.executed_task_ids.add(task_to_run["id"])

    result_summary = (
        f"Task {task_to_run['id']} for {target_file} ({action}) completed. "
        f"Sandbox status: {'PASSED' if diagnostic['success'] else 'FAILED'} (Exit {diagnostic['exit_code']})."
    )

    return {
        "status": "completed",
        "task_id": task_to_run["id"],
        "file": target_file,
        "action": action,
        "staged_path": str(staged_path),
        "diagnostic": diagnostic,
        "result": result_summary,
    }


@app.post("/agent/run")
async def agent_run(payload: RunRequestPayload) -> dict[str, Any]:
    """Autonomous one-shot run through the single agent_core pipeline.

    Everything (planning, coding, staging, verification, audit, deploy) is
    executed by agent_core.engine.Engine in a worker thread; the process-wide
    LLM lock serializes GPU access across concurrent requests/workspaces.
    """
    from agent_core.engine import RunRequest, get_engine

    workspace = payload.workspace_dir or state.workspace_root
    engine = get_engine(workspace)

    def _run() -> dict[str, Any]:
        report = engine.run(
            RunRequest(
                prompt=payload.prompt,
                auto_deploy=payload.auto_deploy,
            )
        )
        return {
            "status": report.status,
            "run_id": report.run_id,
            "files": report.files,
            "deployed_files": report.deployed_files,
            "attempts": report.attempts,
            "llm_calls": report.llm_calls,
            "elapsed_seconds": report.elapsed_seconds,
            "errors": report.errors,
            "needs_human": report.needs_human,
            "audits": report.audits,
        }

    try:
        return await asyncio.to_thread(_run)
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
