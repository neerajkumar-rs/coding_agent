"""Unit tests for run_pipeline hardening: budgets, no-progress, no scaffolds,
run metadata under .agent_runs/. (Ported from the retired VibeOrchestrator
shim onto the single pipeline behind POST /agent/run.)"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import RUNS_DIR_NAME, RunBudgets, run_pipeline

PLANNER = "deepseek-r1:1.5b"
CODER = "qwen2.5-coder:3b"

PLAN = json.dumps([
    {"path": "generated.py", "action": "create",
     "description": "Python module with a greet function returning a greeting"}
])
GOOD_CODE = "```python\ndef greet(name: str) -> str:\n    return f'Hello, {name}!'\n```"
BROKEN_CODE = "```python\ndef broken(:\n```"
VALIDATE_OK = json.dumps({
    "satisfies_request": True, "problems": [], "summary": "A greet module."
})


def _fake(routes: dict[str, str]):
    import re

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        target = re.search(r"TARGET FILE:\s*(\S+)", prompt)
        if target and f"TARGET FILE: {target.group(1)}" in routes:
            return routes[f"TARGET FILE: {target.group(1)}"]
        for marker in sorted(routes, key=len, reverse=True):
            if marker in prompt:
                return routes[marker]
        return ""

    return fake


def _run(coro):
    return asyncio.run(coro)


def test_happy_path_and_metadata_location() -> None:
    fake = _fake({
        "JSON array of tasks": PLAN,
        "TARGET FILE: generated.py": GOOD_CODE,
        "satisfies_request": VALIDATE_OK,
        "Describe only features visible": "",
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create generated.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == ["generated.py"]
        assert result["failed_tasks"] == []
        # Metadata under .agent_runs/<run_id>/, no junk in the workspace root
        run_dir = Path(tmp) / RUNS_DIR_NAME / result["run_id"]
        assert (run_dir / "state.json").exists()
        assert (run_dir / "events.jsonl").exists()
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        assert state["status"] == "completed"
        assert state["llm_calls"] > 0
        root = Path(tmp)
        assert not (root / "generated.py").exists()  # staged, not deployed
        assert not (root / "summary.txt").exists()
        assert not (root / "tasks.md").exists()
        assert not list(root.glob("task*_code.txt"))


def test_budget_exhaustion_llm_calls() -> None:
    fake = _fake({
        "JSON array of tasks": PLAN,
        "TARGET FILE: generated.py": GOOD_CODE,
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create generated.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            budgets=RunBudgets(max_llm_calls=1),  # only the planner call fits
        ))
        assert result["status"] == "budget_exhausted"
        assert "LLM calls" in result["summary"]


def test_budget_exhaustion_wall_time() -> None:
    fake = _fake({
        "JSON array of tasks": PLAN,
        "TARGET FILE: generated.py": GOOD_CODE,
        "satisfies_request": VALIDATE_OK,
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create generated.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
            budgets=RunBudgets(max_wall_seconds=0.0),  # immediately exhausted
        ))
        assert result["status"] == "budget_exhausted"
        assert "wall time" in result["summary"]


def test_no_progress_detector_marks_needs_human() -> None:
    """Two consecutive attempts with the same diagnostic -> early stop marked
    needs human (identical failure twice)."""
    fake = _fake({
        "JSON array of tasks": PLAN,
        "TARGET FILE: generated.py": BROKEN_CODE,  # identical broken code each time
    })
    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create generated.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert len(result["failed_tasks"]) == 1
        assert "needs human" in result["failed_tasks"][0]["error"].lower()


def test_garbage_coder_fails_without_stub_files() -> None:
    """Coder returns garbage (extracts to prose) that fails CHECK twice with
    DIFFERENT diagnostics -> task fails, NO scaffold/stub files staged."""
    garbage_a = "```python\nthis is not python code at all just prose\n```"
    garbage_b = "```python\nstill prose but different words so hash differs here\n```"
    calls = {"n": 0}

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        import re
        if "JSON array of tasks" in prompt:
            return PLAN
        target = re.search(r"TARGET FILE:\s*(\S+)", prompt)
        if target:
            calls["n"] += 1
            return garbage_a if calls["n"] == 1 else garbage_b
        return ""

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create generated.py with a greet function",
            tmp, generate=fake, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert result["failed_tasks"]
        root = Path(tmp)
        assert not (root / "generated.py").exists()
        assert not list(root.glob("task*_code.txt"))
