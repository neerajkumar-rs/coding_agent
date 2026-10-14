"""Unit tests for run_pipeline hardening: budgets, no-progress, no scaffolds,
run metadata under .agent_runs/. (Ported from the retired VibeOrchestrator
shim onto the single pipeline behind POST /agent/run.)"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import RUNS_DIR_NAME, RunBudgets, run_pipeline

PLANNER = "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
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
    """Loop-aware fake: 'JSON array of tasks' becomes write_file actions +
    finish; 'TARGET FILE: x' routes the coder write call for x."""
    import re

    plan = routes.get("JSON array of tasks", "")
    actions: list[dict] = []
    if plan.lstrip().startswith("["):
        for task in json.loads(plan):
            actions.append({
                "action": "write_file" if task.get("action", "create") == "create" else "edit_file",
                "path": task["path"],
                "text": task.get("description", ""),
            })
        actions.append({"action": "finish"})

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "AGENT ACTION JSON:" in prompt:
            if not actions:
                return json.dumps({"action": "reply", "text": "ok"})
            # play the actions in order, then finish forever
            idx = getattr(fake, "_n", 0)
            fake._n = idx + 1
            return json.dumps(actions[min(idx, len(actions) - 1)])
        target = re.search(r"TARGET FILE:\s*(\S+)", prompt) or re.search(r"Write\s+(\S+\.\w+)\.", prompt)
        if target:
            marker = f"TARGET FILE: {target.group(1)}"
            if marker in routes:
                return routes[marker]
        for marker in sorted(routes, key=len, reverse=True):
            if marker in ("JSON array of tasks",):
                continue
            if marker in prompt:
                return routes[marker]
        return ""

    fake._n = 0  # type: ignore[attr-defined]
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
    """The model repeating the same failing write twice -> the loop stops
    (no-progress) and the turn is needs_human, never a silent retry burn."""
    fake = _fake({
        "JSON array of tasks": PLAN,
        "TARGET FILE: generated.py": BROKEN_CODE,  # identical broken code each time
    })
    # The loop model keeps asking to write the same file (the no-progress
    # rule catches it after one repeat).
    loop_actions = [
        {"action": "write_file", "path": "generated.py", "text": "the greet function"},
        {"action": "write_file", "path": "generated.py", "text": "the greet function"},
    ]
    fake2 = _fake({
        "JSON array of tasks": PLAN,
        "TARGET FILE: generated.py": BROKEN_CODE,
    })
    fake2._n = 0

    async def drive(model, prompt, system="", temperature=0.2, **kw):
        if "AGENT ACTION JSON:" in prompt:
            idx = drive._n
            drive._n += 1
            return json.dumps(loop_actions[min(idx, len(loop_actions) - 1)])
        return await fake2(model, prompt, system=system, temperature=temperature, **kw)
    drive._n = 0  # type: ignore[attr-defined]

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "create generated.py with a greet function",
            tmp, generate=drive, planner_model=PLANNER, coder_model=CODER,
        ))
        assert result["staged_files"] == []
        assert result["status"] == "needs_human"
        assert "progress" in result["summary"].lower()


def test_garbage_coder_fails_without_stub_files() -> None:
    """Coder returns garbage (extracts to prose) that fails CHECK twice with
    DIFFERENT diagnostics -> task fails, NO scaffold/stub files staged."""
    garbage_a = "```python\nthis is not python code at all just prose\n```"
    garbage_b = "```python\nstill prose but different words so hash differs here\n```"
    calls = {"n": 0}
    loop_actions = iter([
        {"action": "write_file", "path": "generated.py", "text": "the greet function"},
        {"action": "finish"},
    ])

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        import re
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps(next(loop_actions, {"action": "finish"}))
        target = re.search(r"TARGET FILE:\s*(\S+)", prompt) or re.search(r"Write\s+(\S+\.\w+)\.", prompt)
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
