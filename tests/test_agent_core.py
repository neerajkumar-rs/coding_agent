"""agent_core library tests: schemas, coder parsing, session registry.

The Engine/VibeOrchestrator pipeline is retired; run_pipeline is the only
pipeline. The two old ``engine.run(...)`` behaviour tests are ported here onto
run_pipeline's fake ``generate`` seam (auto-deploy + staging retention).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_core.engine import get_engine
from agent_core.schemas import AuditResult, Plan
from orchestrator import run_pipeline

import pytest

PLANNER = "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
CODER = "qwen2.5-coder:3b-instruct"

PLAN_PAYLOAD = {
    "tasks": [
        {
            "id": 1,
            "name": "create greeting",
            "target_files": ["generated.py"],
            "spec": "def greet(name: str) -> str returning a greeting",
        }
    ]
}
GREETER_PLAN = json.dumps([
    {"path": "generated.py", "action": "create",
     "description": "Python module with a greet(name) -> str function"},
])
GREETER_CODE = (
    "```python\n"
    "def greet(name: str) -> str:\n"
    "    return f'Hello, {name}!'\n"
    "```"
)
VALIDATE_OK = json.dumps({
    "satisfies_request": True, "problems": [], "summary": "Greeter module."
})


def _fake_generate(routes: dict[str, str]):
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

    async def fake(model, prompt, system="", temperature=0.2, **kwargs):
        if "AGENT ACTION JSON:" in prompt:
            if not actions:
                return json.dumps({"action": "reply", "text": "ok"})
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


def _greeter_fake():
    return _fake_generate({
        "JSON array of tasks": GREETER_PLAN,
        "TARGET FILE: generated.py": GREETER_CODE,
        "satisfies_request": VALIDATE_OK,
    })


def test_plan_schema_validation() -> None:
    plan = Plan.model_validate(PLAN_PAYLOAD)
    assert plan.tasks[0].name == "create greeting"
    with pytest.raises(Exception):
        Plan.model_validate({"tasks": []})  # min_length=1


def test_audit_schema_validation() -> None:
    ok = AuditResult.model_validate({"verdict": "concerns", "issues": [
        {"severity": "warning", "message": "missing docstring"}]})
    assert ok.verdict == "concerns"
    with pytest.raises(Exception):
        AuditResult.model_validate({"verdict": "maybe", "issues": []})


def test_run_pipeline_happy_path_deploys() -> None:
    """Ported from the retired Engine.run happy-path test: a full run through
    run_pipeline stages the file, and /save-style deploy lands it on disk."""
    with TemporaryDirectory() as tmp:
        result = asyncio.run(run_pipeline(
            "create generated.py with a greeting function",
            tmp,
            generate=_greeter_fake(),
            planner_model=PLANNER,
            coder_model=CODER,
        ))
        assert result["staged_files"] == ["generated.py"]
        # run_pipeline stages; deploy happens via the staging manager (/save).
        from workspace_staging import WorkspaceStagingManager
        mgr = WorkspaceStagingManager(tmp)
        outcome = mgr.deploy_staged_files()
        assert outcome["status"] == "success"
        assert "generated.py" in outcome["deployed_files"]
        assert (Path(tmp) / "generated.py").exists()
        # Staging flushed after deploy
        assert not list((Path(tmp) / ".agent_staging").rglob("*"))


def test_run_pipeline_stages_without_deploy() -> None:
    """Ported from the retired Engine.run auto_deploy=False test: the file is
    staged under .agent_staging/ and the workspace root stays untouched until
    the user explicitly saves."""
    with TemporaryDirectory() as tmp:
        result = asyncio.run(run_pipeline(
            "create generated.py with a greeting function",
            tmp,
            generate=_greeter_fake(),
            planner_model=PLANNER,
            coder_model=CODER,
        ))
        assert result["staged_files"] == ["generated.py"]
        assert not (Path(tmp) / "generated.py").exists()
        assert (Path(tmp) / ".agent_staging" / "generated.py").exists()


def test_session_registry_is_per_workspace() -> None:
    with TemporaryDirectory() as a, TemporaryDirectory() as b:
        ea, eb = get_engine(a), get_engine(b)
        assert ea is not eb
        assert get_engine(a) is ea  # same workspace -> same session
        assert get_engine(Path(a) / ".") is ea
        # The registry exists only to carry one cancel event per workspace.
        ea.cancel()
        assert ea._cancel_event.is_set()
        assert not eb._cancel_event.is_set()
