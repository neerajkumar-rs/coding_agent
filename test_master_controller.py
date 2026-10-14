"""Comprehensive test suite for Local LLM FastAPI Master Controller."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent_config import (
    AgentConfig,
    benchmark_and_calculate_tokens,
    load_agent_config,
    parse_and_recalculate,
    save_agent_config,
)
from agent_core.run_state import RunJournal
from main import app, state
from sandbox import ExecutionSandbox
from scanner import WorkspaceScanner
from workspace_staging import WorkspaceStagingManager


def test_agent_config_and_recalculate() -> None:
    """Test 1: Dynamic System Engine .agent profile and benchmark recalculation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Save and load config (legacy max_context_tokens is IGNORED in
        #    v4: the role defaults stand — the status bar must never show 4096
        #    for the defaults again).
        cfg = AgentConfig(
            planner_model="erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest",
            coder_5b_or_3b="qwen2.5-coder:3b",
            roles={"thinker": {"num_ctx": 8192}},
        )
        saved = save_agent_config(cfg, tmp_path)
        assert saved.exists()

        loaded = load_agent_config(tmp_path)
        assert loaded.planner_model == "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
        assert loaded.coder_model == "qwen2.5-coder:3b"
        assert loaded.max_context_tokens == 8192  # thinker default, not a legacy fold

        # 2. Benchmark calculation logic
        # 3B coder on RTX 3050 4GB ceiling should cap at 4096
        tokens_3b = benchmark_and_calculate_tokens("qwen2.5-coder:3b")
        assert tokens_3b == 4096

        # Manual override
        tokens_manual = benchmark_and_calculate_tokens("qwen2.5-coder:3b", manual_context=8192)
        assert tokens_manual == 8192

        # 3. CLI recalculate parser
        cli_result = parse_and_recalculate([
            "--recalculate",
            "--workspace", str(tmp_path),
            "--coder-model", "qwen2.5-coder:1.5b",
            "--max-context-tokens", "8192",
        ])
        assert cli_result is not None
        assert cli_result.coder_model == "qwen2.5-coder:1.5b"
        assert cli_result.max_context_tokens == 8192

        # Verify disk persistence
        disk_cfg = load_agent_config(tmp_path)
        assert disk_cfg.max_context_tokens == 8192
        print("Test 1 passed: Dynamic System Engine & Recalculation.")


def test_staging_isolation_and_difflib() -> None:
    """Test 2: Deferred file staging, logging matrix, and difflib comparison."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        staging = WorkspaceStagingManager(tmp_path)

        # Active workspace has an initial file
        orig_file = tmp_path / "app.py"
        orig_file.write_text("def hello():\n    return 'old'\n", encoding="utf-8")

        # Stage an update without touching app.py in active workspace
        staged_code = "def hello():\n    return 'new_vibe'\n"
        staging.stage_file("app.py", staged_code)

        # Verify isolation: app.py in active directory is UNTOUCHED
        assert orig_file.read_text(encoding="utf-8") == "def hello():\n    return 'old'\n"

        # Check difflib comparison
        diff_report = staging.compute_staging_diff()
        assert diff_report["has_changes"] is True
        assert diff_report["staged_count"] == 1
        diff_obj = diff_report["diffs"][0]
        assert diff_obj["file"] == "app.py"
        assert diff_obj["action"] == "edit"
        assert "-    return 'old'" in diff_obj["diff"]
        assert "+    return 'new_vibe'" in diff_obj["diff"]

        # Run logging lives in the per-run journal (the ONE logging path):
        # role logs are written through RunJournal, never to the workspace.
        journal = RunJournal(tmp_path)
        journal.log_llm_call("thinker", model="m", prompt="reasoning step 1",
                             response="plan ready")
        journal.log_llm_call("coder", model="m2", prompt="update greeting",
                             response=staged_code)
        journal.log_role("sandbox", "sandbox ok")
        assert "reasoning step 1" in journal.role_log_path("thinker").read_text(encoding="utf-8")
        assert "new_vibe" in journal.role_log_path("coder").read_text(encoding="utf-8")
        assert "sandbox ok" in journal.role_log_path("sandbox").read_text(encoding="utf-8")

        # Deploy staged files
        deploy_res = staging.deploy_staged_files()
        assert deploy_res["status"] == "success"
        assert "app.py" in deploy_res["deployed_files"]
        assert orig_file.read_text(encoding="utf-8") == staged_code

        # Staging should be flushed
        assert len(staging.list_staged_files()) == 0
        diff_after = staging.compute_staging_diff()
        assert diff_after["has_changes"] is False
        print("Test 2 passed: Staging isolation, difflib, and deployment.")


def test_sandbox_execution() -> None:
    """Test 3: OS-aware sandbox validation script generation and execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        staging = WorkspaceStagingManager(tmp_path)
        sandbox = ExecutionSandbox(tmp_path, staging)

        # Stage valid python file
        staging.stage_file("calc.py", "def add(a, b):\n    return a + b\n")

        # Execute validation
        res = asyncio.run(sandbox.execute_validation("calc.py"))
        assert res["success"] is True
        assert res["exit_code"] == 0
        assert "Syntax compilation successful" in res["stdout"] or "passed" in res["stdout"]

        # Stage invalid python file with syntax error
        staging.stage_file("broken.py", "def broken(\n")
        res_broken = asyncio.run(sandbox.execute_validation("broken.py"))
        assert res_broken["success"] is False
        assert res_broken["exit_code"] != 0

        print("Test 3 passed: OS-aware sandbox execution.")


def test_fastapi_endpoints() -> None:
    """Test 4: FastAPI master controller endpoints (init, run, status, deploy).

    The retired /agent/plan + /agent/step queue is superseded by the single
    pipeline endpoint POST /agent/run (ROUTE -> PLAN -> CODE -> CHECK ->
    VALIDATE -> SUMMARY), which stages CHECK-passed files without deploying.
    """
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. POST /workspace/init
        init_res = client.post(
            "/workspace/init",
            json={"root_path": str(tmp_path), "coder_model": "qwen2.5-coder:3b"},
        )
        assert init_res.status_code == 200
        init_data = init_res.json()
        assert init_data["status"] == "initialized"
        assert (tmp_path / ".agent").exists()
        assert (tmp_path / ".agent_staging").exists()
        # Per-run logging dirs are created when a run starts, not at init.
        assert init_data["staging_dir"].endswith(".agent_staging")
        assert init_data["runs_dir"].endswith(".agent_runs")

        # 2. POST /agent/run — the single pipeline. Mock the supervisor's
        #    generate to route on the prompt markers run_pipeline emits.
        plan_json = json.dumps([
            {"path": "greet.py", "action": "create",
             "description": "Create a greet function returning a greeting string"}
        ])
        good_code = "```python\ndef greet(name: str) -> str:\n    return f'Hello, {name}!'\n```"
        validate_ok = json.dumps({
            "satisfies_request": True, "problems": [], "summary": "A greet module."
        })

        async def mock_generate(role_or_model, prompt, **kwargs):
            if "JSON array of tasks" in prompt:
                return plan_json
            if "TARGET FILE: greet.py" in prompt:
                return good_code
            if "satisfies_request" in prompt:
                return validate_ok
            if "Describe only features visible" in prompt:
                return ""
            return ""

        # The pipeline now funnels through state.gateway.generate (the single
        # process-wide LLM gateway), so patch THAT — the true entry point.
        with patch.object(state.gateway, "generate", side_effect=mock_generate):
            run_res = client.post(
                "/agent/run",
                json={"prompt": "create greet.py with a greet function"},
            )
            assert run_res.status_code == 200, run_res.text
            run_data = run_res.json()
            assert "greet.py" in run_data["staged_files"]
            assert run_data["failed_tasks"] == []
            stages = [t["stage"] for t in run_data["trace"]]
            assert "PLAN" in stages and "CHECK" in stages and "SUMMARY" in stages

            # Verify file was written to staging, NOT production root
            staged_file = tmp_path / ".agent_staging" / "greet.py"
            prod_file = tmp_path / "greet.py"
            assert staged_file.exists()
            assert not prod_file.exists()

            # Run metadata lives under .agent_runs/<run_id>/
            run_dir = tmp_path / ".agent_runs" / run_data["run_id"]
            assert (run_dir / "state.json").exists()
            assert (run_dir / "events.jsonl").exists()

        # 3. POST /workspace/status
        status_res = client.post("/workspace/status", json={})
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["has_changes"] is True
        assert status_data["staged_count"] == 1
        assert "greet.py" in status_data["diffs"][0]["file"]

        # 4. POST /agent/deploy
        deploy_res = client.post("/agent/deploy", json={})
        assert deploy_res.status_code == 200
        deploy_data = deploy_res.json()
        assert deploy_data["status"] == "success"
        assert "greet.py" in deploy_data["deployed_files"]

        # Production file should now exist
        assert prod_file.exists()
        assert "Hello" in prod_file.read_text(encoding="utf-8")

        # Status after deploy should show no changes
        status_after = client.post("/workspace/status", json={}).json()
        assert status_after["has_changes"] is False

        print("Test 4 passed: Complete FastAPI Master Controller lifecycle (single pipeline).")


if __name__ == "__main__":
    test_agent_config_and_recalculate()
    test_staging_isolation_and_difflib()
    test_sandbox_execution()
    test_fastapi_endpoints()
    print("\nALL MASTER CONTROLLER TESTS PASSED SUCCESSFULLY!")
