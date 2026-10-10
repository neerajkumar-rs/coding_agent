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
from main import app, state
from sandbox import ExecutionSandbox
from scanner import WorkspaceScanner
from workspace_staging import WorkspaceStagingManager


def test_agent_config_and_recalculate() -> None:
    """Test 1: Dynamic System Engine .agent profile and benchmark recalculation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Save and load default config
        cfg = AgentConfig(
            planner_model="deepseek-r1:1.5b",
            coder_5b_or_3b="qwen2.5-coder:3b",
            max_context_tokens=4096,
        )
        saved = save_agent_config(cfg, tmp_path)
        assert saved.exists()

        loaded = load_agent_config(tmp_path)
        assert loaded.planner_model == "deepseek-r1:1.5b"
        assert loaded.coder_model == "qwen2.5-coder:3b"
        assert loaded.max_context_tokens == 4096

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

        # Check logs matrix
        staging.log_planner("reasoning step 1", [{"id": 1, "action": "edit"}])
        staging.log_coder("app.py", staged_code, "update greeting")
        staging.log_sandbox(["test_cmd"], 0, "sandbox ok", "")

        assert staging.planner_log_path.exists()
        assert staging.coder_log_path.exists()
        assert staging.sandbox_log_path.exists()
        assert "reasoning step 1" in staging.planner_log_path.read_text(encoding="utf-8")
        assert "new_vibe" in staging.coder_log_path.read_text(encoding="utf-8")
        assert "sandbox ok" in staging.sandbox_log_path.read_text(encoding="utf-8")

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
    """Test 4: FastAPI master controller endpoints (init, plan, step, status, deploy)."""
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
        assert (tmp_path / ".agent_logs").exists()

        # 2. POST /agent/plan (mocking ollama thinker response)
        mock_thinker_output = (
            "<think>Analyzing request to create user greeting module...</think>\n"
            "```json\n"
            "{\n"
            '  "tasks": [\n'
            '    {"id": 1, "file": "greet.py", "action": "create", "description": "Create greet function"}\n'
            "  ]\n"
            "}\n"
            "```"
        )
        with patch.object(state.ollama, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_thinker_output
            plan_res = client.post("/agent/plan", json={"prompt": "Build greeting module"})
            assert plan_res.status_code == 200
            plan_data = plan_res.json()
            assert len(plan_data["tasks"]) == 1
            assert plan_data["tasks"][0]["file"] == "greet.py"
            assert plan_data["tasks"][0]["action"] == "create"

            # Check planner log was written
            planner_log = tmp_path / ".agent_logs" / "planner.log"
            assert planner_log.exists()
            assert "Analyzing request" in planner_log.read_text(encoding="utf-8")

        # 3. POST /agent/step (mocking coder stream response)
        async def mock_stream(*args, **kwargs):
            yield "```python\n"
            yield "def greet(name: str) -> str:\n"
            yield "    return f'Hello, {name}!'\n"
            yield "```"

        with patch.object(state.ollama, "stream_generate", side_effect=mock_stream):
            step_res = client.post("/agent/step", json={"task_id": 1})
            assert step_res.status_code == 200
            step_data = step_res.json()
            assert step_data["status"] == "completed"
            assert step_data["diagnostic"]["success"] is True

            # Verify file was written to staging, NOT production root
            staged_file = tmp_path / ".agent_staging" / "greet.py"
            prod_file = tmp_path / "greet.py"
            assert staged_file.exists()
            assert not prod_file.exists()

            # Check coder and sandbox logs
            assert (tmp_path / ".agent_logs" / "coder.log").exists()
            assert (tmp_path / ".agent_logs" / "sandbox.log").exists()

        # 4. POST /workspace/status
        status_res = client.post("/workspace/status", json={})
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["has_changes"] is True
        assert status_data["staged_count"] == 1
        assert "greet.py" in status_data["diffs"][0]["file"]

        # 5. POST /agent/deploy
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

        print("Test 4 passed: Complete FastAPI Master Controller lifecycle.")


if __name__ == "__main__":
    test_agent_config_and_recalculate()
    test_staging_isolation_and_difflib()
    test_sandbox_execution()
    test_fastapi_endpoints()
    print("\nALL MASTER CONTROLLER TESTS PASSED SUCCESSFULLY!")
