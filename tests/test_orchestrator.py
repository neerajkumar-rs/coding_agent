"""Unit tests for the hardened orchestrator: budgets, no-progress, no scaffolds."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import (
    RUNS_DIR_NAME,
    RunBudgets,
    VibeOrchestrator,
)


class StubEngine:
    """Scriptable stand-in for OllamaEngine (only .chat is used)."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def chat(self, model: str, prompt: str, system: str = "", max_tokens=None, temperature=0.1) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return "PASS\nlooks fine"

    @property
    def active_model(self) -> str:
        return "stub"


PLAN = "## TASK 1: create greeting\n- files: generated.py\n- logic: def greet(name: str) -> str"
GOOD_CODE = "```FILE: generated.py\n\ndef greet(name: str) -> str:\n    return f'Hello, {name}!'\n```"
GARBAGE_A = "i have no idea what you want, here is a poem about bananas"
GARBAGE_B = "still no clue, maybe try asking a bigger model?"


def _run(engine: StubEngine, tmp: str, budgets: RunBudgets | None = None):
    orch = VibeOrchestrator(engine=engine, workspace=tmp, budgets=budgets)
    return orch.run("Create a greeting function", "generated.py")


def test_garbage_coder_fails_without_stub_files() -> None:
    """Coder returns garbage twice -> run ends `failed`, NO scaffold/stub files.

    (The two garbage replies differ, so diagnostics differ and the no-progress
    rule does not fire — the task simply exhausts its attempts.)
    """
    with TemporaryDirectory() as tmp:
        engine = StubEngine([PLAN, GARBAGE_A, "refined spec", GARBAGE_B])
        result = _run(engine, tmp)
        assert result.status == "failed", result
        assert any("no usable files" in e for e in result.errors)
        root = Path(tmp)
        # No stub files, no junk metadata in the workspace root
        assert not (root / "generated.py").exists()
        assert not (root / "summary.txt").exists()
        assert not (root / "tasks.md").exists()
        assert not list(root.glob("task*_code.txt"))
        # Run metadata exists under .agent_runs/
        run_dir = root / RUNS_DIR_NAME / result.run_id
        assert (run_dir / "state.json").exists()
        assert (run_dir / "events.jsonl").exists()


def test_budget_exhaustion_llm_calls() -> None:
    with TemporaryDirectory() as tmp:
        engine = StubEngine([PLAN, GOOD_CODE, "PASS\nok"])
        budgets = RunBudgets(max_llm_calls=1)  # only the planner call fits
        result = _run(engine, tmp, budgets)
        assert result.status == "budget_exhausted", result
        assert any("LLM calls" in e for e in result.errors)


def test_budget_exhaustion_wall_time() -> None:
    with TemporaryDirectory() as tmp:
        engine = StubEngine([PLAN, GOOD_CODE, "PASS\nok"])
        budgets = RunBudgets(max_wall_seconds=0.0)  # immediately exhausted
        result = _run(engine, tmp, budgets)
        assert result.status == "budget_exhausted", result
        assert any("wall time" in e for e in result.errors)


def test_no_progress_detector_marks_needs_human() -> None:
    """Two consecutive attempts with the same diagnostic hash -> needs_human."""
    with TemporaryDirectory() as tmp:
        # Coder emits uncompilable code; identical failure twice in a row.
        bad_code = "```FILE: generated.py\ndef broken(:\n```"
        engine = StubEngine([PLAN, bad_code, "same refined spec", bad_code])
        result = _run(engine, tmp)
        assert result.status == "needs_human", result
        assert result.needs_human, result
        assert result.attempts == 2


def test_happy_path_and_metadata_location() -> None:
    with TemporaryDirectory() as tmp:
        engine = StubEngine([PLAN, GOOD_CODE, "PASS\nall good"])
        result = _run(engine, tmp)
        assert result.status == "completed", result
        assert (Path(tmp) / "generated.py").exists()
        run_dir = Path(tmp) / RUNS_DIR_NAME / result.run_id
        assert (run_dir / "tasks.md").exists()
        events = (run_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert any('"run_finished"' in line for line in events)
