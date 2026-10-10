"""Deterministic verifier. The ONLY gate that can pass or fail a task.

Step chain (each step produces a structured StepResult; first failure stops
the chain):
    1. py_compile        — syntax validity of staged .py files
    2. ruff              — lint, only if ruff is importable in this env
    3. import smoke      — import each staged module in a fresh subprocess
                           (staging dir on sys.path), catching import-time crashes
    4. pytest            — only when the workspace has a tests/ directory and
                           test files referencing the staged modules exist

A caller-supplied test_command bypasses the chain (single custom step).
The verifier is stdlib-first: ruff/pytest steps are skipped (not failed) when
the tools are unavailable, so the pipeline runs with zero heavy dependencies.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

VALIDATION_TIMEOUT_SECONDS = 300
MAX_LOG_CHARS = 8_000


@dataclass
class StepResult:
    step: str
    passed: bool
    skipped: bool = False
    output: str = ""


@dataclass
class VerifyResult:
    passed: bool
    steps: list[StepResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    def diagnostics(self, max_chars: int = MAX_LOG_CHARS) -> str:
        """Combined human-readable diagnostics for retry feedback."""
        parts: list[str] = []
        for step in self.steps:
            if step.skipped:
                continue
            status = "PASS" if step.passed else "FAIL"
            parts.append(f"[{status}] {step.step}")
            if step.output and not step.passed:
                parts.append(step.output)
        combined = "\n".join(parts)
        if self.stdout:
            combined += "\n" + self.stdout
        if self.stderr:
            combined += "\n" + self.stderr
        return combined.strip()[-max_chars:]


def _run(command: list[str], cwd: Path, step: str) -> StepResult:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=VALIDATION_TIMEOUT_SECONDS,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = (
            f"{step} timed out after {VALIDATION_TIMEOUT_SECONDS}s\n"
            f"{str(exc.stderr or '')}"
        )
        return StepResult(step=step, passed=False, output=output[-MAX_LOG_CHARS:])
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return StepResult(
        step=step, passed=completed.returncode == 0, output=output[-MAX_LOG_CHARS:]
    )


def _module_name(relative_name: str) -> str:
    return relative_name[:-3].replace("/", ".").replace("\\", ".")


def _import_smoke(python_rel_names: list[str], staging_root: Path) -> StepResult:
    """Import each staged module in a fresh subprocess with staging on the path."""
    failures: list[str] = []
    for rel in python_rel_names:
        module = _module_name(rel)
        if module.split(".")[-1] == "__init__":
            module = module.rsplit(".", 1)[0]
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import importlib; importlib.import_module(sys.argv[2])"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, str(staging_root), module],
            cwd=staging_root,
            capture_output=True,
            text=True,
            timeout=60,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            failures.append(f"import {module} failed:\n{completed.stderr.strip()}")
    output = "\n".join(failures)[-MAX_LOG_CHARS:]
    return StepResult(step="import_smoke", passed=not failures, output=output)


def _find_related_tests(
    python_rel_names: list[str], workspace: Path
) -> list[str]:
    """Test files under tests/ whose name matches a staged module."""
    tests_dir = workspace / "tests"
    if not tests_dir.is_dir():
        return []
    stems = {Path(name).stem for name in python_rel_names}
    found: list[str] = []
    for test_file in tests_dir.rglob("test_*.py"):
        stem = test_file.stem.removeprefix("test_")
        if stem in stems:
            found.append(str(test_file))
    return sorted(found)


def verify_files(
    staged_files: dict[str, Path],
    staging_root: Path,
    test_command: list[str] | None = None,
    workspace: Path | None = None,
) -> VerifyResult:
    """Run the deterministic step chain over staged files.

    ``staged_files`` maps relative name -> staged path; ``workspace`` is used
    to locate related tests (pytest step) and skips them when absent.
    NOTE: ruff runs with --no-cache so it never writes into the workspace.
    """
    result = VerifyResult(passed=False)

    def finish() -> VerifyResult:
        # First failing step's output becomes stderr; stdout collects passes.
        failing = next((s for s in result.steps if not s.passed and not s.skipped), None)
        result.passed = failing is None
        if failing:
            result.stderr = failing.output
            result.stdout = "\n".join(
                s.output for s in result.steps if s.passed and s.output
            )
        else:
            result.stdout = "\n".join(
                s.output for s in result.steps if not s.skipped and s.output
            )
            result.stderr = ""
        return result

    # 0) Non-empty check
    empty = [
        name
        for name, path in staged_files.items()
        if not path.read_text(encoding="utf-8", errors="replace").strip()
    ]
    if empty:
        result.steps.append(
            StepResult(step="non_empty", passed=False, output="FILE EMPTY: " + ", ".join(empty))
        )
        return finish()

    # Custom command bypasses the chain entirely.
    if test_command is not None:
        if not test_command or any(not isinstance(c, str) or not c for c in test_command):
            raise ValueError("test_command must be a non-empty list of strings")
        result.steps.append(_run(test_command, staging_root, "custom_command"))
        return finish()

    python_rel = [name for name in staged_files if name.endswith(".py")]
    if not python_rel:
        # Non-Python files: non-empty is all we can check deterministically.
        result.steps.append(StepResult(step="non_empty", passed=True))
        return finish()

    # 1) py_compile
    step = _run(
        [sys.executable, "-m", "py_compile", *[str(staged_files[n]) for n in python_rel]],
        staging_root,
        "py_compile",
    )
    result.steps.append(step)
    if not step.passed:
        return finish()

    # 2) ruff (optional tool: skipped when unavailable, never a hard dep)
    ruff_cmd: list[str] | None = None
    if shutil.which("ruff"):
        ruff_cmd = ["ruff", "check", "--no-cache"]
    else:
        try:
            import ruff  # noqa: F401

            ruff_cmd = [sys.executable, "-m", "ruff", "check", "--no-cache"]
        except ImportError:
            ruff_cmd = None
    if ruff_cmd is not None:
        step = _run(
            [*ruff_cmd, *[str(staged_files[n]) for n in python_rel]],
            staging_root,
            "ruff",
        )
        result.steps.append(step)
        if not step.passed:
            return finish()
    else:
        result.steps.append(StepResult(step="ruff", passed=True, skipped=True))

    # 3) import smoke
    step = _import_smoke(python_rel, staging_root)
    result.steps.append(step)
    if not step.passed:
        return finish()

    # 4) pytest on related tests (only when a tests/ tree references them)
    related: list[str] = []
    if workspace is not None:
        related = _find_related_tests(python_rel, Path(workspace))
    if related:
        try:
            import pytest  # noqa: F401

            pytest_available = True
        except ImportError:
            pytest_available = False
        if pytest_available:
            step = _run(
                [sys.executable, "-m", "pytest", "-q", *related],
                Path(workspace) if workspace else staging_root,
                "pytest",
            )
            result.steps.append(step)
            if not step.passed:
                return finish()
        else:
            result.steps.append(StepResult(step="pytest", passed=True, skipped=True))
    else:
        result.steps.append(StepResult(step="pytest", passed=True, skipped=True))

    return finish()
