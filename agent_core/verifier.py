"""Deterministic verifier. The ONLY gate that can pass or fail a task.

Phase 2: py_compile on staged .py files (or a caller-supplied test command).
Phase 4 extends this with ruff -> import smoke -> pytest, each producing a
structured StepResult; the shape below is already that shape.
"""

from __future__ import annotations

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
    output: str = ""


@dataclass
class VerifyResult:
    passed: bool
    steps: list[StepResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    def diagnostics(self, max_chars: int = MAX_LOG_CHARS) -> str:
        return (self.stdout + "\n" + self.stderr).strip()[-max_chars:]


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


def verify_files(
    staged_files: dict[str, Path],
    staging_root: Path,
    test_command: list[str] | None = None,
) -> VerifyResult:
    """Verify staged files. ``staged_files`` maps relative name -> staged path."""
    result = VerifyResult(passed=False)

    empty = [name for name, path in staged_files.items() if not path.read_text(
        encoding="utf-8", errors="replace").strip()]
    if empty:
        result.steps.append(
            StepResult(step="non_empty", passed=False, output="FILE EMPTY: " + ", ".join(empty))
        )
        result.stderr = result.steps[-1].output
        return result

    if test_command is not None:
        if not test_command or any(not isinstance(c, str) or not c for c in test_command):
            raise ValueError("test_command must be a non-empty list of strings")
        step = _run(test_command, staging_root, "custom_command")
        result.steps.append(step)
        result.passed = step.passed
        result.stdout, result.stderr = step.output, "" if step.passed else step.output
        return result

    python_files = [p for name, p in staged_files.items() if name.endswith(".py")]
    if python_files:
        step = _run(
            [sys.executable, "-m", "py_compile", *map(str, python_files)],
            staging_root,
            "py_compile",
        )
        result.steps.append(step)
        result.passed = step.passed
        if step.passed:
            result.stdout = step.output
        else:
            result.stderr = step.output
        return result

    # Non-Python files: non-empty is all we can check deterministically.
    result.steps.append(StepResult(step="non_empty", passed=True))
    result.passed = True
    return result
