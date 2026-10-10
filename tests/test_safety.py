"""Phase 5 tests: path-escape rejection, protected paths, sandbox allowlist."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_core.staging import PathEscapeError, StagingArea, is_protected, safe_target
from sandbox import ExecutionSandbox


def test_path_escape_dotdot_rejected() -> None:
    with TemporaryDirectory() as tmp:
        with pytest.raises(PathEscapeError):
            safe_target(Path(tmp), "../evil.py")


def test_path_escape_absolute_rejected() -> None:
    with TemporaryDirectory() as tmp:
        with pytest.raises(PathEscapeError):
            safe_target(Path(tmp), "/etc/passwd.py")


def test_protected_paths_rejected() -> None:
    with TemporaryDirectory() as tmp:
        for bad in [".git/config.py", ".agent_runs/x.py", "env/lib.py", ".venv/a.py"]:
            assert is_protected(bad)
            with pytest.raises(PathEscapeError):
                safe_target(Path(tmp), bad)


def test_normal_targets_allowed() -> None:
    with TemporaryDirectory() as tmp:
        target = safe_target(Path(tmp), "src/app.py")
        assert target == (Path(tmp) / "src" / "app.py").resolve()


def test_staging_deploy_rejects_protected_and_escapes() -> None:
    with TemporaryDirectory() as tmp:
        area = StagingArea(Path(tmp))
        area.manager.stage_file("ok.py", "X = 1\n")
        area.manager.stage_file(".git/hooked.py", "X = 2\n")  # bypass safe_target on purpose
        with pytest.raises(PathEscapeError):
            area.deploy()


def test_rollback_restores_backed_up_files() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "app.py").write_text("OLD = 1\n", encoding="utf-8")
        backup = root / ".agent_runs" / "run-1" / "backup"
        backup.mkdir(parents=True)
        (backup / "app.py").write_text("NEWER = 2\n", encoding="utf-8")
        restored = StagingArea.rollback(root, backup)
        assert restored == ["app.py"]
        assert (root / "app.py").read_text(encoding="utf-8") == "NEWER = 2\n"


def test_sandbox_rejects_freeform_command() -> None:
    with TemporaryDirectory() as tmp:
        sandbox = ExecutionSandbox(Path(tmp))
        with pytest.raises(ValueError, match="allowlist"):
            sandbox.resolve_custom_command("rm -rf /")


def test_sandbox_allows_allowlisted_name() -> None:
    with TemporaryDirectory() as tmp:
        sandbox = ExecutionSandbox(Path(tmp))
        args = sandbox.resolve_custom_command("py_compile")
        assert args == ["-m", "py_compile"]
        assert sandbox.resolve_custom_command(None) is None


def test_sandbox_rejects_unknown_allowlist_name() -> None:
    with TemporaryDirectory() as tmp:
        sandbox = ExecutionSandbox(Path(tmp))
        with pytest.raises(ValueError, match="allowlist"):
            sandbox.resolve_custom_command("pytest")  # not enabled by default
