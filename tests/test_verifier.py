"""Phase 4 tests: deterministic verifier step chain."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from agent_core.verifier import verify_files


def _stage(root: Path, name: str, content: str) -> dict[str, Path]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {name: path}


def test_py_compile_pass_and_chain_skips_pytest_without_tests_dir() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        staged = _stage(root, "ok.py", "def f():\n    return 1\n")
        result = verify_files(staged, root, workspace=root)
        assert result.passed, result.diagnostics()
        steps = [s.step for s in result.steps]
        assert steps[0] == "py_compile"
        assert "import_smoke" in steps
        # No tests/ directory -> pytest step skipped, not failed
        pytest_step = next(s for s in result.steps if s.step == "pytest")
        assert pytest_step.skipped and pytest_step.passed


def test_py_compile_failure_blocks_chain() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        staged = _stage(root, "bad.py", "def broken(:\n")
        result = verify_files(staged, root, workspace=root)
        assert not result.passed
        assert result.steps[0].step == "py_compile" and not result.steps[0].passed
        # Chain stops at the first failure
        assert all(s.step != "import_smoke" for s in result.steps)
        assert "def broken" in result.diagnostics() or "SyntaxError" in result.diagnostics()


def test_import_smoke_catches_runtime_import_errors() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        staged = _stage(root, "boom.py", "raise RuntimeError('import crash')\n")
        result = verify_files(staged, root, workspace=root)
        assert not result.passed
        smoke = next(s for s in result.steps if s.step == "import_smoke")
        assert not smoke.passed
        assert "import crash" in smoke.output


def test_import_smoke_resolves_package_relative_modules() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pkg").mkdir()
        (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        staged = _stage(root, "pkg/mod.py", "VALUE = 42\n")
        result = verify_files(staged, root, workspace=root)
        assert result.passed, result.diagnostics()


def test_empty_file_fails_fast() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        staged = _stage(root, "empty.py", "   \n")
        result = verify_files(staged, root, workspace=root)
        assert not result.passed
        assert result.steps[0].step == "non_empty"


def test_custom_command_bypasses_chain() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        staged = _stage(root, "ok.py", "X = 1\n")
        result = verify_files(
            staged, root, test_command=["python", "-c", "print('custom ok')"], workspace=root
        )
        assert result.passed
        assert result.steps[0].step == "custom_command"


def test_related_pytest_files_run_when_present() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        tests_dir = root / "tests"
        tests_dir.mkdir()
        # The test imports the staged module name; run against workspace where
        # the staged file has NOT been deployed, so the test should fail to
        # import... unless the module also exists in workspace. Deploy it first:
        _stage(root, "calc.py", "def add(a, b):\n    return a + b\n")
        (tests_dir / "test_calc.py").write_text(
            "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            encoding="utf-8",
        )
        result = verify_files({"calc.py": root / "calc.py"}, root, workspace=root)
        assert result.passed, result.diagnostics()
        pytest_step = next(s for s in result.steps if s.step == "pytest")
        assert not pytest_step.skipped
        assert pytest_step.passed
