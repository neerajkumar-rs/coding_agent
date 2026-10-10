"""OS-aware cross-platform execution sandbox for automated validation."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from workspace_staging import WorkspaceStagingManager

logger = logging.getLogger("sandbox")


def tail_lines(text: str, count: int = 30) -> str:
    """Last ``count`` lines of an error log — line-bounded, never char-truncated."""
    lines = text.strip().splitlines()
    return "\n".join(lines[-count:])


def check_plausible_content(rel_path: str, content: str) -> str | None:
    """Non-Python plausibility check. Returns an error string, or None if OK.

    Fails: empty files, prose-only content, and leftover markdown fences.
    Valid HTML needs a doctype/<html> AND a <body> and reasonable length.
    """
    if not content.strip():
        return "file is empty"
    if "```" in content:
        return "leftover markdown fences (```) in file content"
    suffix = Path(rel_path).suffix.lower()
    if suffix == ".json":
        import json

        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            return f"invalid JSON: {exc}"
    elif suffix == ".html":
        lower = content.lower()
        if "<html" not in lower and "<!doctype" not in lower:
            return "HTML file contains neither <html> nor <!doctype"
        if "<body" not in lower:
            return "HTML file has no <body>"
        if len(content.strip()) < 40:
            return "HTML file is too short to be a real page"
    elif suffix == ".css":
        if "{" not in content or "}" not in content:
            return "CSS file contains no rule blocks"
    elif suffix == ".js":
        plausible = ("function", "=>", "const ", "let ", "var ", "document.", "module")
        if not any(token in content for token in plausible):
            return "JS file has no recognizable JavaScript constructs"
    return None


class ExecutionSandbox:
    """Detects host OS, generates native validation scripts (.ps1 on win32, .sh on unix),
    and executes them within a clean subprocess boundary.
    """

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        staging_manager: WorkspaceStagingManager | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
        self.staging_manager = staging_manager or WorkspaceStagingManager(self.workspace_root)
        self.platform = sys.platform

    def find_python_executable(self) -> str:
        """Find the best local python executable (prefers workspace env if available)."""
        if self.platform == "win32":
            candidates = [
                self.workspace_root / "env" / "Scripts" / "python.exe",
                self.workspace_root / ".venv" / "Scripts" / "python.exe",
                Path(sys.executable),
            ]
        else:
            candidates = [
                self.workspace_root / "env" / "bin" / "python",
                self.workspace_root / ".venv" / "bin" / "python",
                Path(sys.executable),
            ]
        for c in candidates:
            if c.exists() and os.access(c, os.X_OK):
                return str(c.resolve())
        return sys.executable

    def find_powershell_executable(self) -> str:
        """Find the native powershell or pwsh executable on Windows."""
        candidates = [
            shutil.which("pwsh.exe"),
            shutil.which("powershell.exe"),
            Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe",
            Path("C:/Program Files/PowerShell/7/pwsh.exe"),
            Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"),
        ]
        for c in candidates:
            if c:
                p = Path(c)
                if p.exists():
                    return str(p.resolve())
        return "powershell.exe"

    # Free-form shell is NOT allowed. Only these argument vectors may run,
    # and only when explicitly enabled in the workspace's .agent config
    # ("allowed_commands": ["py_compile", ...]). Anything else is rejected.
    COMMAND_ALLOWLIST: dict[str, list[str]] = {
        "py_compile": ["-m", "py_compile"],
        "pytest": ["-m", "pytest", "-q"],
    }

    def _load_allowed_commands(self) -> set[str]:
        """Read the allowlist from .agent (defaults to py_compile only)."""
        try:
            from agent_config import load_agent_config

            config = load_agent_config(self.workspace_root)
            allowed = getattr(config, "allowed_commands", None) or ["py_compile"]
            return {a for a in allowed if a in self.COMMAND_ALLOWLIST}
        except Exception:
            return {"py_compile"}

    def resolve_custom_command(self, name: str | None) -> list[str] | None:
        """Map an allowlisted command NAME to its argument vector.

        Free-form command strings are rejected — only names present in the
        workspace .agent allowlist resolve. Returns None when no command was
        requested; raises ValueError for anything not allowlisted.
        """
        if not name:
            return None
        key = name.strip()
        allowed = self._load_allowed_commands()
        if key not in allowed:
            raise ValueError(
                f"custom command {key!r} is not in the .agent allowlist "
                f"(allowed: {sorted(allowed)}); free-form commands are not supported"
            )
        return self.COMMAND_ALLOWLIST[key]

    def generate_validation_script(
        self,
        target_rel_path: str,
        custom_command: str | None = None,
    ) -> tuple[Path, list[str]]:
        """Generate native validation script (.ps1 on Windows, .sh on Unix) for the staged file."""
        staged_abs = self.staging_manager.get_staged_file_path(target_rel_path)
        py_exe = self.find_python_executable()
        custom_args = self.resolve_custom_command(custom_command)

        if self.platform == "win32":
            script_path = self.staging_manager.staging_dir / "_validate_step.ps1"
            script_lines = [
                "# PowerShell Validation Script for Staged Artefact",
                '$ErrorActionPreference = "Continue"',
                f'$env:PYTHONPATH = "{self.staging_manager.staging_dir};{self.workspace_root};" + $env:PYTHONPATH',
                f'Write-Host "[SANDBOX] Validating staged file: {staged_abs}"',
            ]

            if custom_args:
                joined = " ".join(f'"{py_exe}"' if i == 0 else a for i, a in enumerate([py_exe, *custom_args]))
                script_lines.append(f"Write-Host '[SANDBOX] Executing allowlisted command: {custom_command}'")
                script_lines.append(joined)
            elif target_rel_path.endswith(".py"):
                # In-memory syntax check — no file writes, no __pycache__,
                # works with spaces in paths. Path passed as an ARG, never
                # interpolated into code.
                script_lines.append(
                    f'& "{py_exe}" -c "import sys; compile(open(sys.argv[1], encoding=\'utf-8\').read(), sys.argv[1], \'exec\')" "{staged_abs}"'
                )
                script_lines.append('if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }')
                script_lines.append('Write-Host "[SANDBOX] Syntax compilation successful."')
            else:
                # Non-Python files never reach generate_validation_script:
                # execute_validation checks them in-process. Guard anyway.
                raise ValueError(
                    f"non-Python target {target_rel_path!r} is validated in-process; "
                    "no script is generated for it"
                )

            script_content = "\r\n".join(script_lines) + "\r\n"
            script_path.write_text(script_content, encoding="utf-8")

            pwsh_exe = self.find_powershell_executable()
            cmd = [
                pwsh_exe,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ]
            return script_path, cmd

        else:
            script_path = self.staging_manager.staging_dir / "_validate_step.sh"
            script_lines = [
                "#!/usr/bin/env bash",
                "set -e",
                f'export PYTHONPATH="{self.staging_manager.staging_dir}:{self.workspace_root}:$PYTHONPATH"',
                f'echo "[SANDBOX] Validating staged file: {staged_abs}"',
            ]

            if custom_args:
                quoted = " ".join([f'"{py_exe}"', *custom_args])
                script_lines.append(f'echo "[SANDBOX] Executing allowlisted command: {custom_command}"')
                script_lines.append(quoted)
            elif target_rel_path.endswith(".py"):
                # In-memory syntax check; path passed as an arg, never
                # interpolated into code.
                script_lines.append(
                    f'"{py_exe}" -c "import sys; compile(open(sys.argv[1], encoding=\'utf-8\').read(), sys.argv[1], \'exec\')" "{staged_abs}"'
                )
                script_lines.append('echo "[SANDBOX] Syntax compilation successful."')
            else:
                # Non-Python files never reach generate_validation_script.
                raise ValueError(
                    f"non-Python target {target_rel_path!r} is validated in-process; "
                    "no script is generated for it"
                )

            script_content = "\n".join(script_lines) + "\n"
            script_path.write_text(script_content, encoding="utf-8")
            try:
                os.chmod(script_path, 0o755)
            except Exception:
                pass

            cmd = ["/usr/bin/env", "bash", str(script_path)]
            return script_path, cmd

    async def execute_validation(
        self,
        target_rel_path: str,
        custom_command: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Validate a staged file.

        Non-Python files (and no custom command) are checked IN-PROCESS via
        check_plausible_content — no subprocess, no sys.path edits (the old
        generated-script approach failed with "No module named 'sandbox'"
        whenever the workspace was not the controller directory).
        """
        start_time = time.time()
        if not target_rel_path.endswith(".py") and not custom_command:
            staged_abs = self.staging_manager.get_staged_file_path(target_rel_path)
            content = (
                staged_abs.read_text(encoding="utf-8", errors="replace")
                if staged_abs.is_file()
                else ""
            )
            error = check_plausible_content(target_rel_path, content)
            duration_s = round(time.time() - start_time, 3)
            success = error is None
            stdout = "[SANDBOX] Content plausibility check passed." if success else ""
            stderr = "" if success else f"[SANDBOX] {error}"
            self.staging_manager.log_sandbox(
                command=f"in-process plausibility: {target_rel_path}",
                exit_code=0 if success else 1,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration_s,
            )
            return {
                "success": success,
                "exit_code": 0 if success else 1,
                "stdout": stdout,
                "stderr": stderr,
                "script_path": "",
                "duration_s": duration_s,
            }

        script_path, cmd = self.generate_validation_script(target_rel_path, custom_command)

        logger.info(f"Running sandbox diagnostic via: {' '.join(cmd)}")
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.workspace_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode or 0
        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception:
                pass
            stdout = ""
            stderr = f"Validation timed out after {timeout} seconds."
            exit_code = 124
        except Exception as exc:
            stdout = ""
            stderr = f"Subprocess invocation failure: {str(exc)}"
            exit_code = 1

        duration_s = round(time.time() - start_time, 3)
        success = exit_code == 0

        # Discrete operational logging to sandbox.log (full text in the log).
        self.staging_manager.log_sandbox(
            command=cmd,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration_s,
        )

        return {
            "success": success,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            # Caller-facing error: last ~30 lines, never char-truncated.
            "error_tail": tail_lines(stderr or stdout),
            "script_path": str(script_path),
            "duration_s": duration_s,
        }
