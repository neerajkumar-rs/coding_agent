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

    def generate_validation_script(
        self,
        target_rel_path: str,
        custom_command: str | None = None,
    ) -> tuple[Path, list[str]]:
        """Generate native validation script (.ps1 on Windows, .sh on Unix) for the staged file."""
        staged_abs = self.staging_manager.get_staged_file_path(target_rel_path)
        py_exe = self.find_python_executable()

        if self.platform == "win32":
            script_path = self.staging_manager.staging_dir / "_validate_step.ps1"
            script_lines = [
                "# PowerShell Validation Script for Staged Artefact",
                '$ErrorActionPreference = "Continue"',
                f'$env:PYTHONPATH = "{self.staging_manager.staging_dir};{self.workspace_root};" + $env:PYTHONPATH',
                f'Write-Host "[SANDBOX] Validating staged file: {staged_abs}"',
            ]

            if custom_command:
                script_lines.append(f"Write-Host '[SANDBOX] Executing custom command: {custom_command}'")
                script_lines.append(f"{custom_command}")
            elif target_rel_path.endswith(".py"):
                script_lines.append(
                    f'& "{py_exe}" -m py_compile "{staged_abs}"'
                )
                script_lines.append('if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }')
                script_lines.append('Write-Host "[SANDBOX] Syntax compilation successful."')
            else:
                script_lines.append(
                    f'if (Test-Path "{staged_abs}") {{ Write-Host "[SANDBOX] File verification passed." }} else {{ Write-Error "File not found"; exit 1 }}'
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

            if custom_command:
                script_lines.append(f'echo "[SANDBOX] Executing custom command: {custom_command}"')
                script_lines.append(f"{custom_command}")
            elif target_rel_path.endswith(".py"):
                script_lines.append(
                    f'"{py_exe}" -m py_compile "{staged_abs}"'
                )
                script_lines.append('echo "[SANDBOX] Syntax compilation successful."')
            else:
                script_lines.append(
                    f'test -f "{staged_abs}" && echo "[SANDBOX] File verification passed."'
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
        """Execute the generated validation script in an isolated subprocess boundary."""
        script_path, cmd = self.generate_validation_script(target_rel_path, custom_command)
        start_time = time.time()

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

        # Discrete operational logging to sandbox.log
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
            "script_path": str(script_path),
            "duration_s": duration_s,
        }
