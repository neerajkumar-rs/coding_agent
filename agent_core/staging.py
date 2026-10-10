"""Staging integration for the engine: all generated code lands in
``.agent_staging/`` first, is verified there, and is deployed to the
workspace only on success (or on explicit deploy).

Wraps WorkspaceStagingManager with path-safety checks. Phase 5 adds
protected-path rules and deploy backups.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from agent_core.coder import ALLOWED_TARGET_SUFFIXES
from workspace_staging import WorkspaceStagingManager


class PathEscapeError(ValueError):
    """Target path would leave the workspace or is otherwise not allowed."""


def safe_target(workspace: Path, relative_name: str) -> Path:
    """Resolve ``relative_name`` inside ``workspace`` or raise PathEscapeError."""
    candidate = (workspace / relative_name).resolve()
    root = workspace.resolve()
    if candidate != root and root not in candidate.parents:
        raise PathEscapeError("target_file must remain inside the workspace")
    if candidate.suffix.lower() not in ALLOWED_TARGET_SUFFIXES:
        raise PathEscapeError("target_file must end in .py, .html, .css, or .js")
    return candidate


class StagingArea:
    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.manager = WorkspaceStagingManager(self.workspace)

    def stage_files(self, files: dict[str, str]) -> dict[str, Path]:
        """Validate paths and write files into staging. Returns rel -> staged path."""
        staged: dict[str, Path] = {}
        for name, content in files.items():
            # Raises on escape/bad suffix; staging manager writes under .agent_staging.
            safe_target(self.workspace, name)
            staged[name] = self.manager.stage_file(name, content)
        return staged

    def read_staged(self, rel_path: str) -> str | None:
        return self.manager.read_staged_file(rel_path)

    def diff_for(self, files: dict[str, str], max_chars: int = 3_000) -> str:
        """Unified diff of newly staged content vs the live workspace (for the auditor)."""
        chunks: list[str] = []
        for name, content in files.items():
            target = self.workspace / name
            old = (
                target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                if target.exists()
                else []
            )
            new = content.splitlines(keepends=True)
            chunks.extend(
                difflib.unified_diff(old, new, fromfile=f"a/{name}", tofile=f"b/{name}", lineterm="")
            )
        return "\n".join(chunks)[:max_chars]

    def deploy(self) -> list[str]:
        result = self.manager.deploy_staged_files()
        return list(result.get("deployed_files", []))
