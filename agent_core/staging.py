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

# Path segments that must never be written or deployed to.
PROTECTED_SEGMENTS = {".git", ".agent", ".agent_runs", ".agent_staging", "env", ".venv", "__pycache__"}


class PathEscapeError(ValueError):
    """Target path would leave the workspace or is otherwise not allowed."""


def is_protected(relative_name: str) -> bool:
    parts = {p.lower() for p in Path(relative_name.replace("\\", "/")).parts}
    return bool(parts & PROTECTED_SEGMENTS)


def safe_target(workspace: Path, relative_name: str) -> Path:
    """Resolve ``relative_name`` inside ``workspace`` or raise PathEscapeError."""
    if is_protected(relative_name):
        raise PathEscapeError(f"target path is protected: {relative_name}")
    candidate = (workspace / relative_name).resolve()
    root = workspace.resolve()
    if candidate != root and root not in candidate.parents:
        raise PathEscapeError("target_file must remain inside the workspace")
    if candidate.suffix.lower() not in ALLOWED_TARGET_SUFFIXES:
        raise PathEscapeError(
            f"target_file must end in one of {sorted(ALLOWED_TARGET_SUFFIXES)}"
        )
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

    def deploy(self, backup_dir: Path | None = None) -> list[str]:
        """Deploy staged files, backing up overwritten files to ``backup_dir``
        (typically .agent_runs/<run_id>/backup/) so /agent/rollback can restore.
        Protected paths are rejected before anything is written."""
        staged_files = self.manager.list_staged_files()
        for rel in staged_files:
            safe_target(self.workspace, rel)  # raises on protected/escape

        if backup_dir is not None:
            backup_dir = Path(backup_dir)
            for rel in staged_files:
                existing = self.workspace / rel
                if existing.is_file():
                    dest = backup_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(existing.read_bytes())

        result = self.manager.deploy_staged_files()
        return list(result.get("deployed_files", []))

    @staticmethod
    def rollback(workspace: Path, backup_dir: Path) -> list[str]:
        """Restore files from a run's backup directory into the workspace."""
        workspace = Path(workspace).resolve()
        backup_dir = Path(backup_dir)
        restored: list[str] = []
        if not backup_dir.is_dir():
            return restored
        for backup_file in backup_dir.rglob("*"):
            if not backup_file.is_file():
                continue
            rel = backup_file.relative_to(backup_dir).as_posix()
            safe_target(workspace, rel)
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(backup_file.read_bytes())
            restored.append(rel)
        return restored
