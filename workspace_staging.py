"""Architectural Isolation & Deferred File Staging Manager.

Enforces zero-touch active directory execution: all code generation occurs in
.agent_staging/, with fast atomic deployment. Run logging lives in the
per-run journal (.agent_runs/<run_id>/, see agent_core.run_state) — the old
workspace-level log matrix is retired.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("staging_manager")

STAGING_DIR_NAME = ".agent_staging"


class WorkspaceStagingManager:
    """Manages the isolated staging sandbox (.agent_staging/) and deployment.

    Per-run logging lives in the run journal (.agent_runs/<run_id>/); this
    class intentionally has no logging surface.
    """

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
        self.staging_dir = self.workspace_root / STAGING_DIR_NAME

        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create the isolated staging sandbox folder."""
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Deferred Staging Memory
    # -------------------------------------------------------------------------

    def stage_file(self, rel_path: str, content: str) -> Path:
        """Write generated code exclusively to .agent_staging/ without touching active workspace."""
        clean_rel = Path(rel_path.strip().lstrip("/\\"))
        target_path = self.staging_dir / clean_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        logger.info(f"Staged file: {clean_rel} -> {target_path}")
        return target_path

    def get_staged_file_path(self, rel_path: str) -> Path:
        clean_rel = Path(rel_path.strip().lstrip("/\\"))
        return self.staging_dir / clean_rel

    def list_staged_files(self) -> list[str]:
        """List all relative paths currently in staging, excluding internal sandbox scripts."""
        if not self.staging_dir.exists():
            return []
        files = []
        for p in self.staging_dir.rglob("*"):
            if p.is_file():
                # Ignore internal sandbox runner scripts
                if p.name.startswith("_") or any(part.startswith("_") for part in p.parts):
                    continue
                rel = p.relative_to(self.staging_dir).as_posix()
                files.append(rel)
        return sorted(files)

    def read_staged_file(self, rel_path: str) -> str | None:
        path = self.get_staged_file_path(rel_path)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
        return None

    def flush_staging(self) -> None:
        """Flush the .agent_staging/ cache."""
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Flushed staging cache.")

    # -------------------------------------------------------------------------
    # Native Difflib Comparison
    # -------------------------------------------------------------------------

    def compute_staging_diff(self) -> dict[str, Any]:
        """Compare .agent_staging/ against the active workspace root via difflib.
        
        Returns a line-by-line git-style diff object ready for TUI/GUI previews.
        """
        staged_files = self.list_staged_files()
        file_diffs: list[dict[str, Any]] = []
        total_additions = 0
        total_deletions = 0

        for rel in staged_files:
            staged_path = self.staging_dir / rel
            workspace_path = self.workspace_root / rel

            staged_content = staged_path.read_text(encoding="utf-8", errors="replace")
            staged_lines = staged_content.splitlines(keepends=True)

            action = "create"
            if workspace_path.exists() and workspace_path.is_file():
                action = "edit"
                original_content = workspace_path.read_text(encoding="utf-8", errors="replace")
                original_lines = original_content.splitlines(keepends=True)
            else:
                original_lines = []

            diff_generator = difflib.unified_diff(
                original_lines,
                staged_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
            diff_lines = list(diff_generator)

            additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
            total_additions += additions
            total_deletions += deletions

            file_diffs.append({
                "file": rel,
                "action": action,
                "diff": "\n".join(diff_lines),
                "diff_lines": diff_lines,
                "additions": additions,
                "deletions": deletions,
            })

        return {
            "has_changes": len(file_diffs) > 0,
            "staged_count": len(file_diffs),
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            "diffs": file_diffs,
        }

    # -------------------------------------------------------------------------
    # Deployment Hook
    # -------------------------------------------------------------------------

    def deploy_staged_files(self) -> dict[str, Any]:
        """Atomic fast file-system deployment.
        
        Maps, arranges, and overwrites files from staging into the active workspace,
        then flushes the staging cache.
        """
        staged_files = self.list_staged_files()
        if not staged_files:
            return {
                "status": "noop",
                "message": "No staged files to deploy",
                "deployed_files": [],
            }

        deployed: list[str] = []
        for rel in staged_files:
            source = self.staging_dir / rel
            dest = self.workspace_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Atomic copy
            temp_dest = dest.with_name(f"{dest.name}.tmp.{int(time.time()*1000)}")
            shutil.copy2(source, temp_dest)
            temp_dest.replace(dest)
            deployed.append(rel)

        # Flush staging
        self.flush_staging()

        logger.info(f"Deployed {len(deployed)} files into active workspace: {deployed}")
        return {
            "status": "success",
            "message": f"Successfully deployed {len(deployed)} files to active workspace.",
            "deployed_files": deployed,
        }
