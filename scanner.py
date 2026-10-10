"""Workspace context analysis & environment scanner."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".agent_staging",
    ".agent_logs",
    "env",
    ".env",
    "venv",
    ".venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".agents",
}

# Junk artifacts left by the legacy orchestrator — never feed these to the planner.
JUNK_FILE_PATTERNS = ("tasks.md", "summary.txt", "task*_code.txt")

import fnmatch


def is_junk_file(filename: str) -> bool:
    lower = filename.lower()
    return any(fnmatch.fnmatch(lower, pattern) for pattern in JUNK_FILE_PATTERNS)


CONFIG_FILES = [
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
]


class WorkspaceScanner:
    """Scans workspace directory tree and project configuration files to build system context."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()

    def scan_directory_tree(self, max_depth: int = 3, max_files: int = 200) -> list[str]:
        """Scan active directory tree, filtering internal directories."""
        tree: list[str] = []
        count = 0

        for root, dirs, files in os.walk(self.workspace_root):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".agent")]

            rel_root = Path(root).relative_to(self.workspace_root)
            depth = len(rel_root.parts)
            if depth > max_depth:
                continue

            indent = "  " * depth
            folder_name = rel_root.name if rel_root.name else "."
            tree.append(f"{indent}[DIR] {folder_name}/")

            file_indent = "  " * (depth + 1)
            for f in sorted(files):
                if count >= max_files:
                    tree.append(f"{file_indent}... (file limit reached)")
                    return tree
                if f in IGNORED_DIRS or f.endswith(".pyc") or f == ".agent":
                    continue
                if is_junk_file(f):
                    continue
                tree.append(f"{file_indent}{f}")
                count += 1

        return tree

    def scan_project_configs(self) -> dict[str, str]:
        """Inspect and parse top-level project config files."""
        configs: dict[str, str] = {}
        for filename in CONFIG_FILES:
            target = self.workspace_root / filename
            if target.exists() and target.is_file():
                try:
                    content = target.read_text(encoding="utf-8", errors="replace").strip()
                    # Keep snippet if huge
                    if len(content) > 3000:
                        content = content[:3000] + "\n... [truncated]"
                    configs[filename] = content
                except Exception:
                    pass
        return configs

    def build_system_awareness(self) -> dict[str, Any]:
        """Collect host OS, runtime metadata, directory tree, and configs."""
        tree = self.scan_directory_tree()
        configs = self.scan_project_configs()

        env_summary = {
            "os_platform": sys.platform,
            "os_name": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "cwd": str(self.workspace_root),
            "configs_detected": list(configs.keys()),
            "tree_line_count": len(tree),
        }

        # Format context prompt injection
        context_lines = [
            f"=== SYSTEM AWARENESS ===",
            f"OS: {platform.system()} {platform.release()} ({sys.platform})",
            f"Python Runtime: {platform.python_version()}",
            f"Workspace Root: {self.workspace_root}",
            f"Detected Configs: {', '.join(configs.keys()) or 'None'}",
            "",
            "=== WORKSPACE FILE TREE ===",
            "\n".join(tree[:60]),
        ]

        if configs:
            context_lines.append("\n=== CONFIG HIGHLIGHTS ===")
            for cfg_name, cfg_content in configs.items():
                context_lines.append(f"--- {cfg_name} ---\n{cfg_content[:500]}\n")

        prompt_context = "\n".join(context_lines)

        return {
            "summary": env_summary,
            "tree": tree,
            "configs": configs,
            "prompt_context": prompt_context,
        }
