"""Workspace context gathering (Phase 2: bounded scan; Phase 3: AST repo map
and token budgeter replace the internals, keeping this surface)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

WORKSPACE_FILE_SUFFIXES = {".py", ".html", ".css", ".js", ".json", ".md", ".txt"}
MAX_WORKSPACE_FILES = 60
MAX_WORKSPACE_CONTEXT_CHARS = 12_000
MAX_FILE_SNIPPET_CHARS = 3_000


def list_workspace_files(workspace: Path) -> list[Path]:
    files: list[Path] = []
    for path in workspace.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in WORKSPACE_FILE_SUFFIXES
            and not any(
                part.startswith(".") for part in path.relative_to(workspace).parts
            )
        ):
            files.append(path)
    return sorted(files)


def build_workspace_context(
    workspace: Path,
    request_text: str,
    report: Callable[[str], None] | None = None,
) -> str:
    """Bounded relevant snippets for the planner, never the coder."""
    files = list_workspace_files(workspace)
    if not files:
        return "(workspace is empty)"
    request_lower = request_text.lower()
    ranked = sorted(
        files,
        key=lambda path: (
            0 if path.name.lower() in request_lower else 1,
            0 if path.suffix.lower() in {".py", ".html", ".js", ".css"} else 1,
            str(path),
        ),
    )
    sections: list[str] = [f"FILES ({len(files)} found):"]
    remaining = MAX_WORKSPACE_CONTEXT_CHARS
    for path in ranked[:MAX_WORKSPACE_FILES]:
        relative = path.relative_to(workspace)
        sections.append(f"- {relative}")
        if remaining <= 0:
            break
        try:
            if report:
                report(f"tool • filesystem.read_snippet: {path}")
            snippet = path.read_text(encoding="utf-8", errors="replace")[
                :MAX_FILE_SNIPPET_CHARS
            ]
        except OSError:
            continue
        if snippet:
            section = f"\n[{relative}]\n{snippet}"
            sections.append(section[:remaining])
            remaining -= len(section)
    return "\n".join(sections)[:MAX_WORKSPACE_CONTEXT_CHARS]
