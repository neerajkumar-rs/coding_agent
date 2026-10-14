"""Shared rendering helpers for both TUIs: markdown answers and file previews.

All markdown/preview logic lives HERE — tui.py and tui_textual.py call these
functions; nothing is duplicated between the two UIs. rich, markdown-it-py and
Pygments are already requirements; no new dependency.

History and the debug export always keep the RAW markdown text; only the
on-screen log renders it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text

MD_CODE_THEME = "monokai"
ANSWER_MAX_LINES = 400   # a longer answer is capped with a /export note
PREVIEW_MAX_LINES = 40   # previews show the first 40 lines
PREVIEW_LARGE_BYTES = 512 * 1024  # over this: a one-line note, not a dump


def render_markdown(text: str, *, max_lines: int = ANSWER_MAX_LINES) -> Any:
    """Markdown renderable for an answer/summary (monokai code theme,
    hyperlinks on). Falls back to plain ``Text`` when rendering raises. Long
    texts are capped at ``max_lines`` with a note pointing at /export."""
    try:
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(lines[:max_lines]) + (
                f"\n\n*(+{len(lines) - max_lines} lines — /export for the full text)*"
            )
        return Markdown(text, code_theme=MD_CODE_THEME, hyperlinks=True)
    except Exception:
        return Text(text)


def print_markdown(console: Any, text: str, *, max_lines: int = ANSWER_MAX_LINES) -> None:
    """Rich-console path: markdown when attached to a terminal, plain print
    when stdout is not a terminal."""
    if getattr(console, "is_terminal", False):
        console.print(render_markdown(text, max_lines=max_lines))
    else:
        print(text)


def _language_for(path: Path) -> str:
    """Display language from the single file-type registry."""
    from orchestrator import LANGUAGE_BY_SUFFIX

    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def file_preview(
    path: Path, *, raw: bool = False, max_lines: int = PREVIEW_MAX_LINES
) -> Any:
    """Render one file for the log: .md as markdown (raw text with raw=True),
    other text files as a syntax-highlighted code block, binaries and files
    over 512 KB as a one-line note. Long files are capped at max_lines."""
    path = Path(path)
    name = path.name
    try:
        size = path.stat().st_size
    except OSError:
        return Text(f"{name}: not readable", style="red")
    if size > PREVIEW_LARGE_BYTES:
        return Text(f"{name}: {size // 1024} KB — too large to preview", style="dim")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Text(f"{name}: binary file — no preview", style="dim")
    except OSError:
        return Text(f"{name}: not readable", style="red")

    lines = text.splitlines()
    truncated = ""
    if len(lines) > max_lines:
        truncated = f"(+{len(lines) - max_lines} lines)"
        text = "\n".join(lines[:max_lines])

    is_md = path.suffix.lower() in (".md", ".markdown")
    if is_md and not raw:
        body: Any = render_markdown(text, max_lines=max_lines)
        # render_markdown already appends its own cap note when over max_lines
        truncated = ""
    else:
        body = Syntax(
            text, _language_for(path), theme=MD_CODE_THEME, word_wrap=True
        )
    if truncated:
        return Group(body, Text(truncated, style="dim"))
    return body


def resolve_preview_target(workspace: str | Path, rel: str) -> tuple[Path, str]:
    """The staged version of ``rel`` when one exists, else the workspace file.

    The path is validated (containment + protection) via safe_read_target
    BEFORE anything is read; PathEscapeError propagates to the caller.
    """
    from agent_core.staging import safe_read_target
    from workspace_staging import WorkspaceStagingManager

    ws = Path(workspace).resolve()
    safe_read_target(ws, rel)  # validation only; raises on escape/protected
    staged = WorkspaceStagingManager(ws).staging_dir / rel
    if staged.is_file():
        return staged, "staged"
    return (ws / rel).resolve(), "workspace"


def _latest_staged(workspace: str | Path) -> str | None:
    """The most recently staged file (mtime), or None."""
    from workspace_staging import WorkspaceStagingManager

    ws = Path(workspace).resolve()
    files = WorkspaceStagingManager(ws).list_staged_files()
    if not files:
        return None
    staging = WorkspaceStagingManager(ws).staging_dir
    try:
        return max(files, key=lambda f: (staging / f).stat().st_mtime)
    except OSError:
        return files[-1]


def preview_command(workspace: str | Path, args: str) -> Any:
    """`/preview <file> [raw]` — one renderable for the log.

    No argument: the most recently staged file. Staged version wins over the
    workspace file. Errors are one-line Text renderables; path escapes are
    rejected. Never a model call, never the GPU, never the network.
    """
    from agent_core.staging import PathEscapeError

    parts = args.split()
    raw = "raw" in parts
    names = [p for p in parts if p != "raw"]
    rel = names[0] if names else ""
    if not rel:
        rel = _latest_staged(workspace) or ""
        if not rel:
            return Text("Nothing staged — usage: /preview <file> [raw]", style="dim")
    try:
        target, source = resolve_preview_target(workspace, rel)
    except PathEscapeError:
        return Text(f"preview: path not allowed: {rel}", style="red")
    if not target.is_file():
        return Text(
            f"preview: no such file: {rel} (not staged, not in the workspace)",
            style="red",
        )
    header = Text(
        f"preview: {rel}  ({source}{', raw' if raw else ''})", style="dim"
    )
    return Group(header, file_preview(target, raw=raw))


__all__ = [
    "escape",
    "file_preview",
    "preview_command",
    "print_markdown",
    "render_markdown",
    "resolve_preview_target",
]
