"""Phase C tests: markdown rendering, escaping, previews (offline, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

import render
from render import (
    file_preview,
    preview_command,
    render_markdown,
    resolve_preview_target,
)


def _record(renderable, width: int = 100) -> str:
    console = Console(record=True, width=width, force_terminal=False)
    console.print(renderable)
    return console.export_text()


# --- render_markdown -------------------------------------------------------------

def test_render_markdown_returns_renderable() -> None:
    out = render_markdown("# Hi\n\nSome **bold** text.")
    assert isinstance(out, Markdown)


def test_render_markdown_falls_back_to_text_on_error(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("markdown-it exploded")
    monkeypatch.setattr(render, "Markdown", boom)
    out = render_markdown("plain **text**")
    assert isinstance(out, Text)
    assert "plain **text**" in out.plain


def test_answer_with_brackets_renders_literally() -> None:
    """[red], [/] and [y/N] in an answer must render literally, never parse."""
    text = "Use [red] and [/] carefully; the prompt shows [y/N]."
    out = _record(render_markdown(text))
    assert "[red]" in out and "[/]" in out and "[y/N]" in out


def test_table_and_fenced_code_render() -> None:
    text = (
        "| name | size |\n|---|---|\n| `a.py` | 1 KB |\n\n"
        "```python\ndef greet(name):\n    return f'hi {name}'\n```\n"
    )
    out = _record(render_markdown(text))
    assert "a.py" in out            # table content rendered
    assert "def greet" in out       # fenced code content rendered
    assert "name" in out and "size" in out


def test_wide_table_wraps_within_console_width() -> None:
    row = "| " + " | ".join(f"column-{i} content-{i}" for i in range(8)) + " |"
    text = "| " + " | ".join(f"column-{i}" for i in range(8)) + " |\n" \
        + "|" + "---|" * 8 + "\n" + row + "\n"
    out = _record(render_markdown(text), width=60)
    for line in out.splitlines():
        assert len(line) <= 60, line


def test_long_answer_capped_with_note() -> None:
    text = "\n".join(f"line {i}" for i in range(2_000))
    out = _record(render_markdown(text))
    assert "(+1600 lines" in out and "/export for the full text" in out
    assert "line 1999" not in out  # capped at 400


# --- escaping (the Textual markup-string path) --------------------------------------

def test_escape_for_markup_strings() -> None:
    from rich.markup import escape

    assert escape("a [red] b") == "a \\[red] b"
    assert escape("[/]") == "\\[/]"
    assert escape("[y/N]") == "\\[y/N]"


# --- summary skeleton is markdown -----------------------------------------------------

def test_summary_skeleton_markdown() -> None:
    from orchestrator import _summary_skeleton

    text = _summary_skeleton(
        "make readme.md",
        {"readme.md": "# x\n"},
        [],
        [],
        validate_failed=False,
        total_seconds=1.5,
        validate_ran=True,
    )
    assert "**Built:**" in text
    assert "- `readme.md`" in text
    assert "**How to run:**" in text
    assert "**Known issues:** none" in text
    assert "**Total time:** 1.5s" in text


def test_summary_skeleton_unverified_blockquote() -> None:
    from orchestrator import _summary_skeleton

    text = _summary_skeleton(
        "make readme.md",
        {"readme.md": "# x\n"},
        [],
        [],
        validate_failed=True,
        total_seconds=2.0,
        validate_error="model not installed: 'x:1b'. Run: ollama pull x:1b",
    )
    assert text.startswith("> **UNVERIFIED**")
    assert "Fix: ollama pull x:1b" in text
    assert "**Known issues:** validation could not run" in text


# --- file preview ------------------------------------------------------------

def test_preview_md_renders_markdown_by_default() -> None:
    with TemporaryDirectory() as tmp:
        Path(tmp, "readme.md").write_text(
            "# Story\n\nOnce upon a **time**.\n", encoding="utf-8")
        out = _record(preview_command(tmp, "readme.md"))
        assert "Story" in out and "time" in out
        assert "preview: readme.md" in out
        assert "# Story" not in out  # rendered, not raw


def test_preview_md_raw_shows_source() -> None:
    with TemporaryDirectory() as tmp:
        Path(tmp, "readme.md").write_text(
            "# Story\n\nOnce upon a **time**.\n", encoding="utf-8")
        out = _record(preview_command(tmp, "readme.md raw"))
        assert "# Story" in out  # raw source
        assert "raw" in out


def test_preview_code_file_is_highlighted() -> None:
    with TemporaryDirectory() as tmp:
        Path(tmp, "app.py").write_text("def add(a, b):\n    return a + b\n",
                                       encoding="utf-8")
        out = _record(preview_command(tmp, "app.py"))
        assert "def add(a, b):" in out


def test_preview_missing_file_clean_error() -> None:
    with TemporaryDirectory() as tmp:
        out = preview_command(tmp, "missing.md")
        text = _record(out)
        assert "no such file: missing.md" in text


def test_preview_path_escape_rejected() -> None:
    with TemporaryDirectory() as tmp:
        out = preview_command(tmp, "../evil.md")
        text = _record(out)
        assert "path not allowed" in text


def test_preview_binary_and_large_notes() -> None:
    with TemporaryDirectory() as tmp:
        Path(tmp, "data.bin").write_bytes(b"\x00\x01\xff" * 4)
        out = _record(preview_command(tmp, "data.bin"))
        assert "binary" in out
        big = Path(tmp, "big.txt")
        big.write_text("x" * (600 * 1024), encoding="utf-8")
        out = _record(preview_command(tmp, "big.txt"))
        assert "too large" in out


def test_preview_prefers_staged_over_workspace() -> None:
    with TemporaryDirectory() as tmp:
        Path(tmp, "readme.md").write_text("# workspace version\n", encoding="utf-8")
        from workspace_staging import WorkspaceStagingManager

        WorkspaceStagingManager(tmp).stage_file("readme.md", "# staged version\n")
        out = _record(preview_command(tmp, "readme.md"))
        assert "staged version" in out
        assert "(staged)" in out


def test_preview_no_args_uses_latest_staged() -> None:
    with TemporaryDirectory() as tmp:
        from workspace_staging import WorkspaceStagingManager

        mgr = WorkspaceStagingManager(tmp)
        mgr.stage_file("a.py", "X = 1\n")
        mgr.stage_file("b.md", "# Bee\n")
        out = _record(preview_command(tmp, ""))
        assert "b.md" in out  # most recently staged


def test_preview_no_args_nothing_staged() -> None:
    with TemporaryDirectory() as tmp:
        out = _record(preview_command(tmp, ""))
        assert "Nothing staged" in out


# --- confirm-gate preview -----------------------------------------------------

def test_confirm_gate_output_includes_staged_readme_preview() -> None:
    """The gate display path (as the TUIs render it) shows the staged
    README rendered (markdown), plus the overwrite marker."""
    with TemporaryDirectory() as tmp:
        Path(tmp, "readme.md").write_text("# old\n", encoding="utf-8")
        from workspace_staging import WorkspaceStagingManager

        WorkspaceStagingManager(tmp).stage_file(
            "readme.md", "# Toy Story\n\nA story of toys.\n")
        pending = {
            "intent": "code: plan is create readme.md (overwrites existing file)",
            "plan": [{"path": "readme.md", "action": "create", "overwrites": True}],
        }
        console = Console(record=True, width=100, force_terminal=False)
        from rich.markup import escape as _esc

        console.print(f"Plan ({_esc(pending['intent'])}):")
        for t in pending["plan"]:
            ow = "  (overwrites existing file)" if t.get("overwrites") else ""
            console.print(f"  {_esc(t['action'])}  {_esc(t['path'])}{ow}")
            target, source = resolve_preview_target(tmp, t["path"])
            console.print(f"preview: {t['path']}  ({source})")
            console.print(file_preview(target))
        out = console.export_text()
        assert "Toy Story" in out            # the STAGED readme, rendered
        assert "A story of toys" in out
        assert "(overwrites existing file)" in out
        assert "(staged)" in out
        assert "# Toy Story" not in out      # rendered, not raw source


# --- export keeps raw markdown with a valid fence --------------------------------

def test_export_keeps_raw_markdown_with_valid_fence() -> None:
    import export as export_mod
    import history as history_mod

    with TemporaryDirectory() as tmp:
        answer_with_fence = "Here is code:\n\n```python\nX = 1\n```\n\nDone."
        history_mod.append_turn(tmp, {
            "text": "make app.py", "path": "code", "outcome": "ok",
            "label": "code", "answer_preview": answer_with_fence,
        })
        result = export_mod.do_export(tmp, "last")
        text = Path(result["path"]).read_text(encoding="utf-8")
        # raw markdown preserved verbatim
        assert "```python\nX = 1\n```" in text
        # the outer fence is longer than the inner one (valid nesting)
        assert "````" in text
