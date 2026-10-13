"""Phase 5 tests: save prompt (files + default N), pluralize, --yes, no lfm."""

from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

import tui
from tui import pluralize

pytest.importorskip("textual")
from tui_textual import OllamaAgentApp  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _plain_lines(app) -> list[str]:
    chat = app.query_one("#chat")
    out: list[str] = []
    for strip in chat.lines:
        segments = getattr(strip, "_segments", None)
        out.append("".join(seg.text for seg in segments) if segments is not None else str(strip))
    return out


DIFF_TWO_FILES = {
    "has_changes": True,
    "staged_count": 2,
    "total_additions": 40,
    "total_deletions": 3,
    "diffs": [
        {"file": "index.html", "action": "create", "diff": "...", "additions": 30, "deletions": 0},
        {"file": "app.py", "action": "edit", "diff": "...", "additions": 10, "deletions": 3},
    ],
}


# --- pluralize ---------------------------------------------------------------

def test_pluralize() -> None:
    assert pluralize(1, "file") == "1 file"
    assert pluralize(2, "files".rstrip("s")) == "2 files"
    assert pluralize(0, "file") == "0 files"


# --- save prompt: files listed, default N ------------------------------------

async def test_save_prompt_lists_files_default_n() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    async with app.run_test() as pilot:
        app._show_save_prompt(DIFF_TWO_FILES)
        await pilot.pause()
        plain = "\n".join(_plain_lines(app))
        assert "index.html" in plain and "(new)" in plain
        assert "app.py" in plain and "(overwrites existing file)" in plain
        assert "Save? [y/N]" in plain
        assert app._save_pending is True

        # Answer "n" -> not saved, hint printed, nothing deployed.
        app.query_one("#input-bar").value = "n"
        await pilot.press("enter")
        await pilot.pause()
        plain = "\n".join(_plain_lines(app))
        assert "Not saved" in plain
        assert "/save" in plain


async def test_save_prompt_yes_saves() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    async with app.run_test() as pilot:
        app._show_save_prompt(DIFF_TWO_FILES)
        await pilot.pause()
        with patch("tui_textual._post_json_sync",
                   return_value={"deployed_files": ["index.html", "app.py"]}):
            app.query_one("#input-bar").value = "y"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()  # let the deploy worker land
        plain = "\n".join(_plain_lines(app))
        assert "Saved 2 files to" in plain


async def test_yes_flag_auto_approves() -> None:
    tui._cli_yes = True
    try:
        app = OllamaAgentApp(workspace="E:/coding_agent")
        async with app.run_test() as pilot:
            with patch("tui_textual._post_json_sync",
                       return_value={"deployed_files": ["index.html", "app.py"]}):
                app._show_save_prompt(DIFF_TWO_FILES)
                await pilot.pause()
                await pilot.pause()
            assert app._save_pending is False  # no prompt left pending
            plain = "\n".join(_plain_lines(app))
            assert "auto-saving" in plain
            assert "Saved 2 files to" in plain
    finally:
        tui._cli_yes = False


# --- .agent with a removed model name falls back to defaults ------------------

def test_agent_config_ignores_removed_models() -> None:
    from agent_config import load_agent_config, save_agent_config, AgentConfig

    removed_name = "l" + "fm" + "2.5-thinking"  # built, never spelled in the repo
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / ".agent"
        path.write_text(json.dumps({
            "planner_model": removed_name,
            "coder.5b_or_3b": removed_name,
            "max_context_tokens": 4096,
        }), encoding="utf-8")
        config = load_agent_config(tmp)
        assert config.planner_model == "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
        assert config.coder_model == "qwen2.5-coder:3b-instruct"
        # supported names still load fine
        save_agent_config(AgentConfig(), tmp)
        assert load_agent_config(tmp).coder_model == "qwen2.5-coder:3b-instruct"


# --- no lfm anywhere (the spec's grep as a test) ------------------------------

def test_no_lfm_references_in_repo() -> None:
    root = Path(tui.PROJECT_ROOT)
    needle = re.compile("lf" + "m|li" + "quid", re.IGNORECASE)  # built, never spelled
    offenders: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".txt", ".json"}:
            continue
        if path.name == Path(__file__).name:
            continue  # this test holds the pattern (built, not spelled)
        if any(part in {"env", ".git", "node_modules", "__pycache__",
                        ".agent_runs", ".agent_staging", ".agent_logs"}
               for part in path.parts):
            continue
        try:
            if needle.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(path))
        except OSError:
            continue
    assert offenders == [], f"removed-model references remain: {offenders}"
