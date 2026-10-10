"""Textual UI tests using the pilot harness (offline, no services)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("textual")

from tui_textual import OllamaAgentApp  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_message_appears_once_in_log() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    async with app.run_test() as pilot:
        # Stub the pipeline call so no network happens.
        async def fake_dispatch(text: str) -> None:
            return None

        app._dispatch = fake_dispatch  # type: ignore[assignment]
        chat = app.query_one("#chat")
        input_bar = app.query_one("#input-bar")
        input_bar.value = "hello world"
        await pilot.press("enter")
        await pilot.pause()

        # Strip Rich styling: join plain text of every segment on every line.
        plain_lines = [
            "".join(seg.text for seg in strip._segments) if hasattr(strip, "_segments") else str(strip)
            for strip in chat.lines
        ]
        log_text = "\n".join(plain_lines)
        assert log_text.count("hello world") == 1  # exactly once, not echoed twice
        assert any("you" in line and "hello world" in line for line in plain_lines)


async def test_sidebar_renders_gpu_na_when_nvidia_smi_missing() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    with patch("tui.subprocess.check_output", side_effect=FileNotFoundError):
        async with app.run_test() as pilot:
            app.query_one("#sidebar").refresh_vitals()
            await pilot.pause()
            system = app.query_one("#sb-system")
            rendered = str(system.render())
            assert "GPU: n/a" in rendered


async def test_sidebar_shows_workspace_and_logo() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    async with app.run_test() as pilot:
        await pilot.pause()
        logo = str(app.query_one("#sb-logo").render())
        assert "OLLAMA AGENT" in logo
        workspace = str(app.query_one("#sb-workspace").render())
        assert "coding_agent" in workspace


async def test_last_run_shows_skipped_in_red() -> None:
    import tui

    tui._last_trace = [
        {"stage": "ROUTE", "status": "ok", "seconds": 0.1},
        {"stage": "ANSWER", "status": "ok", "seconds": 1.0},
    ]
    app = OllamaAgentApp(workspace="E:/coding_agent")
    async with app.run_test() as pilot:
        app.query_one("#sidebar").refresh_vitals()
        await pilot.pause()
        rendered = str(app.query_one("#sb-lastrun").render())
        assert "PLAN SKIPPED" in rendered
        assert "ROUTE" in rendered
    tui._last_trace = []
