"""Textual UI tests (pilot harness, offline, no services)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("textual")

import tui  # noqa: E402
from tui_textual import OllamaAgentApp, collect_system_lines, trace_line  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _plain_lines(app) -> list[str]:
    chat = app.query_one("#chat")
    lines: list[str] = []
    for strip in chat.lines:
        segments = getattr(strip, "_segments", None)
        if segments is not None:
            lines.append("".join(seg.text for seg in segments))
        else:
            lines.append(str(strip))
    return lines


async def test_message_appears_once_in_log() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    async with app.run_test() as pilot:
        async def fake_dispatch(text: str) -> None:
            return None

        # Stub the pipeline worker so no network happens.
        app._pipeline_work = lambda text: None  # type: ignore[assignment]
        chat = app.query_one("#chat")
        app.query_one("#input-bar").value = "hello world"
        await pilot.press("enter")
        await pilot.pause()
        plain = _plain_lines(app)
        joined = "\n".join(plain)
        assert joined.count("hello world") == 1  # exactly once
        assert any("you" in line and "hello world" in line for line in plain)


async def test_log_scrolls_beyond_viewport_and_follows_at_bottom() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    async with app.run_test() as pilot:
        chat = app.query_one("#chat")
        for i in range(120):
            app._write(f"line {i}")
        await pilot.pause()
        # Was at bottom -> auto-scroll followed the new output.
        assert chat.scroll_y > 0
        # Scroll back up: new output must NOT yank the view back down.
        chat.scroll_home(animate=False)
        await pilot.pause()
        app._write("new line while scrolled up")
        await pilot.pause()
        assert chat.scroll_y == 0


async def test_vitals_change_with_nvidia_smi_output() -> None:
    first = "1000, 4094, 12\n"
    second = "2500, 4094, 87\n"
    with patch("tui.subprocess.check_output", return_value=first):
        lines_a = collect_system_lines()
    with patch("tui.subprocess.check_output", return_value=second):
        lines_b = collect_system_lines()
    assert lines_a[0] != lines_b[0]
    assert "1000/4094" in lines_a[0]
    assert "2500/4094" in lines_b[0]


async def test_sidebar_gpu_na_when_nvidia_smi_missing() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    with patch("tui.subprocess.check_output", side_effect=FileNotFoundError):
        async with app.run_test() as pilot:
            app._store_vitals(collect_system_lines())
            app._tick()
            await pilot.pause()
            rendered = str(app.query_one("#sb-system").render())
            assert "GPU: n/a" in rendered


async def test_sidebar_wordmark_last_and_no_old_logo() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    async with app.run_test() as pilot:
        await pilot.pause()
        sidebar = app.query_one("#sidebar")
        last = list(sidebar.children)[-1]
        text = last.render()
        plain = text.plain if hasattr(text, "plain") else str(text)
        assert plain == "<_FOUR_PIE"
        # "OLLAMA AGENT" appears nowhere in the sidebar.
        whole = " ".join(
            str(child.render()) for child in sidebar.children
        )
        assert "OLLAMA AGENT" not in whole


async def test_question_route_prints_no_red_text() -> None:
    app = OllamaAgentApp(workspace="E:/coding_agent")
    async with app.run_test() as pilot:
        app._pipeline_done({
            "answer": "Rust is compiled; Python is interpreted.",
            "trace": [
                {"stage": "ROUTE", "status": "ok", "seconds": 0.0, "note": "QUESTION"},
                {"stage": "ANSWER", "status": "ok", "seconds": 1.2, "note": ""},
            ],
        })
        await pilot.pause()
        chat = app.query_one("#chat")
        for strip in chat.lines:
            for seg in getattr(strip, "_segments", []):
                style = str(seg.style or "")
                assert "red" not in style.lower(), f"red style in question route: {style}"
        # and no SKIPPED anywhere for a question route
        assert "SKIPPED" not in "\n".join(_plain_lines(app))


def test_trace_line_code_task_marks_missing_stages_skipped() -> None:
    trace = [{"stage": "ROUTE", "status": "ok", "seconds": 0.0, "note": "CODE_TASK"}]
    line = trace_line(trace)
    assert "PLAN SKIPPED" in line
    assert "ROUTE ok 0.0s" in line


def test_trace_line_question_has_no_skipped() -> None:
    trace = [
        {"stage": "ROUTE", "status": "ok", "seconds": 0.0, "note": "QUESTION"},
        {"stage": "ANSWER", "status": "ok", "seconds": 0.7},
    ]
    line = trace_line(trace)
    assert "SKIPPED" not in line
    assert "ANSWER ok 0.7s" in line
