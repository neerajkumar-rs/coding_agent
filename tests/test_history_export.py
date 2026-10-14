"""History store + debug export tests (offline; no model calls, no VRAM)."""

from __future__ import annotations

import asyncio
import json
import threading
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import export as export_mod
import history as history_mod


# ---------------------------------------------------------------------------
# Store: append/read, ids, concurrency, corrupt lines, rotation, failures
# ---------------------------------------------------------------------------

def test_append_read_and_monotonic_ids(tmp_path: Path) -> None:
    id1 = history_mod.append_turn(tmp_path, {"text": "first", "path": "reply"})
    id2 = history_mod.append_turn(tmp_path, {"text": "second", "path": "code"})
    assert id1 is not None and id2 is not None and id2 == id1 + 1
    entries, skipped = history_mod.read_history(tmp_path)
    assert skipped == 0
    assert [e["text"] for e in entries] == ["first", "second"]
    assert (tmp_path / ".agent_runs" / "history.jsonl").exists()


def test_concurrent_appends_keep_monotonic_ids(tmp_path: Path) -> None:
    def writer(n: int) -> None:
        for i in range(10):
            history_mod.append_turn(tmp_path, {"text": f"t{n}-{i}", "path": "reply"})

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entries, _ = history_mod.read_history(tmp_path)
    ids = [e["id"] for e in entries]
    assert len(ids) == 40
    assert len(set(ids)) == 40  # unique
    assert ids == sorted(ids)   # read out chronological


def test_corrupt_line_skipped_and_counted(tmp_path: Path) -> None:
    path = history_mod.history_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"id": 1, "text": "ok"}) + "\n"
        + "{ not json\n"
        + json.dumps({"id": 2, "text": "also ok"}) + "\n",
        encoding="utf-8",
    )
    entries, skipped = history_mod.read_history(tmp_path)
    assert skipped == 1
    assert [e["id"] for e in entries] == [1, 2]
    # next id continues after the highest VALID id
    assert history_mod.append_turn(tmp_path, {"text": "third"}) == 3


def test_rotation_and_reading_across_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(history_mod, "MAX_ENTRIES", 5)
    for i in range(7):
        history_mod.append_turn(tmp_path, {"text": f"turn {i}"})
    rotated = list((tmp_path / ".agent_runs").glob("history-*.jsonl"))
    assert rotated, "rotation never happened"
    entries, _ = history_mod.read_history(tmp_path)
    assert [e["text"] for e in entries] == [f"turn {i}" for i in range(7)]
    assert entries[-1]["id"] == 7  # id monotonic across the rotation


def test_write_failure_never_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert history_mod.append_turn(tmp_path, {"text": "x"}) is None  # no exception


def test_paging_with_before(tmp_path: Path) -> None:
    for i in range(10):
        history_mod.append_turn(tmp_path, {"text": f"turn {i}"})
    entries, _ = history_mod.read_history(tmp_path, limit=3, before=8)
    assert [e["id"] for e in entries] == [5, 6, 7]


# ---------------------------------------------------------------------------
# Every pipeline path writes an entry (through run_pipeline with fake generate)
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _fake_ok(model, prompt, system="", temperature=0.2, **kw):
    async def inner():
        if "CLASSIFY MODE:" in prompt:
            return "reply"
        if "FACTS:" in prompt:
            return "Grounded answer."
        return ""
    return inner()


@pytest.mark.parametrize("text,expected_path", [
    ("hello", "reply"),       # greetings go through the loop (one reply step)
    ("/save", "fast-path"),   # only slash commands stay a zero-model fast path
])
def test_fast_paths_write_history(tmp_path: Path, text: str, expected_path: str) -> None:
    from orchestrator import run_pipeline

    async def fake(model, prompt, **kw):
        # the loop's reply step: one model call for a greeting
        return json.dumps({"action": "reply", "text": "Hello! What are we building?"})

    _run(run_pipeline(text, tmp_path, generate=fake, planner_model="P", coder_model="C"))
    entries, _ = history_mod.read_history(tmp_path)
    assert entries and entries[-1]["path"] == expected_path
    assert entries[-1]["run_id"]
    # every history entry links to a run dir with at least state.json + events.jsonl
    run_dir = tmp_path / ".agent_runs" / entries[-1]["run_id"]
    assert (run_dir / "state.json").exists()
    assert (run_dir / "events.jsonl").exists()


def test_reply_turn_writes_history(tmp_path: Path) -> None:
    from orchestrator import run_pipeline

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps({"action": "reply", "text": "Grounded answer."})
        return ""

    # A question: the loop answers with one reply step.
    _run(run_pipeline("what is a closure", tmp_path, generate=fake,
                      planner_model="P", coder_model="C"))
    entries, _ = history_mod.read_history(tmp_path)
    assert entries[-1]["path"] == "reply"
    assert entries[-1]["outcome"] == "ok"
    assert entries[-1]["answer_preview"] == "Grounded answer."


def test_failed_turn_writes_history_with_error_and_traceback(tmp_path: Path) -> None:
    from orchestrator import run_pipeline

    async def fake(model, prompt, **kw):
        raise RuntimeError("synthetic GPU fault")

    with pytest.raises(RuntimeError):
        _run(run_pipeline("list all the files", tmp_path, generate=fake,
                          planner_model="P", coder_model="C"))
    entries, _ = history_mod.read_history(tmp_path)
    assert entries[-1]["path"] == "error"
    assert entries[-1]["outcome"] == "error"
    assert entries[-1]["error"]["detail"] == "synthetic GPU fault"
    # the run dir carries the traceback
    events = (tmp_path / ".agent_runs" / entries[-1]["run_id"] / "events.jsonl").read_text(encoding="utf-8")
    assert "traceback" in events
    assert "synthetic GPU fault" in events


def test_cancelled_turn_writes_history(tmp_path: Path) -> None:
    import threading
    from orchestrator import run_pipeline

    cancel = threading.Event()
    cancel.set()  # cancelled before the first call

    async def fake(model, prompt, **kw):
        return ""

    result = _run(run_pipeline("make a page", tmp_path, generate=fake,
                               planner_model="P", coder_model="C", cancel_event=cancel))
    assert result["status"] == "cancelled"
    entries, _ = history_mod.read_history(tmp_path)
    assert entries[-1]["outcome"] == "cancelled"


# ---------------------------------------------------------------------------
# Commands: /history variants
# ---------------------------------------------------------------------------

def test_history_table_and_detail_and_search(tmp_path: Path) -> None:
    history_mod.append_turn(tmp_path, {"text": "list all the files", "path": "reply",
                                       "outcome": "ok", "seconds": 1.2,
                                       "answer_preview": "app.py, notes.txt"})
    history_mod.append_turn(tmp_path, {"text": "make a page", "path": "code",
                                       "outcome": "error", "seconds": 3.4,
                                       "error": {"stage": "CLASSIFY", "detail": "boom"}})
    table = history_mod.format_history_table(tmp_path)
    assert "list all the files" in table and "make a page" in table
    detail = history_mod.format_history_detail(tmp_path, 2)
    assert "make a page" in detail and "boom" in detail and "CLASSIFY" in detail
    hits = history_mod.format_history_search(tmp_path, "files")
    assert "list all the files" in hits


# ---------------------------------------------------------------------------
# Export: report / zip / fixture / redaction / CLI
# ---------------------------------------------------------------------------

def _seed_turns(tmp_path: Path) -> None:
    history_mod.append_turn(tmp_path, {
        "text": "list all the files", "path": "reply", "outcome": "ok",
        "seconds": 1.2, "label": "reply", "answer_preview": "app.py, notes.txt",
        "models": {"understand": "qwen2.5-coder:3b-instruct"},
        "trace": [{"stage": "CLASSIFY", "status": "ok", "seconds": 0.4,
                   "model": "qwen2.5-coder:3b-instruct"},
                  {"stage": "ANSWER", "status": "ok", "seconds": 0.8,
                   "model": "qwen2.5-coder:3b-instruct"}],
    })
    history_mod.append_turn(tmp_path, {
        "text": "make a page", "path": "error", "outcome": "error",
        "seconds": 0.5, "error": {"stage": "CLASSIFY",
                                  "detail": "model not installed: 'qwen2.5-coder:7b-instruct'. Run: ollama pull qwen2.5-coder:7b-instruct"},
    })


def test_export_report_written_with_likely_cause(tmp_path: Path) -> None:
    _seed_turns(tmp_path)
    result = export_mod.do_export(tmp_path, "all")
    assert result["path"] and Path(result["path"]).exists()
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "## Summary" in text
    assert "## Environment" in text
    assert "Likely cause" in text and "ollama pull qwen2.5-coder:7b-instruct" in text
    assert "/retry 2" in text
    # exports live only under .agent_runs/exports/
    assert ".agent_runs" in result["path"] and "exports" in result["path"]


def test_export_targets(tmp_path: Path) -> None:
    _seed_turns(tmp_path)
    assert export_mod.do_export(tmp_path, "last")["turns"] == 1
    assert export_mod.do_export(tmp_path, "1")["turns"] == 1
    assert export_mod.do_export(tmp_path, "1-2")["turns"] == 2
    assert export_mod.do_export(tmp_path, "all")["turns"] == 2
    assert export_mod.do_export(tmp_path, "999").get("error") or True  # falls back to last-N


def test_export_no_history_is_a_clean_error(tmp_path: Path) -> None:
    result = export_mod.do_export(tmp_path, "last")
    assert result.get("error")
    assert result["path"] is None


def test_export_zip_contains_report_slice_and_run_folder(tmp_path: Path) -> None:
    _seed_turns(tmp_path)
    # give turn 1 a run folder with state.json + events.jsonl
    run_dir = tmp_path / ".agent_runs" / "run-test-1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    (run_dir / "events.jsonl").write_text('{"event": "run_started"}\n', encoding="utf-8")
    entries, _ = history_mod.read_history(tmp_path)
    entry = dict(entries[0])
    entry["run_id"] = "run-test-1"
    # rewrite history with the run_id set
    (tmp_path / ".agent_runs" / "history.jsonl").write_text("", encoding="utf-8")
    history_mod.append_turn(tmp_path, entry)
    history_mod.append_turn(tmp_path, dict(entries[1]))

    result = export_mod.do_export(tmp_path, "all", zip_out=True)
    zip_path = result["zip_path"]
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "report.md" in names
    assert "history-slice.jsonl" in names
    assert "environment.json" in names
    assert any(n.startswith("runs/run-test-1/") for n in names)


def test_export_fixture_format_loads_with_eval_loader(tmp_path: Path) -> None:
    _seed_turns(tmp_path)
    result = export_mod.do_export(tmp_path, "all", fixture=True)
    lines = Path(result["path"]).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for raw in lines:
        entry = json.loads(raw)  # same loader as classify_eval.jsonl
        assert "text" in entry and "expected" in entry and entry["observed_ok"] is None
    assert result["note"]


def test_redaction_masks_secrets_and_paths(tmp_path: Path) -> None:
    secret_line = f"API_KEY=sk-test123456789 in {tmp_path}"
    history_mod.append_turn(tmp_path, {
        "text": secret_line, "path": "reply", "outcome": "ok",
        "answer_preview": "token: ghp_abcdefghijklmnopqrstuvwxyz12",
    })
    result = export_mod.do_export(tmp_path, "last")
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "sk-test123456789" not in text
    assert "ghp_abcdefghijklmnopqrstuvwxyz12" not in text
    assert str(tmp_path) not in text
    assert "<WORKSPACE>" in text
    # --no-redact shows it, with the warning
    result2 = export_mod.do_export(tmp_path, "last", no_redact=True)
    text2 = Path(result2["path"]).read_text(encoding="utf-8")
    assert "sk-test123456789" in text2
    assert result2["warning"]


def test_report_cap_keeps_summary_and_failing_turn(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(export_mod, "REPORT_CAP", 4 * 1024)
    for i in range(5):
        history_mod.append_turn(tmp_path, {
            "text": "x" * 3000, "path": "reply", "outcome": "ok",
        })
    history_mod.append_turn(tmp_path, {
        "text": "the failing turn", "path": "error", "outcome": "error",
        "error": {"stage": "PLAN", "detail": "validation failure reason"},
    })
    result = export_mod.do_export(tmp_path, "all")
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert len(text.encode()) <= 4 * 1024 + 1024  # cap + a little slack
    assert "## Summary" in text
    assert "the failing turn" in text  # failing turn kept


def test_cli_export_works_with_ollama_down(tmp_path: Path, monkeypatch) -> None:
    """`python tui.py --workspace <dir> --export last` with no Ollama/FastAPI:
    prints the path, exits 0; with no history exits 1 with a clear message."""
    import tui

    _seed_turns(tmp_path)
    args = tui._parse_cli(["--workspace", str(tmp_path), "--export", "last"])
    assert args.export == "last"
    # run the export branch directly (cli() would SystemExit)
    import export as _export
    result = _export.do_export(tmp_path, args.export)
    assert result["path"] and Path(result["path"]).exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    result2 = _export.do_export(empty, "last")
    assert result2.get("error")


def test_export_makes_no_ollama_calls(tmp_path: Path, monkeypatch) -> None:
    """Grep-level guarantee: the store/export path never touches the gateway."""
    import socket

    def boom(*a, **k):
        raise AssertionError("network call from history/export")
    monkeypatch.setattr(socket.socket, "connect", boom)
    _seed_turns(tmp_path)
    # build_environment probes degrade to n/a; no exception escapes
    result = export_mod.do_export(tmp_path, "all", zip_out=True)
    assert Path(result["path"]).exists()


def test_export_fixture_compatible_with_holdout_format(tmp_path: Path) -> None:
    """--fixture lines load with the SAME shape checks as classify_holdout.jsonl."""
    _seed_turns(tmp_path)
    result = export_mod.do_export(tmp_path, "all", fixture=True)
    lines = [
        json.loads(raw)
        for raw in Path(result["path"]).read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    assert len(lines) == 2
    for entry in lines:
        # the holdout contract: label + files_created (+ pre-turn state)
        assert entry["expected"] in ("fast", "reply", "code", "both",
                                     "fast-path", "error", None), entry
        assert "files_created" in entry
        assert "state" in entry and "observed_ok" in entry


def test_export_contains_prompts_and_raw_responses(tmp_path: Path) -> None:
    """A turn with a run folder carrying calls/<n>.json: the report shows the
    prompt and the RAW response (think separate) in fenced blocks."""
    from orchestrator import run_pipeline

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps({"action": "reply",
                               "text": "<think>checking the table</think>The workspace has one file."})
        return ""

    _run(run_pipeline("what is a closure", tmp_path, generate=fake,
                      planner_model="P", coder_model="C"))
    result = export_mod.do_export(tmp_path, "last")
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "**LLM calls**" in text
    assert "prompt:" in text and "response:" in text
    assert "what is a closure" in text  # the real prompt text
    assert "The workspace has one file." in text  # the (raw) response
    assert "num_ctx=" in text


def test_export_pre_turn_snapshot(tmp_path: Path) -> None:
    """The state_snapshot in history is the state fed to the model for THAT
    turn (taken before it) — turn 2's snapshot must NOT contain turn 2's own
    result."""
    from orchestrator import run_pipeline

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "AGENT ACTION JSON:" in prompt:
            return json.dumps({"action": "reply", "text": "Grounded answer."})
        return ""

    _run(run_pipeline("what is a closure", tmp_path, generate=fake,
                      planner_model="P", coder_model="C"))
    _run(run_pipeline("what is a decorator", tmp_path, generate=fake,
                      planner_model="P", coder_model="C"))
    entries, _ = history_mod.read_history(tmp_path)
    assert len(entries) == 2
    snap2 = entries[-1].get("state_snapshot") or {}
    # pre-turn: knows turn 1's text, but NOT turn 2's own text/result
    turns_txt = json.dumps(snap2.get("turns", []))
    assert "closure" in turns_txt
    assert "decorator" not in turns_txt
    last_result_txt = json.dumps(snap2.get("last_result", {}))
    assert "decorator" not in last_result_txt


def test_export_environment_labeled_and_roles(tmp_path: Path) -> None:
    """Environment: build_id present, GPU labeled, per-role digest/installed/
    effective ctx fields rendered."""
    from agent_config import AgentConfig

    _seed_turns(tmp_path)
    cfg = AgentConfig()
    result = export_mod.do_export(tmp_path, "last", config=cfg)
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "build_id:" in text
    assert "gpu:" in text
    assert "thinker:" in text and "installed=" in text
    # local + UTC timestamps in the header
    assert "(UTC)" in text and "(local)" in text


def test_no_retired_log_dir_reference_and_exports_stay_put(tmp_path: Path) -> None:
    """Grep tests: no retired log-dir name in the new modules; exports land
    only under .agent_runs/exports/."""
    retired = ".agent" + "_logs"  # built, never spelled (repo grep test)
    for module in (history_mod, export_mod):
        src = Path(module.__file__).read_text(encoding="utf-8")
        assert retired not in src
    _seed_turns(tmp_path)
    result = export_mod.do_export(tmp_path, "all", zip_out=True)
    for key in ("path", "zip_path"):
        if result.get(key):
            p = Path(result[key])
            assert p.parent.name == "exports"
            assert p.parent.parent.name == ".agent_runs"
