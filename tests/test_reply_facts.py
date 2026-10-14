"""Reply evidence tests: named-file content in FACTS, listing skeleton,
empty-workspace sentence, Python table override (offline, fake generate)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from understand import (
    build_facts_block,
    build_listing_skeleton,
    build_reply_facts,
    find_named_files,
    is_listing_request,
    reply_contradicts_table,
    reply_step,
)


def _run(coro):
    return asyncio.run(coro)


# --- named-file resolution -----------------------------------------------------

def test_find_named_files_exact_path_and_basename() -> None:
    all_files = ["index.html", "src/app.py", "README.md"]
    found, absent = find_named_files("what is in index.html use cat to see", all_files)
    assert found == ["index.html"] and absent == []
    found, absent = find_named_files("open src/app.py please", all_files)
    assert found == ["src/app.py"] and absent == []
    found, absent = find_named_files("what does app.py do", all_files)
    assert found == ["src/app.py"] and absent == []  # basename match
    found, absent = find_named_files("what is in missing.py", all_files)
    assert found == [] and absent == ["missing.py"]


def test_find_named_files_extensionless() -> None:
    all_files = ["index.html", "README.md", "Dockerfile"]
    found, _absent = find_named_files("update the readme", all_files)
    assert found == ["README.md"]
    found, _absent = find_named_files("show the dockerfile", all_files)
    assert found == ["Dockerfile"]
    # an extensionless word that matches nothing is not "absent" (English use)
    _f, absent = find_named_files("what license should i pick", ["app.py"])
    assert absent == []


# --- FACTS block carries content ------------------------------------------------

def test_reply_facts_include_named_file_content() -> None:
    with TemporaryDirectory() as tmp:
        Path(tmp, "index.html").write_text(
            "<!doctype html><html><body><h1>File List</h1></body></html>\n",
            encoding="utf-8",
        )
        from orchestrator import build_file_facts

        facts = build_file_facts(Path(tmp))
        block = build_reply_facts(
            tmp, "what is in index.html use cat to see", facts,
        )
        assert "FILE CONTENT: index.html" in block
        assert "File List" in block  # the actual content went in


def test_reply_facts_notes_absent_named_file() -> None:
    with TemporaryDirectory() as tmp:
        Path(tmp, "index.html").write_text("<html>x</html>", encoding="utf-8")
        from orchestrator import build_file_facts

        facts = build_file_facts(Path(tmp))
        block = build_reply_facts(tmp, "what is in config.py", facts)
        # the note marker the reply prompt maps to "<name> is not in this workspace"
        assert "NOT-IN-WORKSPACE: config.py" in block


def test_reply_facts_skips_binary_with_note() -> None:
    with TemporaryDirectory() as tmp:
        Path(tmp, "data.bin").write_bytes(b"\x00\x01\x02\xff" * 8)
        Path(tmp, "index.html").write_text("<html>x</html>", encoding="utf-8")
        from orchestrator import build_file_facts

        facts = build_file_facts(Path(tmp))
        # data.bin is in all_files (listed) but not readable as text
        assert "data.bin" in facts["all_files"]
        block = build_reply_facts(tmp, "what is in data.bin", facts)
        assert "binary" in block
        assert "not in this workspace" not in block  # listed, not absent


# --- the answer is not just a file name ------------------------------------------

def test_reply_answers_from_named_file_content() -> None:
    """Pipeline level: 'what is in index.html use cat to see' gets an answer
    grounded in the file CONTENT (the model sees 'File List')."""
    from orchestrator import run_pipeline

    seen: list[str] = []

    async def fake(model, prompt, system="", temperature=0.2, **kw):
        if "CLASSIFY MODE:" in prompt:
            return "reply"
        if "FACTS:" in prompt:
            seen.append(prompt)
            assert "FILE CONTENT: index.html" in prompt
            assert "File List" in prompt
            return "index.html contains a heading that says 'File List'."
        return ""

    with TemporaryDirectory() as tmp:
        Path(tmp, "index.html").write_text(
            "<!doctype html><html><body><h1>File List</h1></body></html>\n",
            encoding="utf-8",
        )
        result = _run(run_pipeline(
            "what is in index.html use cat to see", tmp, generate=fake,
            planner_model="P", coder_model="C",
        ))
        assert result["answer"] != "index.html"  # never just the file name
        assert "File List" in result["answer"]
        assert seen, "reply step never ran"


# --- empty workspace ------------------------------------------------------------

def test_empty_workspace_listing_answers_one_sentence_no_model_call() -> None:
    from orchestrator import run_pipeline

    calls: list[str] = []

    async def fake(model, prompt, **kw):
        calls.append(prompt)
        if "CLASSIFY MODE:" in prompt:
            return "reply"
        return "unused"

    with TemporaryDirectory() as tmp:
        result = _run(run_pipeline(
            "list all the files", tmp, generate=fake,
            planner_model="P", coder_model="C",
        ))
        assert result["answer"] == "The workspace is empty — there are no files to list."
        # one model call at most (the classify) — the answer is deterministic
        assert len(calls) == 1


def test_build_facts_block_empty_workspace_line() -> None:
    block = build_facts_block({"files": [], "flags": {}})
    assert "empty" in block.lower()


# --- listing skeleton + contradiction override -----------------------------------

def test_listing_skeleton_built_in_python() -> None:
    facts = {"files": [
        {"name": "index.html", "language": "html", "lines": 5, "bytes": 2048},
        {"name": "notes.txt", "language": "text", "lines": 2, "bytes": 12},
    ]}
    skeleton = build_listing_skeleton(facts)
    assert "index.html | html | 5 lines | 2.0 KB" in skeleton
    assert "notes.txt" in skeleton
    assert is_listing_request("list all the files")
    assert is_listing_request("ok use cli tool and list all file and and there types")
    assert not is_listing_request("make readme.md")


def test_python_table_overrides_contradicting_model_reply() -> None:
    """The model lists a file that is not in the table -> the Python table
    replaces the reply."""
    facts = {"files": [
        {"name": "index.html", "language": "html", "lines": 5, "bytes": 2048},
    ]}
    skeleton = build_listing_skeleton(facts)

    async def fake(model, prompt, **kw):
        return "Here are the files: index.html and secret.py."  # invented

    answer = _run(reply_step(
        "list all the files", generate=fake,
        facts_block=build_facts_block(facts),
        skeleton=skeleton, table_names=["index.html"],
    ))
    assert "secret.py" not in answer
    assert answer == skeleton


def test_reply_kept_when_consistent_with_table() -> None:
    facts = {"files": [
        {"name": "index.html", "language": "html", "lines": 5, "bytes": 2048},
    ]}
    skeleton = build_listing_skeleton(facts)

    async def fake(model, prompt, **kw):
        return "There is one file, index.html (2.0 KB)."

    answer = _run(reply_step(
        "list all the files", generate=fake,
        facts_block=build_facts_block(facts),
        skeleton=skeleton, table_names=["index.html"],
    ))
    assert "index.html" in answer
    assert "one file" in answer  # model framing kept


def test_reply_contradicts_table_detector() -> None:
    assert reply_contradicts_table("files: a.py and b.py", ["a.py"]) == "b.py"
    assert reply_contradicts_table("only index.html", ["index.html"]) is None
    assert reply_contradicts_table("no files at all", ["index.html"]) is None


# --- pipeline level: listing answer built from the table ---------------------------

def test_listing_answer_comes_from_python_table() -> None:
    from orchestrator import run_pipeline
    from tests.test_run_pipeline import make_fake_generate

    fake = make_fake_generate({
        "CLASSIFY MODE:": "reply",
        "REPLY:": "You have index.html plus definitely utils.py and app.py.",
    })
    with TemporaryDirectory() as tmp:
        Path(tmp, "index.html").write_text(
            "<!doctype html><html><body><h1>File List</h1></body></html>\n",
            encoding="utf-8",
        )
        result = _run(run_pipeline(
            "ok use cli tool and list all file and and there types",
            tmp, generate=fake, planner_model="P", coder_model="C",
        ))
        # the model invented utils.py/app.py -> the Python table replaces it
        assert "utils.py" not in result["answer"]
        assert "app.py" not in result["answer"]
        assert "index.html" in result["answer"]
        assert "KB" in result["answer"]
