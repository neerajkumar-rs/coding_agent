"""Phase 3 tests: SEARCH/REPLACE applier, token budgeter, AST repo map."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_core.coder import (
    EditApplyError,
    FileEdit,
    apply_file_edit,
    apply_search_replace,
    parse_coder_output,
    parse_edit_blocks,
)
from agent_core.context import (
    CHARS_PER_TOKEN,
    CONTEXT_SAFETY_FRACTION,
    ContextBudget,
    build_repo_map,
    estimate_tokens,
)

ORIGINAL = (
    "def greet(name):\n"
    "    return 'hi ' + name\n"
    "\n"
    "def farewell(name):\n"
    "    return 'bye ' + name\n"
)


def test_apply_exact_match() -> None:
    out = apply_search_replace(ORIGINAL, "return 'hi ' + name", "return f'Hello, {name}!'")
    assert "Hello, {name}!" in out
    assert "farewell" in out


def test_apply_fuzzy_whitespace_match() -> None:
    # Different indentation than the original -> exact fails, fuzzy wins.
    search = "  def greet(name):\n      return 'hi ' + name"
    out = apply_search_replace(ORIGINAL, search, "def greet(name):\n    return 'yo ' + name")
    assert "yo " in out


def test_apply_no_match_raises() -> None:
    with pytest.raises(EditApplyError, match="not found"):
        apply_search_replace(ORIGINAL, "def nonexistent():", "x = 1")


def test_apply_ambiguous_raises() -> None:
    dup = "x = 1\ny = 2\nx = 1\ny = 2\n"
    with pytest.raises(EditApplyError, match="ambiguous"):
        apply_search_replace(dup, "x = 1", "x = 9")


def test_apply_ambiguous_fuzzy_raises() -> None:
    dup = "if a:\n    pass\nif a:\n    pass\n"
    with pytest.raises(EditApplyError, match="ambiguous"):
        apply_search_replace(dup, "if a:\npass", "if a:\n    run()")


def test_parse_and_apply_edit_block_roundtrip() -> None:
    response = (
        "```FILE: app.py\n"
        "<<<<<<< SEARCH\n"
        "def greet(name):\n"
        "    return 'hi ' + name\n"
        "=======\n"
        "def greet(name: str) -> str:\n"
        "    return f'Hello, {name}!'\n"
        ">>>>>>> REPLACE\n"
        "```"
    )
    edits = parse_coder_output(response)
    assert len(edits) == 1
    assert edits[0].path == "app.py"
    assert edits[0].full_content is None
    result = apply_file_edit(ORIGINAL, edits[0])
    assert "Hello, {name}!" in result
    assert result.endswith("\n")


def test_edit_block_on_missing_file_fails() -> None:
    fe = FileEdit(path="new.py", edits=[("a", "b")])
    with pytest.raises(EditApplyError, match="does not exist"):
        apply_file_edit(None, fe)


def test_new_file_full_block_passthrough() -> None:
    response = "```FILE: fresh.py\nprint('new')\n```"
    edits = parse_coder_output(response)
    assert edits[0].full_content == "print('new')\n"
    assert apply_file_edit(None, edits[0]) == "print('new')\n"


def test_parse_edit_blocks_multiple() -> None:
    body = (
        "<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\nc\n=======\nd\n>>>>>>> REPLACE\n"
    )
    assert parse_edit_blocks(body) == [("a", "b"), ("c", "d")]


def test_token_estimator_ratio() -> None:
    text = "x" * 350
    assert estimate_tokens(text) == int(350 / CHARS_PER_TOKEN) + 1


def test_context_budget_85_percent_rule() -> None:
    budget = ContextBudget(num_ctx=4096, reserve_output_tokens=512)
    expected = int(4096 * CONTEXT_SAFETY_FRACTION) - 512
    assert budget.input_token_limit == expected
    assert budget.input_char_limit == int(expected * CHARS_PER_TOKEN)
    assert budget.fits("a" * 100)
    assert not budget.fits("a" * (budget.input_char_limit + 10))


def test_context_budget_trim_marks_tail() -> None:
    budget = ContextBudget(num_ctx=128, reserve_output_tokens=0)
    trimmed = budget.trim("z" * 10_000)
    assert len(trimmed) <= budget.input_char_limit
    assert trimmed.endswith("...[trimmed to fit context]")


def test_repo_map_extracts_symbols() -> None:
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "mod.py"
        p.write_text(
            "class Greeter:\n    def hello(self, name):\n        pass\n\n"
            "def helper(x):\n    pass\n",
            encoding="utf-8",
        )
        repo_map = build_repo_map(Path(tmp))
        assert "mod.py:" in repo_map
        assert "class Greeter()" in repo_map
        assert "def hello(self, name)" in repo_map
        assert "def helper(x)" in repo_map


def test_repo_map_skips_syntax_errors() -> None:
    with TemporaryDirectory() as tmp:
        Path(tmp, "bad.py").write_text("def broken(:", encoding="utf-8")
        assert build_repo_map(Path(tmp)) == ""
