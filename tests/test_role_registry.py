"""Group F seam tests: ceiling_for, RoleRegistry (validate + clamp + persist)."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_config import (
    AgentConfig,
    RoleRegistry,
    ceiling_for,
    canonical_role,
    load_agent_config,
    save_agent_config,
)


def test_ceiling_for_static_ceilings() -> None:
    assert ceiling_for("thinker") == 16384
    assert ceiling_for("coder") == 12288
    assert ceiling_for("validator") == 16384
    assert ceiling_for("planner") == 16384  # alias


def test_canonical_role_alias() -> None:
    assert canonical_role("planner") == "thinker"
    assert canonical_role("coder") == "coder"


def test_get_role_config_clamps_through_ceiling() -> None:
    cfg = AgentConfig(roles={"coder": {"num_ctx": 99999}})
    assert cfg.get_role_config("coder")["num_ctx"] == 12288
    # compat property clamps too
    cfg2 = AgentConfig(roles={"thinker": {"num_ctx": 99999}})
    assert cfg2.max_context_tokens == 16384


def test_registry_set_clamps_and_validates() -> None:
    cfg = AgentConfig()
    reg = RoleRegistry(cfg)
    out = reg.set("coder", num_ctx=99999, temperature=5.0)
    assert out["num_ctx"] == 12288  # clamped to the coder ceiling
    assert out["temperature"] == 2.0  # clamped
    with pytest.raises(ValueError, match="unknown role"):
        reg.set("nonexistent", num_ctx=1024)
    with pytest.raises(ValueError, match="unknown role fields"):
        reg.set("coder", bogus_field=1)
    with pytest.raises(ValueError, match="must not be empty"):
        reg.set("coder", model="  ")


def test_registry_set_persists_to_agent_file() -> None:
    with TemporaryDirectory() as tmp:
        cfg = AgentConfig()
        save_agent_config(cfg, tmp)
        reg = RoleRegistry(cfg, tmp)
        reg.set("thinker", model="someone/model:7b", num_ctx=4096)
        # persisted on disk
        disk = json.loads((Path(tmp) / ".agent").read_text(encoding="utf-8"))
        assert disk["roles"]["thinker"]["model"] == "someone/model:7b"
        assert disk["roles"]["thinker"]["num_ctx"] == 4096
        # and a fresh load sees it (round-trip survives)
        loaded = load_agent_config(tmp)
        assert loaded.get_role_config("thinker")["model"] == "someone/model:7b"
        # other roles untouched
        assert loaded.get_role_config("coder")["model"] == "qwen2.5-coder:3b-instruct"


def test_registry_without_workspace_does_not_write() -> None:
    with TemporaryDirectory() as tmp:
        cfg = AgentConfig()
        reg = RoleRegistry(cfg)  # no workspace -> in-memory only
        reg.set("coder", num_ctx=4096)
        assert cfg.get_role_config("coder")["num_ctx"] == 4096
        assert not (Path(tmp) / ".agent").exists()


# --- writer role (4th role): prose/summaries, thinker model by default ------

def test_writer_role_defaults() -> None:
    from agent_config import ROLE_CTX_CEILINGS, ROLE_NAMES

    assert "writer" in ROLE_NAMES
    cfg = AgentConfig()
    writer = cfg.get_role_config("writer")
    assert writer["model"] == "erwan2/DeepSeek-R1-Distill-Qwen-1.5B:latest"
    assert writer["temperature"] == 0.4  # warmer prose
    assert writer["num_predict"] == 1024
    assert writer["inline_system"] is True
    assert ROLE_CTX_CEILINGS["writer"] == 16384


def test_writer_role_via_registry_and_channels() -> None:
    from agent_core.run_state import LOG_CHANNELS

    assert "writer" in LOG_CHANNELS  # /logs writer works out of the box
    cfg = AgentConfig()
    reg = RoleRegistry(cfg)
    out = reg.set("writer", num_ctx=4096)
    assert out["num_ctx"] == 4096
    with pytest.raises(ValueError, match="unknown role"):
        reg.set("writers", num_ctx=1024)
