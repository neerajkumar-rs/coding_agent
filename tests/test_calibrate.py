"""/setup calibration: binary search, cancel-safety, atomic per-role write,
fingerprint gating of ceiling_for. All offline — the probe is injected."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import agent_config
from agent_config import (
    CALIBRATION_PATH,
    ROLE_CTX_CEILINGS,
    ceiling_for,
    machine_fingerprint,
    measured_ceiling,
)
from calibrate import (
    RUNGS,
    calibrate_role,
    run_calibration,
    write_calibration_atomic,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_caches():
    agent_config._reset_calibration_cache()
    yield
    agent_config._reset_calibration_cache()


# --- binary search -----------------------------------------------------------

def test_binary_search_picks_highest_passing_rung() -> None:
    calls: list[int] = []

    async def probe(role: str, ctx: int) -> dict:
        calls.append(ctx)
        return {"ok": ctx <= 8192, "fraction": 1.0 if ctx <= 8192 else 0.5}

    result = _run(calibrate_role("thinker", probe))
    assert result["ceiling"] == 8192
    assert result["cancelled"] is False
    # binary search over 5 rungs never probes more than 3
    assert len(calls) <= 3
    assert result["fractions"]["8192"] == 1.0


def test_binary_search_all_fail_gives_no_ceiling() -> None:
    async def probe(role: str, ctx: int) -> dict:
        return {"ok": False, "fraction": 0.3}

    result = _run(calibrate_role("coder", probe))
    assert result["ceiling"] is None
    # run_calibration backstops the lowest rung so the role still runs
    assert RUNGS[0] == 4096


# --- cancel-safety -----------------------------------------------------------

def test_cancel_mid_search_keeps_best_so_far() -> None:
    cancel = threading.Event()
    probed: list[int] = []

    async def probe(role: str, ctx: int) -> dict:
        probed.append(ctx)
        if len(probed) >= 1:
            cancel.set()  # cancel after the first probe
        return {"ok": True, "fraction": 1.0}

    result = _run(calibrate_role("thinker", probe, cancel_event=cancel))
    assert result["cancelled"] is True
    assert result["ceiling"] == probed[0]  # the one successful probe
    assert len(probed) == 1


# --- run_calibration: per-role atomic write ----------------------------------

class _FakeGateway:
    """Stands in for AgentGateway: probe_ctx reports ok up to a per-role cap."""

    def __init__(self, caps: dict[str, int]) -> None:
        self.caps = caps

    async def probe_ctx(self, role: str, num_ctx: int) -> dict:
        cap = self.caps.get(role, 0)
        ok = num_ctx <= cap
        return {"ok": ok, "fraction": 1.0 if ok else 0.5}


def test_run_calibration_writes_after_each_role(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    snapshots: list[dict] = []

    def on_done(role: str, _result: dict) -> None:
        # The file must already contain THIS role's result (atomic per role).
        snapshots.append(json.loads(path.read_text(encoding="utf-8")))

    gateway = _FakeGateway({"thinker": 8192, "coder": 6144})
    data = _run(run_calibration(
        gateway, ["thinker", "coder"], path=path, on_role_done=on_done,
    ))
    assert data["roles"]["thinker"]["ceiling"] == 8192
    assert data["roles"]["coder"]["ceiling"] == 6144
    assert data["fingerprint"] == machine_fingerprint()
    # after role 1: only thinker; after role 2: both
    assert set(snapshots[0]["roles"]) == {"thinker"}
    assert set(snapshots[1]["roles"]) == {"thinker", "coder"}


def test_run_calibration_cancel_between_roles_keeps_completed(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    cancel = threading.Event()

    def on_done(role: str, _result: dict) -> None:
        cancel.set()

    gateway = _FakeGateway({"thinker": 8192, "coder": 6144})
    data = _run(run_calibration(
        gateway, ["thinker", "coder"], path=path,
        cancel_event=cancel, on_role_done=on_done,
    ))
    # thinker completed and was persisted; coder never probed
    assert set(data["roles"]) == {"thinker"}
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert set(on_disk["roles"]) == {"thinker"}


# --- atomic write ------------------------------------------------------------

def test_write_calibration_atomic(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    write_calibration_atomic({"fingerprint": "abc", "roles": {}}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["fingerprint"] == "abc"
    assert not path.with_suffix(".json.tmp").exists()  # temp cleaned up


# --- fingerprint gating + ceiling_for integration ----------------------------

def test_measured_ceiling_requires_matching_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    write_calibration_atomic({
        "fingerprint": "someone-elses-machine",
        "roles": {"thinker": {"ceiling": 12288}},
    }, path)
    assert measured_ceiling("thinker", path) is None

    write_calibration_atomic({
        "fingerprint": machine_fingerprint(),
        "roles": {"thinker": {"ceiling": 12288}},
    }, path)
    assert measured_ceiling("thinker", path) == 12288


def test_ceiling_for_prefers_measured_when_fingerprint_matches(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "calibration.json"
    write_calibration_atomic({
        "fingerprint": machine_fingerprint(),
        "roles": {"coder": {"ceiling": 4096}},
    }, path)
    monkeypatch.setattr(agent_config, "CALIBRATION_PATH", path)
    agent_config._reset_calibration_cache()
    try:
        assert ceiling_for("coder") == 4096  # measured wins
        # unmeasured roles keep the static ceiling
        assert ceiling_for("thinker") == ROLE_CTX_CEILINGS["thinker"]
    finally:
        agent_config._reset_calibration_cache()


def test_ceiling_for_static_when_no_calibration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent_config, "CALIBRATION_PATH", tmp_path / "nope.json")
    agent_config._reset_calibration_cache()
    try:
        assert ceiling_for("coder") == ROLE_CTX_CEILINGS["coder"]
    finally:
        agent_config._reset_calibration_cache()


def test_calibration_path_is_repo_root() -> None:
    # machine-level: one file next to the controller, not per workspace
    assert CALIBRATION_PATH.name == "calibration.json"
    assert CALIBRATION_PATH.parent == Path(agent_config.__file__).resolve().parent
