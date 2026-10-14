"""Session-level test isolation.

A machine-local ``calibration.json`` in the repo root (written by /setup or
calibrate.py) must never skew the suite: point agent_config at a path that
does not exist so the static role ceilings apply. test_calibrate.py overrides
this with its own monkeypatched paths per test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_calibration(tmp_path, monkeypatch):
    import agent_config

    monkeypatch.setattr(agent_config, "CALIBRATION_PATH",
                        tmp_path / "no-calibration-here.json")
    agent_config._reset_calibration_cache()
    yield
    agent_config._reset_calibration_cache()
