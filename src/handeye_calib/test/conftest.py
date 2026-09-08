"""Shared pytest fixtures for the handeye_calib suite.

CRITICAL TEST HYGIENE: ``do_capture`` / ``do_solve`` / ``_persist_session`` now
write capture sessions + solve dumps to disk under ``$HANDEYE_DUMP_DIR`` (default
``calibration_data/``). Any test that drives a real capture/solve would otherwise
pollute the OPERATOR'S real ``calibration_data/wrist_handeye_sessions/`` tree with
synthetic n=1 sessions. This autouse fixture points every test at a throwaway tmp
dir, so the real tree is never touched no matter which test runs.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolate_handeye_dump_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDEYE_DUMP_DIR", str(tmp_path))
