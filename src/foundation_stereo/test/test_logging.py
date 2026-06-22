"""Tests for the vision_log session-directory resolver.

Resolution order per tk26_vision convention (see top-level CLAUDE.md):
  1. $TINKER_VISION_SESSION_TS (must match YYYYmmdd_HHMMSS).
  2. Newest existing `<base>/<YYYYmmdd_HHMMSS>/` subdir by mtime.
  3. Fresh `strftime` cold-start.
"""

import os
import time

from foundation_stereo._logging import resolve_session_dir


def test_env_var_takes_priority(tmp_path, monkeypatch):
    base = tmp_path / "vision_log"
    base.mkdir()
    monkeypatch.setenv("TINKER_VISION_SESSION_TS", "20260101_120000")
    out = resolve_session_dir(str(base))
    assert out == str(base / "20260101_120000")
    assert os.path.isdir(out)


def test_env_var_rejected_if_malformed(tmp_path, monkeypatch):
    base = tmp_path / "vision_log"
    base.mkdir()
    monkeypatch.setenv("TINKER_VISION_SESSION_TS", "not-a-timestamp")
    out = resolve_session_dir(str(base))
    # Falls through to fresh-strftime cold-start; basename matches YYYYmmdd_HHMMSS.
    assert os.path.basename(out).count("_") == 1
    assert len(os.path.basename(out)) == 15


def test_newest_subdir_wins_when_env_missing(tmp_path, monkeypatch):
    base = tmp_path / "vision_log"
    base.mkdir()
    older = base / "20250101_000000"
    newer = base / "20260101_000000"
    older.mkdir()
    time.sleep(0.05)
    newer.mkdir()
    monkeypatch.delenv("TINKER_VISION_SESSION_TS", raising=False)

    out = resolve_session_dir(str(base))
    assert out == str(newer)


def test_cold_start_when_no_subdirs(tmp_path, monkeypatch):
    base = tmp_path / "vision_log"
    base.mkdir()
    monkeypatch.delenv("TINKER_VISION_SESSION_TS", raising=False)

    out = resolve_session_dir(str(base))
    assert os.path.dirname(out) == str(base)
    assert len(os.path.basename(out)) == 15
    assert os.path.isdir(out)
