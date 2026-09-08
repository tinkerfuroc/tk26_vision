"""Atomic write-path tests for the per-robot apply (pair writer + symlink),
plus tombstone guards that the deleted shared-source patchers stay deleted."""
import os
from pathlib import Path

import pytest

from pan_tilt.calibration.apply_to_urdf import (
    CalibrationApplyError,
    _atomic_write_pair,
    _atomic_write_single,
)


def test_atomic_write_single_preserves_symlink(tmp_path):
    # A symlinked target (e.g. --symlink-install): write must reach the real
    # file through the link, preserving the link itself.
    src = tmp_path / "src" / "pan_tilt" / "config"
    src.mkdir(parents=True)
    src_file = src / "pan_tilt.yaml"
    src_file.write_text("tilt_offset_rad: 0.0\n")
    inst = tmp_path / "install" / "pan_tilt" / "share" / "pan_tilt" / "config"
    inst.mkdir(parents=True)
    link = inst / "pan_tilt.yaml"
    link.symlink_to(src_file)

    res = _atomic_write_single(link, "tilt_offset_rad: 1.5\n")

    assert res["applied"] is True
    assert link.is_symlink(), "the symlink must be preserved, not replaced by a real file"
    assert src_file.read_text() == "tilt_offset_rad: 1.5\n", \
        "the write must reach the real target through the symlink"


def test_atomic_write_pair_writes_both_with_backups(tmp_path):
    a = tmp_path / "a.xacro"
    b = tmp_path / "b.yaml"
    a.write_text("old-a\n")
    b.write_text("old-b\n")

    res = _atomic_write_pair([(a, "new-a\n"), (b, "new-b\n")])

    assert [r["applied"] for r in res] == [True, True]
    assert a.read_text() == "new-a\n" and b.read_text() == "new-b\n"
    for r, original in zip(res, ("old-a\n", "old-b\n")):
        assert r["backup_path"] is not None
        assert Path(r["backup_path"]).read_text() == original
    assert not list(tmp_path.glob("*.tmp-*"))


def test_atomic_write_pair_noop_when_unchanged(tmp_path):
    a = tmp_path / "a.xacro"
    b = tmp_path / "b.yaml"
    a.write_text("same-a\n")
    b.write_text("same-b\n")

    res = _atomic_write_pair([(a, "same-a\n"), (b, "same-b\n")])

    assert [r["applied"] for r in res] == [False, False]
    assert all(r["backup_path"] is None for r in res)
    assert not list(tmp_path.glob("*.old-*")) and not list(tmp_path.glob("*.tmp-*"))


def test_atomic_write_pair_partial_change_backs_up_only_changed(tmp_path):
    a = tmp_path / "a.xacro"
    b = tmp_path / "b.yaml"
    a.write_text("old-a\n")
    b.write_text("same-b\n")

    res = _atomic_write_pair([(a, "new-a\n"), (b, "same-b\n")])

    assert [r["applied"] for r in res] == [True, False]
    assert res[0]["backup_path"] is not None and res[1]["backup_path"] is None
    assert len(list(tmp_path.glob("*.old-*"))) == 1


def test_atomic_write_pair_creates_missing_file_without_backup(tmp_path):
    a = tmp_path / "a.xacro"
    b = tmp_path / "b.yaml"          # does not exist yet
    a.write_text("old-a\n")

    res = _atomic_write_pair([(a, "new-a\n"), (b, "fresh-b\n")])

    assert [r["applied"] for r in res] == [True, True]
    assert b.read_text() == "fresh-b\n"
    assert res[1]["backup_path"] is None


def test_atomic_write_pair_rolls_back_first_when_second_replace_fails(
        tmp_path, monkeypatch):
    """Lockstep: if the second os.replace fails, the first file must be
    rolled back to its original content — never a half-applied pair."""
    a = tmp_path / "a.xacro"
    b = tmp_path / "b.yaml"
    a.write_text("old-a\n")
    b.write_text("old-b\n")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(srcp, dstp):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated replace failure")
        return real_replace(srcp, dstp)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated"):
        _atomic_write_pair([(a, "new-a\n"), (b, "new-b\n")])

    assert a.read_text() == "old-a\n", "first file must be rolled back"
    assert b.read_text() == "old-b\n"
    assert not list(tmp_path.glob("*.tmp-*")), "no staged tmp files left behind"


def test_old_shared_source_patchers_are_gone():
    """Tombstone: the macro/regex patchers targeting the SHARED xacros must
    stay deleted — running an old-style apply against the per-robot macro
    format would silently destroy per-robot behavior."""
    from pan_tilt.calibration import apply_to_urdf as ap
    for name in (
        "_patched_macro", "_patched_standalone", "_patched_xacro",
        "ATTACH_XYZ_DEFAULT_RE", "ATTACH_RPY_DEFAULT_RE", "JOINT_BLOCK_RE",
        "MACRO_DECL_RE", "_patch_yaml_offsets", "_patch_urdf_overrides",
        "_resolve_overrides_yaml", "_resolve_yaml_path", "resolve_source_path",
    ):
        assert not hasattr(ap, name), f"{name} must stay deleted"


def test_calib_web_wired_to_per_robot_apply():
    # Wiring check — use find_spec so we read the source without executing the
    # module, which would pull in rclpy / ROS2 not present in .venv-calib.
    import importlib.util
    spec = importlib.util.find_spec("pan_tilt.calib_web")
    assert spec is not None and spec.origin is not None, "pan_tilt.calib_web not found"
    src = Path(spec.origin).read_text()
    assert "apply_calibration_detail" in src, \
        "calib_web Apply must go through the per-robot apply entry"
    assert "render_calibration" in src, \
        "calib_web Preview must render the per-robot files"
    for legacy in ("_patched_xacro", "resolve_source_path",
                   "_patch_yaml_offsets", "_patch_urdf_overrides",
                   "list_yaml_targets"):
        assert legacy not in src, \
            f"calib_web must not reference the deleted {legacy} path"
    # Refusals (ROBOT_NAME unset etc.) must surface as HTTP 400, not 200/500.
    assert "CalibrationApplyError as exc" in src and "HTTPException(400" in src


def test_calibration_apply_error_importable():
    assert issubclass(CalibrationApplyError, RuntimeError)
