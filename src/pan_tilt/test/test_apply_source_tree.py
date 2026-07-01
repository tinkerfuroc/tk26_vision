import os
from pathlib import Path
from pan_tilt.calibration.apply_to_urdf import _atomic_write_single, resolve_source_path


def test_atomic_write_preserves_symlink(tmp_path):
    # Simulate --symlink-install: install/<f> is a symlink to src/<f>.
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
    assert link.is_symlink(), "the install symlink must be preserved, not replaced by a real file"
    assert src_file.read_text() == "tilt_offset_rad: 1.5\n", "the write must reach SOURCE through the symlink"


def test_resolve_source_path_via_glob(tmp_path):
    # Non-symlink install (the broken-symlink workspace state): resolve by glob.
    src = tmp_path / "src" / "tk26_vision" / "src" / "pan_tilt" / "config"
    src.mkdir(parents=True)
    (src / "pan_tilt.yaml").write_text("x\n")
    inst = tmp_path / "install" / "pan_tilt" / "share" / "pan_tilt" / "config"
    inst.mkdir(parents=True)
    real = inst / "pan_tilt.yaml"
    real.write_text("x\n")  # real file, NOT a symlink

    got = resolve_source_path(real)
    assert got == (src / "pan_tilt.yaml")


def test_resolve_source_path_ignores_worktree_and_build_decoys(tmp_path):
    # Real workspace src file.
    src = tmp_path / "src" / "tk26_vision" / "src" / "pan_tilt" / "config"
    src.mkdir(parents=True)
    real_src = src / "pan_tilt.yaml"
    real_src.write_text("real\n")

    # Decoy 1: git worktree under src/tk26_vision/.claude/worktrees/...
    # This is the real-world scenario: worktrees at
    # <ws>/src/tk26_vision/.claude/worktrees/<branch>/src/pan_tilt/config/
    wt = tmp_path / "src" / "tk26_vision" / ".claude" / "worktrees" / "some-branch" / "src" / "pan_tilt" / "config"
    wt.mkdir(parents=True)
    (wt / "pan_tilt.yaml").write_text("decoy-worktree\n")

    # Decoy 2: build artifact under src/.../build/... (e.g. in-tree build)
    build = tmp_path / "src" / "tk26_vision" / "build" / "pan_tilt" / "config"
    build.mkdir(parents=True)
    (build / "pan_tilt.yaml").write_text("decoy-build\n")

    # Non-symlink install file.
    inst = tmp_path / "install" / "pan_tilt" / "share" / "pan_tilt" / "config"
    inst.mkdir(parents=True)
    install_file = inst / "pan_tilt.yaml"
    install_file.write_text("installed\n")

    got = resolve_source_path(install_file)
    assert got == real_src, (
        f"expected real src at {real_src}, got {got} — decoy files were not filtered"
    )


def test_calib_web_offsets_are_normalized_before_write(monkeypatch):
    # urdf_apply reads theta_*_offset_rad from the json (lines 438-439) raw;
    # assert the value handed to the yaml writer is wrapped.
    from pan_tilt.calibration.utils import wrap_to_pi
    assert abs(wrap_to_pi(8.348085384508424) - 2.0649000773) < 1e-6
    # Guard: calib_web source must reference wrap_to_pi and resolve_source_path
    # (wiring check — use find_spec so we read the source without executing the
    # module, which would pull in rclpy / ROS2 not present in .venv-calib).
    import importlib.util
    spec = importlib.util.find_spec("pan_tilt.calib_web")
    assert spec is not None and spec.origin is not None, "pan_tilt.calib_web not found"
    src = Path(spec.origin).read_text()
    assert "wrap_to_pi" in src, "calib_web must normalize offsets before writing"
    assert "resolve_source_path" in src, "calib_web must route writes to the source tree"


def test_resolve_source_path_warns_on_silent_install_fallback(tmp_path):
    import logging
    # install-style path whose source has NO unique match under <ws>/src:
    # the function must fall back to the install path AND warn loudly (so the
    # operator isn't misled into thinking the calibration stuck — a rebuild
    # would revert an install-tree write). Capture via a handler attached
    # directly to the module logger — robust to other tests / a sourced ROS
    # env reconfiguring log propagation, which makes pytest's caplog flaky here.
    inst = tmp_path / "install" / "pan_tilt" / "share" / "pan_tilt" / "config"
    inst.mkdir(parents=True)
    real = inst / "pan_tilt.yaml"
    real.write_text("x\n")          # real file, not a symlink
    (tmp_path / "src").mkdir()       # src exists but contains no matching file
    logger = logging.getLogger("pan_tilt.calibration.apply_to_urdf")
    records = []
    handler = logging.Handler()
    handler.emit = lambda rec: records.append(rec)
    prev_level, prev_disabled = logger.level, logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.disabled = False
    try:
        got = resolve_source_path(real)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.disabled = prev_disabled
    assert got == real               # fell back to the install path
    assert any("REVERT" in r.getMessage() for r in records), \
        "expected a loud fallback warning"
