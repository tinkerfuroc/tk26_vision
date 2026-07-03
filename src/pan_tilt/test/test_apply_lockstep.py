"""Lockstep + apply-target tests: the Calibrate tab exposes exactly ONE
apply target (the $ROBOT_NAME per-robot pair) and the pair travels
both-or-neither."""
import os

import pytest

from pan_tilt.calibration import apply_to_urdf, urdf_targets


def test_single_per_robot_target(monkeypatch):
    monkeypatch.setenv("ROBOT_NAME", "tinker1")
    targets = urdf_targets.list_targets()
    assert len(targets) == 1, "exactly one per-robot apply target"
    t = targets[0]
    assert t.label == "robots/tinker1/pan_tilt/ (per-robot apply target)"
    assert t.form == "per-robot"
    assert t.build_package == "tinker_robot_config"
    assert t.build_command == "tkbuild tk25_basic --packages-select tinker_robot_config"


def test_target_without_robot_name_is_greyed_out(monkeypatch):
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    targets = urdf_targets.list_targets()
    assert len(targets) == 1
    assert targets[0].exists is False
    assert "ROBOT_NAME" in targets[0].label


def test_target_to_dict_keeps_ui_shape(monkeypatch):
    monkeypatch.setenv("ROBOT_NAME", "tinker1")
    d = urdf_targets.list_targets()[0].to_dict()
    # app.js renders these keys — keep the JSON shape stable.
    assert set(d) == {"label", "path", "exists", "form", "build_package",
                      "build_command", "workspace_hint"}


def _seed_tree(tmp_path, robot):
    d = tmp_path / "src/tinker_robot_config/robots" / robot / "pan_tilt"
    d.mkdir(parents=True)
    (d / "pan_tilt_overrides.xacro").write_text(
        '<?xml version="1.0"?>\n<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="x">\n'
        '  <xacro:property name="pan_tilt_attach_xyz" value="1 2 3"/>\n'
        '  <xacro:property name="pan_tilt_attach_rpy" value="0.1 0.2 0.3"/>\n'
        '  <xacro:property name="pan_tilt_camera_mount_xyz" value="4 5 6"/>\n'
        '  <xacro:property name="pan_tilt_camera_mount_rpy" value="-3.1 -0.8 -0.01"/>\n'
        '</robot>\n')
    (d / "offsets.yaml").write_text(
        "pan_tilt:\n  offsets:\n    pan_offset_rad: 0.0\n    tilt_offset_rad: 0.0\n")
    return d


def test_apply_is_lockstep_both_or_neither(tmp_path, monkeypatch):
    """A failure while landing the second file must roll back the first —
    geometry (overrides xacro) and runtime offsets (offsets.yaml) MUST come
    from the same solve, or the TF chain misrepresents the camera pose (the
    2026-04-30 half-applied-calibration bug)."""
    monkeypatch.setenv("ROBOT_NAME", "tinker1")
    d = _seed_tree(tmp_path, "tinker1")
    xacro_before = (d / "pan_tilt_overrides.xacro").read_text()
    offsets_before = (d / "offsets.yaml").read_text()

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(srcp, dstp):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated failure on the offsets.yaml replace")
        return real_replace(srcp, dstp)

    monkeypatch.setattr(os, "replace", flaky_replace)
    params = {
        "t_a": [-0.30, 0.007, 1.37],
        "t_b_trans": [0.057, 0.030, -0.039],
        "t_b_rotvec": [0.0, 0.0, -0.05],
        "theta_p_offset_rad": 0.012345,
        "theta_t_offset_rad": -0.523599,
    }
    with pytest.raises(OSError, match="simulated"):
        apply_to_urdf.apply_calibration(params, basic_root=tmp_path)

    assert (d / "pan_tilt_overrides.xacro").read_text() == xacro_before, \
        "xacro must be rolled back when the offsets write fails"
    assert (d / "offsets.yaml").read_text() == offsets_before
    assert not list(d.glob("*.tmp-*"))
