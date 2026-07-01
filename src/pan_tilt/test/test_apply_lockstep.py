"""Task 5 tests: lockstep enforcement + authoritative URDF label checks."""

from pan_tilt.calibration import urdf_targets


def test_macro_is_authoritative_and_first():
    targets = urdf_targets.list_targets()
    assert targets, "expected at least the macro target"
    assert targets[0].form == "macro"
    assert "authoritative" in targets[0].label.lower()


def test_dead_standalone_marked_non_runtime():
    # The tk26 pan_tilt/urdf/pan_tilt.urdf.xacro is NOT the runtime URDF
    # (the dev launch renders tinker_urdf/.../pan_tilt_standalone.urdf.xacro,
    # which includes the macro). It must be flagged so an operator can't apply
    # it alone and think they patched the runtime.
    targets = urdf_targets.list_targets()
    standalone = [t for t in targets if t.form == "standalone"]
    for t in standalone:
        assert ("not the runtime" in t.label.lower()) or ("legacy" in t.label.lower())


def test_no_yaml_without_allow_partial_is_refused(tmp_path, monkeypatch):
    import sys
    import pytest
    from pan_tilt.calibration import apply_to_urdf

    # Minimal results json + a macro xacro to satisfy file reads.
    results = tmp_path / "chain.json"
    results.write_text(
        '{"params": {"t_a":[0,0,0],"t_b_trans":[0,0,0],'
        '"t_b_rotvec":[0,0,0],"theta_t_offset_rad":0.0,'
        '"theta_p_offset_rad":0.0}}'
    )
    xacro = tmp_path / "pan_tilt.urdf.xacro"
    xacro.write_text(
        '<robot><xacro:macro name="pan_tilt_macro" '
        "params=\"parent attach_xyz:='0 0 0' attach_rpy:='0 0 0'\">"
        '<joint name="camera_mount_joint" type="fixed">'
        '<origin xyz="0 0 0" rpy="0 0 0"/></joint></xacro:macro></robot>'
    )
    argv = ["apply_to_urdf", "--results", str(results), "--xacro", str(xacro), "--no-yaml"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(apply_to_urdf.CalibrationApplyError):
        apply_to_urdf.main()
