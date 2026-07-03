"""Task 3 (Phase 1c): calibration Apply targets ONLY the two per-robot files.

`apply_calibration` writes `robots/<ROBOT_NAME>/pan_tilt/pan_tilt_overrides.xacro`
+ `offsets.yaml` in the tk25_basic SOURCE tree — never the shared tinker_urdf
xacros — and refuses outright when ROBOT_NAME is unset.

Also hosts the ports of the old `_patched_xacro` forward-camera-guard tests and
the old CLI `main()` lockstep/idempotency tests (test_calibration.py), retargeted
at the per-robot renderer/writer.
"""
import json
import time

import pytest

from pan_tilt.calibration import apply_to_urdf as ap


def _fake_params(with_rotvecs=True):
    p = {
        't_a': [-0.30, 0.007, 1.37],
        't_b_trans': [0.057, 0.030, -0.039],
        'theta_p_offset_rad': 3.0866366614,
        'theta_t_offset_rad': -3.0669471551,
    }
    if with_rotvecs:
        p['t_a_rotvec'] = [0.009, -0.028, -0.0001]
        p['t_b_rotvec'] = [-1.2, -0.4, 2.9]
    return p


def _seed_tree(tmp_path, robot):
    d = tmp_path / 'src/tinker_robot_config/robots' / robot / 'pan_tilt'
    d.mkdir(parents=True)
    (d / 'pan_tilt_overrides.xacro').write_text(
        '<?xml version="1.0"?>\n<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="x">\n'
        '  <xacro:property name="pan_tilt_attach_xyz" value="1 2 3"/>\n'
        '  <xacro:property name="pan_tilt_attach_rpy" value="0.1 0.2 0.3"/>\n'
        '  <xacro:property name="pan_tilt_camera_mount_xyz" value="4 5 6"/>\n'
        '  <xacro:property name="pan_tilt_camera_mount_rpy" value="-3.1 -0.8 -0.01"/>\n'
        '</robot>\n')
    (d / 'offsets.yaml').write_text(
        'pan_tilt:\n  offsets:\n    pan_offset_rad: 0.0\n    tilt_offset_rad: 0.0\n')
    return d


def test_refuses_without_robot_name(tmp_path, monkeypatch):
    monkeypatch.delenv('ROBOT_NAME', raising=False)
    with pytest.raises(ap.CalibrationApplyError, match='ROBOT_NAME'):
        ap.apply_calibration(_fake_params(), basic_root=tmp_path)


def test_writes_only_per_robot_files(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    d = _seed_tree(tmp_path, 'tinker1')
    written = ap.apply_calibration(_fake_params(), basic_root=tmp_path,
                                   allow_flipped_camera=True)
    assert set(written) == {d / 'pan_tilt_overrides.xacro', d / 'offsets.yaml'}
    text = (d / 'pan_tilt_overrides.xacro').read_text()
    assert 'pan_tilt_attach_xyz' in text and '<joint' not in text
    assert not (tmp_path / 'src/tinker_urdf').exists()   # never touches shared xacros


def test_trivial_rotvec_preserves_existing_rpy(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    d = _seed_tree(tmp_path, 'tinker1')
    ap.apply_calibration(_fake_params(with_rotvecs=False), basic_root=tmp_path,
                         allow_flipped_camera=True)
    text = (d / 'pan_tilt_overrides.xacro').read_text()
    assert 'value="0.1 0.2 0.3"' in text        # attach_rpy preserved
    assert 'value="-3.1 -0.8 -0.01"' in text    # camera_mount_rpy preserved


def test_tinker1_apply_never_touches_tinker2(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    _seed_tree(tmp_path, 'tinker1')
    d2 = _seed_tree(tmp_path, 'tinker2')
    before = (d2 / 'pan_tilt_overrides.xacro').read_text()
    ap.apply_calibration(_fake_params(), basic_root=tmp_path, allow_flipped_camera=True)
    assert (d2 / 'pan_tilt_overrides.xacro').read_text() == before


# ---- refusal / resolution details -------------------------------------------


def test_apply_error_alias():
    assert ap.ApplyError is ap.CalibrationApplyError


def test_refuses_unknown_robot_profile(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker9')
    _seed_tree(tmp_path, 'tinker1')      # only tinker1 exists
    with pytest.raises(ap.CalibrationApplyError, match='tinker9'):
        ap.apply_calibration(_fake_params(), basic_root=tmp_path,
                             allow_flipped_camera=True)


def test_refuses_missing_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    d = _seed_tree(tmp_path, 'tinker1')
    (d / 'pan_tilt_overrides.xacro').unlink()
    with pytest.raises(ap.CalibrationApplyError, match='pan_tilt_overrides.xacro'):
        ap.apply_calibration(_fake_params(), basic_root=tmp_path,
                             allow_flipped_camera=True)


def test_offsets_yaml_is_rendered_normalized(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    d = _seed_tree(tmp_path, 'tinker1')
    p = _fake_params(with_rotvecs=False)
    p['theta_t_offset_rad'] = 8.348085384508424   # +478.31 deg, un-normalized
    ap.apply_calibration(p, basic_root=tmp_path, allow_flipped_camera=True)
    out = (d / 'offsets.yaml').read_text()
    assert 'tilt_offset_rad: 2.0649' in out       # wrapped to +118.31 deg
    assert '8.34' not in out
    assert 'pan_offset_rad: 3.0866366614' in out  # in range, unchanged


# ---- ported: forward-camera invariant (was test_calibration.py:1248-1344,
#      against the deleted `_patched_xacro`) ----------------------------------


def test_flipped_yaw_refused_by_default(tmp_path, monkeypatch):
    """Layer-3 failsafe: |yaw| > pi/2 in the fitted camera_mount rotation is
    refused unless allow_flipped_camera is explicitly set."""
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    _seed_tree(tmp_path, 'tinker1')
    p = _fake_params(with_rotvecs=False)
    p['t_b_rotvec'] = [0.10, -0.005, 3.06]        # the live-incident value
    with pytest.raises(ap.CalibrationApplyError, match='yaw'):
        ap.apply_calibration(p, basic_root=tmp_path)


def test_flipped_yaw_allowed_with_override(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    d = _seed_tree(tmp_path, 'tinker1')
    p = _fake_params(with_rotvecs=False)
    p['t_b_rotvec'] = [0.10, -0.005, 3.06]
    ap.apply_calibration(p, basic_root=tmp_path, allow_flipped_camera=True)
    text = (d / 'pan_tilt_overrides.xacro').read_text()
    expect = ap._rotvec_to_rpy_str(p['t_b_rotvec'], True)
    assert f'name="pan_tilt_camera_mount_rpy" value="{expect}"' in text
    import numpy as np
    yaw_written = float(expect.split()[2])
    assert abs(abs(yaw_written) - np.pi) < 0.2


def test_normal_yaw_passes_without_override(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    d = _seed_tree(tmp_path, 'tinker1')
    p = _fake_params(with_rotvecs=False)
    p['t_b_rotvec'] = [0.10, -0.005, 0.05]        # ~3 deg yaw, fine
    ap.apply_calibration(p, basic_root=tmp_path)  # must not raise
    text = (d / 'pan_tilt_overrides.xacro').read_text()
    expect = ap._rotvec_to_rpy_str(p['t_b_rotvec'], False)
    assert f'name="pan_tilt_camera_mount_rpy" value="{expect}"' in text


def test_nontrivial_rotvecs_overwrite_rpy(tmp_path, monkeypatch):
    """Ported from the macro-form `_patched_xacro` tests: a non-trivial rotvec
    must REPLACE the stored rpy (trivial ones preserve — tested above)."""
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    d = _seed_tree(tmp_path, 'tinker1')
    p = _fake_params(with_rotvecs=True)
    ap.apply_calibration(p, basic_root=tmp_path, allow_flipped_camera=True)
    text = (d / 'pan_tilt_overrides.xacro').read_text()
    assert 'value="0.1 0.2 0.3"' not in text       # attach_rpy overwritten
    assert 'value="-3.1 -0.8 -0.01"' not in text   # camera_mount_rpy overwritten
    assert f'value="{ap._pan_axis_rpy_str(p["t_a_rotvec"])}"' in text
    assert f'value="{ap._rotvec_to_rpy_str(p["t_b_rotvec"], True)}"' in text
    assert f'value="{ap._fmt_triplet(p["t_a"])}"' in text
    assert f'value="{ap._fmt_triplet(p["t_b_trans"])}"' in text


# ---- ported: CLI main() lockstep + idempotency (was
#      test_calibration.py::test_apply_to_urdf_main_*) ------------------------


def _forward_params():
    """Forward-facing camera rotation so the default (no override flag) CLI
    path passes the yaw guard."""
    p = _fake_params(with_rotvecs=False)
    p['t_b_rotvec'] = [0.0, 0.0, -0.05]
    return p


def test_cli_main_writes_both_files_with_backups(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    d = _seed_tree(tmp_path, 'tinker1')
    results = tmp_path / 'polish.json'
    results.write_text(json.dumps({'params': _forward_params()}))

    ap.main(['--results', str(results), '--basic-root', str(tmp_path)])

    text = (d / 'pan_tilt_overrides.xacro').read_text()
    assert 'value="-0.3 0.007 1.37"' in text
    out = (d / 'offsets.yaml').read_text()
    assert 'pan_offset_rad: 3.0866366614' in out
    # Both `.old-<ts>` backups present, carrying the pre-apply content.
    xacro_backups = list(d.glob('pan_tilt_overrides.xacro.old-*'))
    yaml_backups = list(d.glob('offsets.yaml.old-*'))
    assert len(xacro_backups) == 1 and len(yaml_backups) == 1
    assert 'value="1 2 3"' in xacro_backups[0].read_text()
    assert 'pan_offset_rad: 0.0' in yaml_backups[0].read_text()
    # No tmp files left behind.
    assert not list(d.glob('*.tmp-*'))


def test_cli_main_idempotent_second_run(tmp_path, monkeypatch):
    monkeypatch.setenv('ROBOT_NAME', 'tinker1')
    d = _seed_tree(tmp_path, 'tinker1')
    results = tmp_path / 'polish.json'
    results.write_text(json.dumps({'params': _forward_params()}))
    argv = ['--results', str(results), '--basic-root', str(tmp_path)]

    ap.main(argv)
    time.sleep(1.05)   # a second run would get a distinct backup timestamp
    ap.main(argv)

    assert len(list(d.glob('pan_tilt_overrides.xacro.old-*'))) == 1
    assert len(list(d.glob('offsets.yaml.old-*'))) == 1


def test_cli_main_refuses_without_robot_name(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('ROBOT_NAME', raising=False)
    results = tmp_path / 'polish.json'
    results.write_text(json.dumps({'params': _forward_params()}))
    with pytest.raises(SystemExit) as exc_info:
        ap.main(['--results', str(results), '--basic-root', str(tmp_path)])
    assert exc_info.value.code == 2
    assert 'ROBOT_NAME' in capsys.readouterr().err
