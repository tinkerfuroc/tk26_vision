from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from handeye_calib import transforms as tf
from handeye_calib import apply_handeye as ah


def test_compose_eef_to_mount_roundtrip():
    # Known internal chain mount->color_optical; recover eef->mount from eef->color_optical.
    T_mount_color = tf.T_from_Rt(R.from_euler('xyz', [-90, 0, -90], degrees=True).as_matrix(),
                                 [0.0, 0.015, 0.0])
    T_eef_mount_true = tf.T_from_vec(np.array([0.1, -0.2, 0.05, 0.07, -0.018, 0.024]))
    T_eef_color = T_eef_mount_true @ T_mount_color
    T_eef_mount = ah.compose_eef_to_mount(T_eef_color, T_mount_color)
    np.testing.assert_allclose(T_eef_mount, T_eef_mount_true, atol=1e-9)


def test_yaml_dict_has_required_fields():
    T = tf.T_from_vec(np.array([0.0, 0.0, 0.0, 0.06, -0.01, 0.02]))
    d = ah.handeye_yaml_dict(T_eef_mount=T, T_eef_color=T, num_poses=18,
                             metrics={"trans_rmse_m": 0.002, "rot_rmse_rad": 0.005,
                                      "reproj_px": 0.9}, date="2026-06-15")
    he = d["hand_eye"]
    assert he["reference_frame"] == "link_eef"
    assert he["camera_frame"] == "xarm_camera_link"
    assert len(he["arm_to_camera_xyz"].split()) == 3
    assert len(he["arm_to_camera_rpy"].split()) == 3
    assert he["num_poses"] == 18


def test_patch_urdf_origin(tmp_path):
    xacro = (
        '<robot>\n'
        '  <joint name="xarm_camera_joint" type="fixed">\n'
        '    <origin xyz="0.06746 -0.0175 0.0237" rpy="3.14159 -1.5708 0"/>\n'
        '    <parent link="link_eef"/>\n'
        '    <child link="xarm_camera_link"/>\n'
        '  </joint>\n'
        '</robot>\n'
    )
    new = ah.patch_urdf_origin(xacro, "xarm_camera_joint",
                               xyz=(0.1, 0.2, 0.3), rpy=(0.0, 0.0, 0.0))
    assert 'xyz="0.1 0.2 0.3"' in new
    assert 'rpy="0.0 0.0 0.0"' in new
    assert '0.06746' not in new           # old value replaced
    assert new.count("<joint") == 1       # only the targeted joint touched


# ---------------------------------------------------------------------------
# T6: per-robot xacro override resolver + seed template + d435i-shape patch
# ---------------------------------------------------------------------------

def test_resolve_robot_xacro_path_for_tinker2(tmp_path):
    # synthesize a basic-repo-shaped fixture
    (tmp_path / "src/tinker_robot_config/robots/tinker2").mkdir(parents=True)
    p = ah.resolve_robot_xacro_path("tinker2", tmp_path)
    assert p == tmp_path / "src/tinker_robot_config/robots/tinker2/wrist_camera.xacro"


def test_resolve_robot_xacro_path_for_tinker1(tmp_path):
    (tmp_path / "src/tinker_robot_config/robots/tinker1").mkdir(parents=True)
    p = ah.resolve_robot_xacro_path("tinker1", tmp_path)
    assert p == tmp_path / "src/tinker_robot_config/robots/tinker1/wrist_camera.xacro"


def test_resolve_robot_xacro_path_none_when_robot_unset(tmp_path):
    assert ah.resolve_robot_xacro_path(None, tmp_path) is None
    assert ah.resolve_robot_xacro_path("", tmp_path) is None


def test_override_xacro_is_property_form():
    """The vendor d435i xacro consumes handeye_xyz/handeye_rpy PROPERTIES
    (redefined by the per-robot include), not a <joint> block — the
    pre-2026-07-03 <joint> writer silently never took effect on the real
    URDF. Promote must emit the property-redefinition form."""
    text = ah.seed_handeye_override_xacro(
        'tinker1', '0.1 0.2 0.3', '0.4 0.5 0.6')
    assert '<xacro:property name="handeye_xyz" value="0.1 0.2 0.3"/>' in text
    assert '<xacro:property name="handeye_rpy" value="0.4 0.5 0.6"/>' in text
    assert '<joint' not in text
    assert 'handeye_override_tinker1' in text
    import xml.etree.ElementTree as ET
    ET.fromstring(text)


def test_patch_urdf_origin_against_realsense_d435i_shape():
    sample = ('<robot><joint name="camera_link_joint" type="fixed">\n'
              '  <parent link="link_eef"/><child link="camera_link"/>\n'
              '  <origin xyz="0.06746 -0.0175 0.0237" rpy="3.14 -1.57 0"/>\n'
              '</joint></robot>\n')
    patched = ah.patch_urdf_origin(sample, "camera_link_joint",
                                    [0.08, -0.01, 0.02], [3.1, -1.6, 0.0])
    assert 'xyz="0.08 -0.01 0.02"' in patched
    assert 'rpy="3.1 -1.6 0.0"' in patched
    assert 'xyz="0.06746' not in patched


def test_patch_urdf_origin_raises_valueerror_on_missing_joint():
    """When the named joint isn't in the override file, raise ValueError so the
    web layer can fall back to ``seed_handeye_override_xacro`` (mode='seed')."""
    sample = '<robot><joint name="some_other_joint" type="fixed"><origin xyz="0 0 0" rpy="0 0 0"/></joint></robot>'
    with pytest.raises(ValueError):
        ah.patch_urdf_origin(sample, "camera_link_joint", [0, 0, 0], [0, 0, 0])
