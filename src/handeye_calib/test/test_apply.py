import numpy as np
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
