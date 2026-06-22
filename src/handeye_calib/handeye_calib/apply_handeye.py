"""Compose the solved transform into the URDF mount frame and persist it.

Solver outputs T_eef_color (link_eef -> xarm_camera_color_optical_frame). The URDF
attaches xarm_camera_link to link_eef; color_optical is a fixed child of camera_link.
So the URDF mount-joint origin we must write is:
    T_eef_mount = T_eef_color @ inv(T_mount_color)
where T_mount_color is the (factory, unchanged) camera_link->color_optical chain.
"""
import re
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from handeye_calib import transforms as tf


def compose_eef_to_mount(T_eef_color, T_mount_color):
    return np.asarray(T_eef_color) @ tf.invert(T_mount_color)


def _xyz_rpy(T):
    xyz = np.asarray(T)[:3, 3]
    rpy = R.from_matrix(np.asarray(T)[:3, :3]).as_euler('xyz')
    return " ".join(f"{v:.9g}" for v in xyz), " ".join(f"{v:.9g}" for v in rpy)


def handeye_yaml_dict(T_eef_mount, T_eef_color, num_poses, metrics, date,
                      square_len_m=0.04):
    mount_xyz, mount_rpy = _xyz_rpy(T_eef_mount)
    color_xyz, color_rpy = _xyz_rpy(T_eef_color)
    return {"hand_eye": {
        "reference_frame": "link_eef",
        "camera_frame": "xarm_camera_link",
        "arm_to_camera_xyz": mount_xyz,
        "arm_to_camera_rpy": mount_rpy,
        "color_optical_xyz": color_xyz,
        "color_optical_rpy": color_rpy,
        "calibration_date": date,
        "calibration_method": "calibrateHandEye+BA",
        "board": {"type": "charuco", "squares": "5x5", "square_len_m": square_len_m},
        "num_poses": int(num_poses),
        "heldout_trans_rmse_m": round(float(metrics["trans_rmse_m"]), 6),
        "heldout_rot_rmse_rad": round(float(metrics["rot_rmse_rad"]), 6),
        "heldout_reproj_px": round(float(metrics["reproj_px"]), 4),
    }}


def patch_urdf_origin(xacro_text, joint_name, xyz, rpy):
    """Replace the <origin .../> inside the named <joint>, leaving everything else intact."""
    xyz_s = " ".join(str(v) for v in xyz)
    rpy_s = " ".join(str(v) for v in rpy)
    pat = re.compile(
        r'(<joint\s+name="' + re.escape(joint_name) + r'".*?<origin\b)[^>]*?(/?>)',
        re.DOTALL)
    if not pat.search(xacro_text):
        raise ValueError(f"origin for joint {joint_name} not found")
    return pat.sub(rf'\1 xyz="{xyz_s}" rpy="{rpy_s}"\2', xacro_text, count=1)


def write_with_backup(path, text):
    import os
    if os.path.exists(path):
        backup = f"{path}.old-{time.strftime('%Y%m%dT%H%M%S')}"
        os.replace(path, backup)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def main():
    raise SystemExit("apply_handeye is used as a library by handeye_web; see README.")
