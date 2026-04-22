import numpy as np
from scipy.spatial.transform import Rotation

def transform_matrix(rotvec, trans):
    """Convert rotation vector + translation to 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = trans
    return T

def rotation_about_z(theta):
    """Rotate around Z axis (pan)."""
    R = Rotation.from_euler('z', -theta).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    return T

def rotation_about_x(theta):
    """Rotate around X axis (tilt)."""
    R = Rotation.from_euler('x', theta).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    return T

def rotation_about_y(theta):
    """Rotate around Y axis (tilt)."""
    R = Rotation.from_euler('y', theta).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    return T

def forward_kinematics_fixed(theta_pan, theta_tilt, params, L_pan=0.1, L_tilt=0.1):
    """
    Compute T_base_camera using known kinematics for pan/tilt.
    - L_pan: distance from pan axis to tilt axis
    - L_tilt: distance from tilt axis to camera mount
    - params: 12D: [r0(3), p0(3), r2(3), p2(3)]
    """

    # Estimated unknowns
    r0 = params[0:3]  # base → pan_base
    p0 = params[3:6]
    r2 = params[6:9]  # tilt_end → camera_link
    p2 = params[9:12]

    # Known fixed kinematics
    T0 = transform_matrix(r0, p0)
    R_pan = rotation_about_z(theta_pan)
    T1 = np.eye(4)
    T1[:3, 3] = [0, 0, L_pan]
    R_tilt = rotation_about_y(theta_tilt)
    T2 = np.eye(4)
    T2[:3, 3] = [L_tilt, 0, 0]
    T3 = transform_matrix(r2, p2)

    # Chain together
    T = T0 @ R_pan @ T1 @ R_tilt @ T2 @ T3
    # T = T0 @ R_pan @ T1 @ R_tilt @ T3
    return T

