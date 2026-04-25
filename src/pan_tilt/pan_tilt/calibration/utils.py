"""Transform + residual utilities for pan-tilt calibration."""

import numpy as np
from scipy.spatial.transform import Rotation


# ---- basic conversions ------------------------------------------------------

def pose_to_matrix(translation, quaternion):
    """[tx, ty, tz], [qx, qy, qz, qw] -> 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    T[:3, 3] = translation
    return T


def matrix_to_pose(T):
    """4x4 homogeneous -> ([tx, ty, tz], [qx, qy, qz, qw])."""
    trans = T[:3, 3].tolist()
    quat = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()
    return trans, quat


def invert_transform(T):
    """Numerically stable inverse of a rigid transform."""
    Ti = np.eye(4)
    R = T[:3, :3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


# ---- SE(3) log residual -----------------------------------------------------
#
# For nonlinear least-squares we need a 6-vector residual per sample. The usual
# choice is the SE(3) logarithm of the pose error: xi = log(T_pred^-1 @ T_meas)
# which gives [omega (3), v (3)] in local-tangent coordinates. Using the log
# keeps rotation and translation on the same manifold metric and avoids the
# scaling ambiguity of separate "angle + k*trans" sums.

def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) -> axis-angle vector (length = rotation angle in radians)."""
    return Rotation.from_matrix(R).as_rotvec()


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> 6-vector [omega, v] in the Lie algebra se(3).

    Near the identity this reduces to [rotvec, translation]; for larger residuals
    we compute the exact V^-1 * t correction so the residual remains a proper
    manifold tangent vector.
    """
    omega = so3_log(T[:3, :3])
    theta = float(np.linalg.norm(omega))
    t = T[:3, 3]

    if theta < 1e-10:
        v = t
    else:
        axis = omega / theta
        K = _skew(axis)
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        V = np.eye(3) + B * (theta * K) + (1.0 - A) * (K @ K)
        v = np.linalg.solve(V, t)

    return np.concatenate([omega, v])


def se3_log_residual(T_pred: np.ndarray, T_meas: np.ndarray) -> np.ndarray:
    """6-vector residual between predicted and measured SE(3) poses."""
    return se3_log(invert_transform(T_pred) @ T_meas)


def pose_error_scalars(T_pred: np.ndarray, T_meas: np.ndarray) -> tuple[float, float]:
    """(translation_error_meters, rotation_error_radians) for reporting."""
    trans_err = float(np.linalg.norm(T_pred[:3, 3] - T_meas[:3, 3]))
    R_err = T_pred[:3, :3].T @ T_meas[:3, :3]
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.arccos(cos_angle))
    return trans_err, rot_err


# ---- frame convention helpers -----------------------------------------------
#
# REP 103 canonical body <-> optical rotation. Matrix applied to an optical-frame
# *vector* yields its body-frame coordinates. The columns of R_body_from_optical
# are the optical axes expressed in the body frame:
#   body_x =  optical_z  (camera forward)
#   body_y = -optical_x  (camera left)
#   body_z = -optical_y  (camera up)

R_BODY_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def optical_to_body(T_optical: np.ndarray) -> np.ndarray:
    """Re-express a pose measured in the optical frame as a pose in the body frame.

    If `T_optical = T_cam_optical_to_marker`, then
    `R_body_from_optical @ T_optical` applied to both rotation and translation
    yields `T_cam_body_to_marker`.
    """
    T = np.eye(4)
    T[:3, :3] = R_BODY_FROM_OPTICAL @ T_optical[:3, :3]
    T[:3, 3] = R_BODY_FROM_OPTICAL @ T_optical[:3, 3]
    return T


# ---- sample I/O -------------------------------------------------------------
#
# On-disk sample format (JSON):
#   {
#     "theta_pan_rad":  float,  # firmware-reported, already in radians
#     "theta_tilt_rad": float,
#     "t_base_ee": {            # from tf2 lookup base_link -> link_eef
#         "translation": [x, y, z],
#         "rotation":    [qx, qy, qz, qw]
#     },
#     "t_cam_marker_body": {    # ArUco detection, preprocessed to body frame
#         "translation": [x, y, z],
#         "rotation":    [qx, qy, qz, qw]
#     },
#     "image_stamp_ns":   int,
#     "state_stamp_ns":   int,
#     "detection_quality": float
#   }

def sample_to_matrices(sample: dict):
    """Return (theta_p, theta_t, T_base_ee, T_cam_marker_body) from a JSON sample dict."""
    theta_p = float(sample["theta_pan_rad"])
    theta_t = float(sample["theta_tilt_rad"])
    T_be = pose_to_matrix(
        sample["t_base_ee"]["translation"],
        sample["t_base_ee"]["rotation"],
    )
    T_cm = pose_to_matrix(
        sample["t_cam_marker_body"]["translation"],
        sample["t_cam_marker_body"]["rotation"],
    )
    return theta_p, theta_t, T_be, T_cm


def matrix_to_pose_dict(T: np.ndarray) -> dict:
    trans, quat = matrix_to_pose(T)
    return {"translation": list(trans), "rotation": list(quat)}
