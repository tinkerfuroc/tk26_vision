"""SE(3) helpers (numpy + scipy only). Parameterization is [rotvec(3), trans(3)]."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def T_from_vec(v6):
    v6 = np.asarray(v6, float)
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(v6[:3]).as_matrix()
    T[:3, 3] = v6[3:]
    return T


def vec_from_T(T):
    rotvec = R.from_matrix(np.asarray(T)[:3, :3]).as_rotvec()
    return np.concatenate([rotvec, np.asarray(T)[:3, 3]])


def T_from_Rt(Rm, t):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert(T):
    Rm = np.asarray(T)[:3, :3]
    t = np.asarray(T)[:3, 3]
    out = np.eye(4)
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ t
    return out


def se3_average(Ts):
    """Approximate SE(3) mean: normalized quaternion average of rotations + arithmetic
    mean of translations. Valid for small angular scatter (the hand-eye repeat-sample
    case); not a true Fréchet mean for wide angular spreads."""
    Ts = [np.asarray(T) for T in Ts]
    if not Ts:
        raise ValueError("se3_average requires at least one transform")
    quats = R.from_matrix([T[:3, :3] for T in Ts]).as_quat()
    # sign-align quaternions to the first to avoid cancellation
    ref = quats[0]
    quats = np.array([q if np.dot(q, ref) >= 0 else -q for q in quats])
    mean_q = quats.mean(axis=0)
    mean_q /= np.linalg.norm(mean_q)
    out = np.eye(4)
    out[:3, :3] = R.from_quat(mean_q).as_matrix()
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def rotation_angle_deg(R1, R2):
    Rrel = np.asarray(R1).T @ np.asarray(R2)
    return float(np.degrees(R.from_matrix(Rrel).magnitude()))


def rigid_3d_3d(src, dst):
    """Least-squares rigid transform ``T`` (4x4) mapping ``src`` onto ``dst``.

    Solves the no-scale Kabsch/Umeyama problem: returns ``T`` minimizing
    ``sum_i || (R @ src_i + t) - dst_i ||²`` over proper rotations ``R`` (det
    +1) and translations ``t``. ``src`` and ``dst`` are ``(N, 3)`` arrays of
    corresponding points.

    Standalone utility for the FFS-depth work (NOT currently on the solve path —
    the solver injects depth as a soft 3D residual in the bundle adjust, not a
    per-view 3D-3D fit). Intended use: ``src`` = board-frame ChArUco corner model
    points, ``dst`` = the FFS-deprojected metric camera-frame points, giving a
    depth-only ``T_cam_board`` whose optical-axis translation is *measured*
    rather than inferred by monocular PnP — useful as a per-view depth-vs-PnP
    disagreement check or a metric seed if wired in later.

    The ``np.diag([1, 1, sign(det(V Uᵀ))])`` correction forbids a reflection
    when the point set is planar (the ChArUco board is exactly z=0 in board
    frame) plus noise — the classic Kabsch SVD trap that would otherwise flip
    handedness and corrupt the pose.
    """
    src = np.asarray(src, float).reshape(-1, 3)
    dst = np.asarray(dst, float).reshape(-1, 3)
    if src.shape != dst.shape or len(src) < 3:
        raise ValueError(
            f"rigid_3d_3d needs >=3 matched points, got src{src.shape} dst{dst.shape}")
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = mu_d - Rm @ mu_s
    return T_from_Rt(Rm, t)
