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
