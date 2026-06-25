"""Forward kinematics for the pan-tilt head.

Chain (body-frame convention: x-forward, y-left, z-up at every link):

    base_link
      |  T_A  (translation-only; rotation locked identity per plan)
      v
    pan_axis
      |  R_z(-(theta_p + theta_p_off))    # firmware "+pan = right" => negative Z rotation
      |  Trans_z(L_pan)
      v
    tilt_axis
      |  R_y(theta_t + theta_t_off)       # firmware "+tilt = up"  => positive Y rotation
      v
    tilt_end
      |  T_B  (6-DOF; translation primary, rotation near identity)
      v
    head_camera_link (body frame)

L_tilt is intentionally absorbed into T_B translation (see plan Parameter vector);
leaving it as a separate fixed length was a redundant reparameterization.

The optimizer owns 13 DOF by default:
    T_A translation        (3)
    T_B translation        (3)
    T_B rotation rotvec    (3)  <- init identity, locked identity for baseline fit
    T_ee_marker            (6)  <- only used by the joint / polish phase
    theta_t_offset         (1)
    theta_p_offset         (1)

but `chain`-phase fits only the pan-tilt-side subset (T_A_trans, T_B_trans, theta_t_off,
optional theta_p_off) with T_B rotation frozen at identity.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation


# ---- primitive transforms ---------------------------------------------------

def transform_matrix(rotvec, trans):
    """rotvec + translation -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = trans
    return T


def rotation_about_z(theta):
    """Rotate about +Z by `-theta` (firmware pan convention: +pan = right)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', -theta).as_matrix()
    return T


def rotation_about_x(theta):
    """Rotate about +X by `+theta`. Kept for legacy callers; not used in the new chain."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('x', theta).as_matrix()
    return T


def rotation_about_y(theta):
    """Rotate about +Y by `+theta` (firmware tilt convention: +tilt = up)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('y', theta).as_matrix()
    return T


def translation(xyz):
    T = np.eye(4)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


# ---- parameter container ----------------------------------------------------

@dataclass
class PanTiltParams:
    """Calibration parameter block."""

    # base_link -> pan_axis translation.
    t_a: np.ndarray = field(default_factory=lambda: np.array([-0.2754, -0.0134, 1.5459]))
    # base_link -> pan_axis rotation (rotvec). Default identity: historically the
    # pan axis was assumed to be exactly base +Z. A physically non-vertical pan
    # axis (this head sits ~3 deg off +Z about base-Y) cannot be absorbed by
    # T_B (post-tilt) or theta_t_offset, so it shows up as a ~1.3 deg chain
    # rotation floor. Fitting the X/Y components of this rotvec (the
    # `fit_pan_axis_tilt` path) collapses that floor. The Z component is left at
    # zero on purpose: a yaw of T_A is degenerate with theta_p_offset.
    t_a_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # tilt_end -> head_camera_link (body frame).
    t_b_trans: np.ndarray = field(default_factory=lambda: np.array([-0.0724, -0.009, 0.075]))
    # Rotvec for T_B rotation. Default identity. Locked during chain-phase; optionally freed
    # in polish phase if residuals show rotation structure.
    t_b_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # xArm end-effector -> ChArUco board origin. Solved in hand-eye phase.
    t_ee_marker_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    t_ee_marker_trans: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Firmware-reported angles have constant offsets from physical angles.
    # theta_t_off ~= -pi/6 because the T:502 zero-set parked the mechanism looking 30 deg down.
    theta_t_offset: float = -np.pi / 6.0
    theta_p_offset: float = 0.0

    # Fixed geometry (not optimized).
    l_pan: float = 0.135  # tilt axis height above pan axis (URDF tilt_joint xyz z)

    def t_b(self) -> np.ndarray:
        return transform_matrix(self.t_b_rotvec, self.t_b_trans)

    def t_ee_marker(self) -> np.ndarray:
        return transform_matrix(self.t_ee_marker_rotvec, self.t_ee_marker_trans)


def forward_kinematics(
    theta_pan: float,
    theta_tilt: float,
    params: PanTiltParams,
) -> np.ndarray:
    """T_base_link_to_head_camera_link for firmware-reported (theta_p, theta_t) in **radians**."""
    # T_a carries the base->pan-axis translation AND an optional small rotation
    # (params.t_a_rotvec). When t_a_rotvec is zero this is identical to a pure
    # translation, so legacy behavior is unchanged.
    T_a = transform_matrix(params.t_a_rotvec, params.t_a)
    R_pan = rotation_about_z(theta_pan + params.theta_p_offset)
    T_lp = translation([0.0, 0.0, params.l_pan])
    R_tilt = rotation_about_y(theta_tilt + params.theta_t_offset)
    T_b = params.t_b()
    return T_a @ R_pan @ T_lp @ R_tilt @ T_b


# ---- parameter packing (for scipy.optimize.least_squares) -------------------
#
# Two packings:
#   - "chain"  (7 or 8 params): T_A_trans (3) + T_B_trans (3) + theta_t_off [+ theta_p_off]
#   - "joint"  (13 or 16)     : above + T_ee_marker (6) + T_B_rotvec (3 if unlocked)

def pack_chain(
    params: PanTiltParams,
    fit_pan_offset: bool = False,
    fit_tb_rotation: bool = True,
    fit_pan_axis_tilt: bool = False,
) -> np.ndarray:
    flat = [
        params.t_a,
        params.t_b_trans,
        np.array([params.theta_t_offset]),
    ]
    if fit_pan_offset:
        flat.append(np.array([params.theta_p_offset]))
    if fit_tb_rotation:
        flat.append(params.t_b_rotvec)
    # Pan-axis tilt is appended LAST (after t_b_rotvec) so the index of every
    # pre-existing parameter — crucially t_b_rotvec[Z], whose position the
    # bounds code computes via _t_b_rotvec_z_index_chain — is unchanged when
    # this flag is off. Only the X/Y rotvec components are fit; Z stays 0
    # (degenerate with theta_p_offset).
    if fit_pan_axis_tilt:
        flat.append(np.asarray(params.t_a_rotvec, dtype=float)[:2])
    return np.concatenate(flat)


def unpack_chain(
    x: np.ndarray,
    template: PanTiltParams,
    fit_pan_offset: bool = False,
    fit_tb_rotation: bool = True,
    fit_pan_axis_tilt: bool = False,
) -> PanTiltParams:
    offset = 0

    def take(n):
        nonlocal offset
        out = x[offset:offset + n].copy()
        offset += n
        return out

    t_a = take(3)
    t_b_trans = take(3)
    theta_t = float(take(1)[0])
    theta_p = float(take(1)[0]) if fit_pan_offset else template.theta_p_offset
    t_b_rotvec = take(3) if fit_tb_rotation else template.t_b_rotvec.copy()
    if fit_pan_axis_tilt:
        rx, ry = take(2)
        t_a_rotvec = np.array([rx, ry, 0.0])
    else:
        t_a_rotvec = np.asarray(template.t_a_rotvec, dtype=float).copy()

    return PanTiltParams(
        t_a=t_a,
        t_a_rotvec=t_a_rotvec,
        t_b_trans=t_b_trans,
        t_b_rotvec=t_b_rotvec,
        t_ee_marker_rotvec=template.t_ee_marker_rotvec.copy(),
        t_ee_marker_trans=template.t_ee_marker_trans.copy(),
        theta_t_offset=theta_t,
        theta_p_offset=theta_p,
        l_pan=template.l_pan,
    )


def pack_joint(
    params: PanTiltParams,
    fit_tb_rotation: bool = False,
    fit_pan_offset: bool = False,
    fit_pan_axis_tilt: bool = False,
) -> np.ndarray:
    flat = [
        params.t_a,
        params.t_b_trans,
        params.t_ee_marker_rotvec,
        params.t_ee_marker_trans,
        np.array([params.theta_t_offset]),
    ]
    if fit_pan_offset:
        flat.append(np.array([params.theta_p_offset]))
    if fit_tb_rotation:
        flat.append(params.t_b_rotvec)
    # Appended LAST, same rationale as pack_chain: keeps t_b_rotvec[Z]'s index
    # (used by _t_b_rotvec_z_index_joint) stable when this flag is off.
    if fit_pan_axis_tilt:
        flat.append(np.asarray(params.t_a_rotvec, dtype=float)[:2])
    return np.concatenate(flat)


def unpack_joint(
    x: np.ndarray,
    template: PanTiltParams,
    fit_tb_rotation: bool = False,
    fit_pan_offset: bool = False,
    fit_pan_axis_tilt: bool = False,
) -> PanTiltParams:
    offset = 0

    def take(n):
        nonlocal offset
        out = x[offset:offset + n].copy()
        offset += n
        return out

    t_a = take(3)
    t_b_trans = take(3)
    t_ee_rot = take(3)
    t_ee_tr = take(3)
    theta_t = float(take(1)[0])
    theta_p = float(take(1)[0]) if fit_pan_offset else template.theta_p_offset
    t_b_rotvec = take(3) if fit_tb_rotation else template.t_b_rotvec.copy()
    if fit_pan_axis_tilt:
        rx, ry = take(2)
        t_a_rotvec = np.array([rx, ry, 0.0])
    else:
        t_a_rotvec = np.asarray(template.t_a_rotvec, dtype=float).copy()

    return PanTiltParams(
        t_a=t_a,
        t_a_rotvec=t_a_rotvec,
        t_b_trans=t_b_trans,
        t_b_rotvec=t_b_rotvec,
        t_ee_marker_rotvec=t_ee_rot,
        t_ee_marker_trans=t_ee_tr,
        theta_t_offset=theta_t,
        theta_p_offset=theta_p,
        l_pan=template.l_pan,
    )


# ---- back-compat shim -------------------------------------------------------
#
# The old 12-DOF `forward_kinematics_fixed(theta_p, theta_t, params, L_pan, L_tilt)` is still
# referenced by `optimize_kinematics.py` (kept until new code validates). This shim re-expresses
# it through the new model so both scripts see the same FK semantics.

def forward_kinematics_fixed(theta_pan, theta_tilt, params_vec, L_pan=0.135, L_tilt=0.0):
    """Legacy signature. `params_vec` = [r0(3), p0(3), r2(3), p2(3)]; L_tilt merged into p2.x."""
    r0 = np.asarray(params_vec[0:3], dtype=float)
    p0 = np.asarray(params_vec[3:6], dtype=float)
    r2 = np.asarray(params_vec[6:9], dtype=float)
    p2 = np.asarray(params_vec[9:12], dtype=float).copy()
    if L_tilt:
        p2[0] = p2[0] + L_tilt

    # Legacy chain: T_A (6-DOF) @ R_pan @ trans_z(L_pan) @ R_tilt @ T_B.
    # Emulate via explicit construction rather than the new clean FK because the old
    # T_A has a rotation component and the new `translation`-only T_A doesn't.
    T_a = transform_matrix(r0, p0)
    R_pan = rotation_about_z(theta_pan)
    T_lp = translation([0.0, 0.0, L_pan])
    R_tilt = rotation_about_y(theta_tilt)
    T_b = transform_matrix(r2, p2)
    return T_a @ R_pan @ T_lp @ R_tilt @ T_b
