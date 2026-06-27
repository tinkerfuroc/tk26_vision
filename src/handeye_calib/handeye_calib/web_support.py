"""Pure, ROS-free helpers + inline UI for handeye_web. No rclpy/fastapi here."""
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def json_safe(obj):
    """Recursively replace non-finite floats (NaN/Inf) with ``None``.

    Starlette's ``JSONResponse.render`` calls ``json.dumps(..., allow_nan=False)``
    which raises ``ValueError`` on NaN/Inf, and FastAPI then surfaces a
    plain-text ``Internal Server Error`` 500 — the browser's ``JSON.parse``
    chokes on that body at "line 1 column 1". Scrub at the response boundary
    so a stray non-finite (degenerate solve, empty-stat divide-by-zero, etc.)
    renders as a blank field in the UI instead of breaking the whole response.
    Numpy scalars are coerced to Python floats so ``json.dumps`` handles them.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def tf_to_matrix(translation_xyz, quaternion_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quaternion_xyzw, float)).as_matrix()
    T[:3, 3] = np.asarray(translation_xyz, float)
    return T


def matrix_to_xyz_rpy(T):
    T = np.asarray(T, float)
    xyz = T[:3, 3].tolist()
    rpy = R.from_matrix(T[:3, :3]).as_euler('xyz').tolist()  # URDF fixed-axis convention
    return xyz, rpy


def charuco_to_sample_arrays(charuco_corners, charuco_ids):
    px = np.asarray(charuco_corners, float).reshape(-1, 2)
    idx = np.asarray(charuco_ids).reshape(-1).astype(int)
    return px, idx


_GATE_COLORS = {"PASS": "#1a9850", "WARN": "#f59e0b", "FAIL": "#d73027"}


def gate_color(status):
    return _GATE_COLORS.get(status, "#888888")


def state_payload(camera_connected, intrinsics_ok, num_samples, last_detection, status_msg):
    return {
        "camera_connected": bool(camera_connected),
        "intrinsics_ok": bool(intrinsics_ok),
        "num_samples": int(num_samples),
        "last_detection": last_detection,
        "status_msg": status_msg,
    }


def solve_payload(res):
    xyz, rpy = matrix_to_xyz_rpy(res.X)
    return {
        "status": res.status,
        "X_xyz": xyz,
        "X_rpy": rpy,
        "heldout_metrics": res.heldout_metrics,
        "train_metrics": res.train_metrics,
    }


def _metrics_mm_deg(m):
    """Convert a ``{trans_rmse_m, rot_rmse_rad, reproj_px}`` block into the
    mm/deg-rendered shape the Solve tab displays directly.

    When the block carries the FFS-depth-grounded metric
    (``depth_point_rmse_mm`` / ``n_depth_corners``), they are passed through so
    the UI can show the honest metric real-world accuracy next to the
    reprojection number. They are *absent* (never a fake ``0``) for a
    monocular-only solve, so the UI can branch on presence."""
    if not m:
        return {}
    out = {
        "trans_rmse_mm": mm(m.get("trans_rmse_m", 0.0)),
        "rot_rmse_deg": deg(m.get("rot_rmse_rad", 0.0)),
        "reproj_px": float(m.get("reproj_px", 0.0)),
    }
    if m.get("depth_point_rmse_mm") is not None:
        out["depth_point_rmse_mm"] = round(float(m["depth_point_rmse_mm"]), 4)
        out["n_depth_corners"] = int(m.get("n_depth_corners", 0))
    return out


def solve_payload_v2(res, samples, K, dist, board_pts):
    """v2 solve payload for the Solve tab.

    Adds, on top of :func:`solve_payload`:

    * ``X_xyz_mm`` / ``X_rpy_deg``: ready-to-render mm/deg quantities so the JS
      doesn't have to know about radians or metres.
    * ``train_metrics_mm_deg`` / ``heldout_metrics_mm_deg``: same conversion
      applied to the metric blocks (trans in mm, rot in deg, reproj_px unchanged).
    * ``per_method_summary``: a compact ``[{name, reproj_px}, ...]`` projection of
      ``res.per_method`` so the method-comparison table doesn't carry full 4x4s
      across the wire.
    * ``per_sample_reproj_px``: a list[float] aligned 1:1 with ``samples`` giving
      the post-BA reprojection RMS for each sample — feeds the histogram /
      scatter canvases. Empty list when ``samples`` is empty (smoke-test path).

    The original ``train_metrics`` / ``heldout_metrics`` / ``X_xyz`` / ``X_rpy``
    keys remain so callers can still inspect raw SI units; the ``*_mm_deg``
    keys are additive.
    """
    base = solve_payload(res)
    xyz_m, rpy_rad = matrix_to_xyz_rpy(res.X)
    if samples is not None and len(samples) > 0:
        from handeye_calib import handeye_solve as hs  # local import to keep ws ROS-free
        _, per_sample = hs._reproj_rms(
            res.X, res.Tbb, samples, K, dist, board_pts, per_sample=True)
    else:
        per_sample = []
    base.update({
        "X_xyz_mm": [mm(v) for v in xyz_m],
        "X_rpy_deg": [deg(v) for v in rpy_rad],
        "train_metrics_mm_deg": _metrics_mm_deg(res.train_metrics),
        "heldout_metrics_mm_deg": _metrics_mm_deg(res.heldout_metrics),
        "per_method_summary": [
            {"name": str(m.get("name", "?")),
             "reproj_px": float(m.get("reproj_px", 0.0))}
            for m in (res.per_method or [])
        ],
        "per_sample_reproj_px": [float(v) for v in per_sample],
    })
    return base


def encode_jpeg(bgr):
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def placeholder_jpeg(text="no camera", size=(480, 640)):
    img = np.full((size[0], size[1], 3), 40, np.uint8)
    cv2.putText(img, text, (20, size[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (200, 200, 200), 2, cv2.LINE_AA)
    return encode_jpeg(img)


def draw_charuco_overlay(bgr, corners_xy, ids=None, rms_px=None, image_topic=None):
    """Render the detection overlay used by the live frame panel.

    - Green dot at every corner (legacy behaviour, always on).
    - When ``ids`` is provided, render the corresponding integer next to each
      corner (cv2.putText, small cyan text so the green dot stays readable).
    - When ``rms_px`` and/or ``image_topic`` are provided, render a translucent
      header strip across the top with the relevant diagnostics ("rms=X.XXpx"
      / "topic=..."). The header is rendered LAST so it stays legible regardless
      of corner density.

    Shape (HxWxC) is preserved — IDs and the header strip are drawn in-place,
    no resize. The 960 px bandwidth downscale lives in ``HandeyeWebNode.latest_jpeg``
    (matches pan_tilt's ``_downscale``) so callers asking for the overlay always
    get the source resolution.
    """
    out = bgr.copy()
    corners = np.asarray(corners_xy, float).reshape(-1, 2)
    # Green dots first so the IDs (if any) overlay them.
    for (x, y) in corners:
        cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1, cv2.LINE_AA)

    if ids is not None and len(corners) > 0:
        ids_flat = np.asarray(ids).reshape(-1).astype(int)
        for (x, y), cid in zip(corners, ids_flat):
            cv2.putText(out, str(int(cid)),
                        (int(round(x)) + 6, int(round(y)) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # Header bar (top strip) — only when there's something to put in it.
    header_parts = []
    if rms_px is not None:
        try:
            header_parts.append(f"rms={float(rms_px):.2f}px")
        except (TypeError, ValueError):
            pass
    if image_topic:
        header_parts.append(f"topic={image_topic}")
    if header_parts:
        text = "  ".join(header_parts)
        h, w = out.shape[:2]
        bar_h = 18
        # Translucent dark bar (blend with existing pixels so it doesn't wipe the frame).
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, dst=out)
        cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def mm(x_m):
    return round(float(x_m) * 1000.0, 4)


def deg(x_rad):
    import math
    return round(float(x_rad) * 180.0 / math.pi, 4)


# ---------------------------------------------------------------------------
# T4: diversity meter + per-sample metadata
# ---------------------------------------------------------------------------

def _rotation_angle_deg(R_a, R_b):
    """Geodesic angle (deg) between two 3x3 rotation matrices.

    Uses the trace formula ``arccos((tr(R_a.T R_b) - 1) / 2)`` with a
    clip to dodge numerical overshoot. Pure numpy — no scipy dep on the
    diversity hot path. Returns 0.0 when either input is degenerate
    (numerical noise on small angles is fine).
    """
    Ra = np.asarray(R_a, float).reshape(3, 3)
    Rb = np.asarray(R_b, float).reshape(3, 3)
    M = Ra.T @ Rb
    tr = float(M[0, 0] + M[1, 1] + M[2, 2])
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return float(np.degrees(np.arccos(c)))


def compute_diversity_deg(samples):
    """Max pairwise rotation-angle (deg) between any two ``T_base_eef[:3,:3]``.

    Operates on a list of :class:`handeye_calib.handeye_model.Sample`.
    Returns ``0.0`` when there are fewer than two samples (no pair).
    """
    if not samples or len(samples) < 2:
        return 0.0
    Rs = [np.asarray(s.T_base_eef, float)[:3, :3] for s in samples]
    best = 0.0
    n = len(Rs)
    for i in range(n):
        for j in range(i + 1, n):
            a = _rotation_angle_deg(Rs[i], Rs[j])
            if a > best:
                best = a
    return float(best)


def sample_metadata(idx, sample, prev_sample=None, *,
                    n_corners=None, reproj_px=None, area_frac=None,
                    joint_positions=None, ts=None, depth_source=None):
    """JSON-friendly per-sample dict for the Capture-tab gallery row.

    Composed off the accepted :class:`Sample` plus the original
    capture-time scalars (which aren't stored on the dataclass).
    ``angular_delta_deg`` is the rotation between this sample's
    ``T_base_eef[:3,:3]`` and ``prev_sample.T_base_eef[:3,:3]`` (or
    ``None`` for the first sample). ``depth_source`` records whether this
    sample carried FFS metric depth ('ffs') or fell back to monocular
    ('unavailable' / 'shape-mismatch' / 'ffs-too-sparse' / 'moved-during-ffs'
    / 'disabled'), so the gallery can show per-sample depth provenance.
    """
    if prev_sample is None:
        ang = None
    else:
        ang = _rotation_angle_deg(
            np.asarray(prev_sample.T_base_eef, float)[:3, :3],
            np.asarray(sample.T_base_eef, float)[:3, :3],
        )
    return {
        "idx": int(idx),
        "n_corners": (None if n_corners is None else int(n_corners)),
        "reproj_px": (None if reproj_px is None else float(reproj_px)),
        "area_frac": (None if area_frac is None else float(area_frac)),
        "angular_delta_deg": (None if ang is None else float(ang)),
        "joint_positions": (None if joint_positions is None
                            else [float(j) for j in joint_positions]),
        "ts": (None if ts is None else float(ts)),
        "depth_source": (None if depth_source is None else str(depth_source)),
    }


def waypoint_metadata(idx: int, joints_rad) -> dict:
    """JSON-friendly per-waypoint dict for the Waypoints tab.

    Keys:
      ``idx``       — integer index in the store.
      ``joints_rad``— list of 7 floats (radians).
      ``abbrev``    — human-readable summary: first 3 joints rounded to 2 dp,
                      followed by "…" (e.g. ``"0.42, -0.30, 1.57, …"``).
    """
    j = [float(v) for v in joints_rad]
    abbrev = ", ".join(f"{v:.2f}" for v in j[:3]) + ", …"
    return {
        "idx": int(idx),
        "joints_rad": j,
        "abbrev": abbrev,
    }


SEQUENCE_IDLE_DEFAULT = {
    "running": False,
    "dry_run": False,
    "current_idx": None,
    "total": 0,
    "current_step": "idle",
    "log": [],
}


def enriched_state_payload(*, camera_connected, intrinsics_ok, num_samples,
                           last_detection, status_msg,
                           frame_count, frame_hz, frame_age_sec,
                           image_topic, ros_domain_id,
                           t_base_ee, xarm_joint_positions,
                           board, safety_envelope,
                           stability, samples, diversity, last_solve,
                           safety_preview=None,
                           waypoints=None,
                           sequence=None):
    """v2 enriched state for the WebSocket push.

    Extends ``state_payload`` with everything the new static UI needs to
    render the info / move / capture / solve / promote tabs. T1 wires every
    field; T3 adds ``safety_preview`` (server-evaluated SafetyEnvelope check
    against the cached EE pose, ``{safe: bool|None, detail: str}``) so the
    Move tab doesn't have to duplicate the safety math in JS; T4/T5 populate
    ``samples`` and ``last_solve``.

    ``safety_preview`` defaults to ``None`` for back-compat with T1/T2
    callers; the field is always emitted (under that key) so the UI can branch
    on its presence consistently.

    ``sequence`` defaults to ``SEQUENCE_IDLE_DEFAULT`` when omitted (T3-seq
    auto-capture runner state); always emitted under the ``sequence`` key so
    the UI's renderer can branch on ``state.sequence.running`` unconditionally.
    """
    base = state_payload(camera_connected, intrinsics_ok, num_samples,
                         last_detection, status_msg)
    base.update({
        "frame_count": int(frame_count),
        "frame_hz": float(frame_hz),
        "frame_age_sec": (None if frame_age_sec is None else float(frame_age_sec)),
        "image_topic": str(image_topic),
        "ros_domain_id": int(ros_domain_id),
        "t_base_ee": (None if t_base_ee is None else
                      [list(map(float, row)) for row in t_base_ee]),
        "xarm_joint_positions": (None if xarm_joint_positions is None else
                                 [float(j) for j in xarm_joint_positions]),
        "board": dict(board),
        "safety_envelope": dict(safety_envelope),
        "stability": dict(stability),
        "samples": list(samples),
        "diversity": dict(diversity),
        "last_solve": last_solve,
        "safety_preview": (None if safety_preview is None else dict(safety_preview)),
        "waypoints": list(waypoints) if waypoints is not None else [],
        "sequence": (dict(sequence) if sequence is not None
                     else dict(SEQUENCE_IDLE_DEFAULT)),
    })
    return base
