"""Pure, ROS-free helpers + inline UI for handeye_web. No rclpy/fastapi here."""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


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


def enriched_state_payload(*, camera_connected, intrinsics_ok, num_samples,
                           last_detection, status_msg,
                           frame_count, frame_hz, frame_age_sec,
                           image_topic, ros_domain_id,
                           t_base_ee, xarm_joint_positions,
                           board, safety_envelope,
                           stability, samples, diversity, last_solve):
    """v2 enriched state for the WebSocket push.

    Extends ``state_payload`` with everything the new static UI needs to
    render the info / move / capture / solve / promote tabs. T1 wires every
    field; T4/T5 populate ``samples`` and ``last_solve``.
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
    })
    return base
