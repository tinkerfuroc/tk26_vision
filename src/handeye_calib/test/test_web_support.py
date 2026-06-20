import numpy as np
from scipy.spatial.transform import Rotation as R
from handeye_calib import web_support as ws


def test_tf_to_matrix_identity():
    T = ws.tf_to_matrix([0, 0, 0], [0, 0, 0, 1])
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_tf_to_matrix_known():
    q = R.from_euler('z', 90, degrees=True).as_quat()  # xyzw
    T = ws.tf_to_matrix([1, 2, 3], q)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], R.from_quat(q).as_matrix(), atol=1e-12)


def test_matrix_to_xyz_rpy_urdf_convention():
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', [0.3, -0.7, 1.1]).as_matrix()
    T[:3, 3] = [0.06, -0.01, 0.02]
    xyz, rpy = ws.matrix_to_xyz_rpy(T)
    assert xyz == [0.06, -0.01, 0.02] or np.allclose(xyz, [0.06, -0.01, 0.02])
    Rr = (R.from_euler('z', rpy[2]).as_matrix() @ R.from_euler('y', rpy[1]).as_matrix()
          @ R.from_euler('x', rpy[0]).as_matrix())
    np.testing.assert_allclose(Rr, T[:3, :3], atol=1e-9)


def test_charuco_to_sample_arrays():
    corners = np.array([[[10., 20.]], [[30., 40.]], [[50., 60.]]])  # (3,1,2) cv2 shape
    ids = np.array([[5], [2], [9]])                                  # (3,1)
    px, idx = ws.charuco_to_sample_arrays(corners, ids)
    assert px.shape == (3, 2) and idx.shape == (3,)
    np.testing.assert_allclose(px, [[10, 20], [30, 40], [50, 60]])
    assert idx.tolist() == [5, 2, 9] and idx.dtype.kind == 'i'


def test_gate_color():
    assert ws.gate_color("PASS") == "#1a9850"
    assert ws.gate_color("WARN") == "#f59e0b"
    assert ws.gate_color("FAIL") == "#d73027"
    assert ws.gate_color("???") == "#888888"


def test_state_payload_keys():
    d = ws.state_payload(False, False, 0, None, "idle")
    assert set(d) == {"camera_connected", "intrinsics_ok", "num_samples",
                      "last_detection", "status_msg"}
    assert d["camera_connected"] is False and d["last_detection"] is None


def test_solve_payload_keys():
    from handeye_calib.handeye_solve import SolveResult
    res = SolveResult(X=np.eye(4), Tbb=np.eye(4),
                      train_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.002, "reproj_px": 0.3},
                      heldout_metrics={"trans_rmse_m": 0.002, "rot_rmse_rad": 0.004, "reproj_px": 0.5},
                      status="PASS", per_method=[])
    p = ws.solve_payload(res)
    assert p["status"] == "PASS"
    assert len(p["X_xyz"]) == 3 and len(p["X_rpy"]) == 3
    assert p["heldout_metrics"]["reproj_px"] == 0.5


def test_placeholder_jpeg_is_jpeg():
    b = ws.placeholder_jpeg("no camera")
    assert isinstance(b, (bytes, bytearray)) and bytes(b[:2]) == b"\xff\xd8"


def test_encode_jpeg_roundtrips_shape():
    import cv2
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :, 1] = 200
    b = ws.encode_jpeg(img)
    assert bytes(b[:2]) == b"\xff\xd8"
    dec = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    assert dec.shape == (48, 64, 3)


def test_draw_overlay_preserves_shape_and_handles_empty():
    img = np.zeros((48, 64, 3), np.uint8)
    out = ws.draw_charuco_overlay(img, np.array([[10.0, 20.0], [30.0, 40.0]]))
    assert out.shape == img.shape
    out2 = ws.draw_charuco_overlay(img, np.empty((0, 2)))
    assert out2.shape == img.shape


def test_overlay_with_indices_preserves_shape():
    """T2: overlay accepts optional ids + rms + image_topic, shape stays the same.

    The IDs annotate each corner; rms_px is rendered in a header bar; image_topic
    is rendered in the header bar for diagnostics. None of these change the
    BGR's HxWxC.
    """
    img = np.zeros((100, 200, 3), np.uint8)
    out = ws.draw_charuco_overlay(
        img,
        corners_xy=np.array([[50, 50], [150, 80]]),
        ids=np.array([3, 7]),
        rms_px=0.42,
        image_topic="/foo",
    )
    assert out.shape == img.shape


def test_overlay_kwargs_optional_default_to_legacy_behavior():
    """T2: omitting ids/rms_px/image_topic must behave like the v1 overlay."""
    img = np.zeros((48, 64, 3), np.uint8)
    out = ws.draw_charuco_overlay(img, np.array([[10.0, 20.0]]),
                                  ids=None, rms_px=None, image_topic=None)
    assert out.shape == img.shape


def test_mm_and_deg_round_to_4dp():
    assert ws.mm(0.0012345) == 1.2345
    assert ws.deg(0.0174533) == 1.0  # 1 deg in rad -> ~= 1.0
    assert ws.mm(-0.001) == -1.0


def test_enriched_state_payload_has_all_keys():
    d = ws.enriched_state_payload(
        camera_connected=False, intrinsics_ok=False, num_samples=0,
        last_detection=None, status_msg="idle",
        frame_count=0, frame_hz=0.0, frame_age_sec=None,
        image_topic="/foo", ros_domain_id=0,
        t_base_ee=None, xarm_joint_positions=None,
        board={"squares_x": 5}, safety_envelope={"z_floor_m": 0.0},
        stability={"steady": False, "since_frames": 0, "target_frames": 3},
        samples=[], diversity={"coverage_deg": 0.0, "target_deg": 30.0},
        last_solve=None,
    )
    required = {
        "camera_connected", "intrinsics_ok", "num_samples", "last_detection",
        "status_msg", "frame_count", "frame_hz", "frame_age_sec", "image_topic",
        "ros_domain_id", "t_base_ee", "xarm_joint_positions", "board",
        "safety_envelope", "stability", "samples", "diversity", "last_solve",
    }
    assert set(d) >= required
