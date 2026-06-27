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


def test_solve_payload_v2_units_and_keys():
    """T5: ``solve_payload_v2`` adds mm/deg rendered metrics, per-method summary,
    per-sample reprojection list, and X_xyz_mm / X_rpy_deg keys. Works even when
    ``samples`` is empty (smoke-test path: per_sample_reproj_px == [])."""
    from handeye_calib.handeye_solve import SolveResult
    res = SolveResult(
        X=np.eye(4), Tbb=np.eye(4),
        train_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.00174, "reproj_px": 0.3},
        heldout_metrics={"trans_rmse_m": 0.002, "rot_rmse_rad": 0.00349, "reproj_px": 0.5},
        status="PASS",
        per_method=[{"name": "TSAI", "X": np.eye(4), "Tbb": np.eye(4), "reproj_px": 0.31}],
    )
    p = ws.solve_payload_v2(res, samples=[], K=np.eye(3), dist=None, board_pts=np.zeros((0, 3)))
    assert p["status"] == "PASS"
    assert p["X_xyz_mm"] == [0.0, 0.0, 0.0]
    assert len(p["X_rpy_deg"]) == 3
    assert p["train_metrics_mm_deg"]["trans_rmse_mm"] == 1.0
    assert abs(p["train_metrics_mm_deg"]["rot_rmse_deg"] - 0.1) < 0.01
    assert p["train_metrics_mm_deg"]["reproj_px"] == 0.3
    assert p["heldout_metrics_mm_deg"]["trans_rmse_mm"] == 2.0
    assert abs(p["heldout_metrics_mm_deg"]["rot_rmse_deg"] - 0.2) < 0.01
    assert p["heldout_metrics_mm_deg"]["reproj_px"] == 0.5
    assert p["per_method_summary"] == [{"name": "TSAI", "reproj_px": 0.31}]
    assert isinstance(p["per_sample_reproj_px"], list)
    assert p["per_sample_reproj_px"] == []


def test_solve_payload_v2_surfaces_depth_metrics():
    """FFS-depth path: when the solve carries a depth-grounded metric
    (``depth_point_rmse_mm`` / ``n_depth_corners``), solve_payload_v2 must
    surface it in the mm/deg blocks so the UI can show the honest, metric
    real-world accuracy alongside the reprojection number."""
    from handeye_calib.handeye_solve import SolveResult
    res = SolveResult(
        X=np.eye(4), Tbb=np.eye(4),
        train_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.00174,
                       "reproj_px": 0.3, "depth_point_rmse_mm": 3.2,
                       "n_depth_corners": 240},
        heldout_metrics={"trans_rmse_m": 0.002, "rot_rmse_rad": 0.00349,
                         "reproj_px": 0.5, "depth_point_rmse_mm": 4.1,
                         "n_depth_corners": 64},
        status="PASS", per_method=[])
    p = ws.solve_payload_v2(res, samples=[], K=np.eye(3), dist=None,
                            board_pts=np.zeros((0, 3)))
    assert p["heldout_metrics_mm_deg"]["depth_point_rmse_mm"] == 4.1
    assert p["heldout_metrics_mm_deg"]["n_depth_corners"] == 64
    assert p["train_metrics_mm_deg"]["depth_point_rmse_mm"] == 3.2


def test_solve_payload_v2_omits_depth_metrics_when_absent():
    """Monocular-only solve (no FFS): depth keys are simply absent, never a
    fake 0 that would read as a perfect depth fit."""
    from handeye_calib.handeye_solve import SolveResult
    res = SolveResult(
        X=np.eye(4), Tbb=np.eye(4),
        train_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.00174, "reproj_px": 0.3},
        heldout_metrics={"trans_rmse_m": 0.002, "rot_rmse_rad": 0.00349, "reproj_px": 0.5},
        status="PASS", per_method=[])
    p = ws.solve_payload_v2(res, samples=[], K=np.eye(3), dist=None,
                            board_pts=np.zeros((0, 3)))
    assert "depth_point_rmse_mm" not in p["heldout_metrics_mm_deg"]


def test_solve_payload_v2_per_sample_residuals_match_samples():
    """T5: ``per_sample_reproj_px`` is a 1:1 list with the input samples."""
    from handeye_calib import synthetic as syn, handeye_solve as hs
    sc = syn.make_scenario(n_poses=12, pixel_noise=0.3, seed=7)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25, rng_seed=0)
    p = ws.solve_payload_v2(res, samples=sc.samples, K=sc.K, dist=None, board_pts=sc.board_pts)
    assert len(p["per_sample_reproj_px"]) == len(sc.samples)
    assert all(isinstance(v, float) and v >= 0 for v in p["per_sample_reproj_px"])
    # "rejected_indices" may appear when default-on rejection (reject_sigma=2.5)
    # drops a borderline sample; the solve still PASSes in that case.
    assert {m["name"] for m in p["per_method_summary"]} <= {
        "TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS", "rejected_indices"}


def test_enriched_state_payload_safety_preview_roundtrip():
    """T3: ``safety_preview`` is a new optional kwarg that round-trips into the
    payload under the same key. Defaults to ``None`` for back-compat (T1/T2
    callers omit the kwarg); when given a ``{safe, detail}`` dict it must show
    up in the returned dict so the Move tab can render it without re-doing the
    SafetyEnvelope math in JS.
    """
    # Back-compat: omitting the kwarg keeps the key present but ``None``.
    d_default = ws.enriched_state_payload(
        camera_connected=False, intrinsics_ok=False, num_samples=0,
        last_detection=None, status_msg="idle",
        frame_count=0, frame_hz=0.0, frame_age_sec=None,
        image_topic="/foo", ros_domain_id=0,
        t_base_ee=None, xarm_joint_positions=None,
        board={}, safety_envelope={},
        stability={"steady": False, "since_frames": 0, "target_frames": 3},
        samples=[], diversity={"coverage_deg": 0.0, "target_deg": 30.0},
        last_solve=None,
    )
    assert "safety_preview" in d_default
    assert d_default["safety_preview"] is None

    # When supplied, the dict round-trips verbatim.
    sp = {"safe": True, "detail": "ok"}
    d = ws.enriched_state_payload(
        camera_connected=False, intrinsics_ok=False, num_samples=0,
        last_detection=None, status_msg="idle",
        frame_count=0, frame_hz=0.0, frame_age_sec=None,
        image_topic="/foo", ros_domain_id=0,
        t_base_ee=None, xarm_joint_positions=None,
        board={}, safety_envelope={},
        stability={"steady": False, "since_frames": 0, "target_frames": 3},
        samples=[], diversity={"coverage_deg": 0.0, "target_deg": 30.0},
        last_solve=None, safety_preview=sp,
    )
    assert d["safety_preview"] == sp


# ---------------------------------------------------------------------------
# T4: diversity meter + per-sample metadata helpers
# ---------------------------------------------------------------------------

def test_compute_diversity_zero_for_zero_or_one_sample():
    from handeye_calib import handeye_model as hm
    assert ws.compute_diversity_deg([]) == 0.0
    s = hm.Sample(np.eye(4), np.eye(4), np.zeros((0, 2)), np.zeros((0,), int))
    assert ws.compute_diversity_deg([s]) == 0.0


def test_compute_diversity_max_pairwise_deg():
    from handeye_calib import handeye_model as hm

    def mk(rpy_deg):
        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
        return hm.Sample(T, np.eye(4), np.zeros((0, 2)), np.zeros((0,), int))

    samples = [mk([0, 0, 0]), mk([45, 0, 0]), mk([0, 30, 0])]
    cov = ws.compute_diversity_deg(samples)
    assert cov >= 45.0  # at least the 0->45 about X


def test_sample_metadata_shape_first_sample():
    """First accepted sample has angular_delta_deg=None (no predecessor)."""
    from handeye_calib import handeye_model as hm
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    s = hm.Sample(T, np.eye(4), np.zeros((0, 2)), np.zeros((0,), int))
    md = ws.sample_metadata(0, s, prev_sample=None, n_corners=12,
                            reproj_px=0.4, area_frac=0.08,
                            joint_positions=None, ts=1234.5)
    assert md["idx"] == 0
    assert md["n_corners"] == 12
    assert md["reproj_px"] == 0.4
    assert md["area_frac"] == 0.08
    assert md["angular_delta_deg"] is None
    assert md["joint_positions"] is None
    assert md["ts"] == 1234.5


def test_sample_metadata_angular_delta_vs_prev():
    """Second sample reports the rotation angle vs its predecessor."""
    from handeye_calib import handeye_model as hm
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, :3] = R.from_euler('x', 45, degrees=True).as_matrix()
    s0 = hm.Sample(T0, np.eye(4), np.zeros((0, 2)), np.zeros((0,), int))
    s1 = hm.Sample(T1, np.eye(4), np.zeros((0, 2)), np.zeros((0,), int))
    md = ws.sample_metadata(1, s1, prev_sample=s0, n_corners=10,
                            reproj_px=0.5, area_frac=0.1,
                            joint_positions=[0.0] * 7, ts=2.0)
    assert md["idx"] == 1
    assert md["angular_delta_deg"] is not None
    assert abs(md["angular_delta_deg"] - 45.0) < 1e-6
    assert md["joint_positions"] == [0.0] * 7


def test_solve_payload_v2_carries_observability():
    import numpy as np
    from handeye_calib import synthetic as syn, handeye_solve as hs, web_support as ws
    sc = syn.make_scenario(n_poses=12, pixel_noise=0.3, seed=11)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, reject_sigma=None)
    payload = ws.solve_payload_v2(res, sc.samples, sc.K, None, sc.board_pts)
    assert "observability" in payload
    assert "ok" in payload["observability"]
