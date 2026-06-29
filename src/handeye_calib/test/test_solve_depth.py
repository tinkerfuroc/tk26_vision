"""TDD for the FFS-depth residual in the bundle adjust.

The calibration's per-view T_cam_board is monocular planar PnP, whose
optical-axis (depth/scale) translation is the weakest-constrained DOF and is
biased by any focal-length / board-scale error. FoundationStereo gives metric
depth at the corners; feeding it as a 3D point residual in the bundle adjust
pins that DOF. These tests prove the residual (a) is harmless when depth is
perfect, (b) corrects a focal-scale-biased monocular solve, (c) degrades
gracefully to monocular when depth is absent, (d) honors the validity mask.
"""
import numpy as np
from handeye_calib import synthetic as syn, transforms as tf, handeye_solve as hs


def test_depth_residual_harmless_when_consistent():
    # Noiseless, correct K, perfect metric depth -> BA still recovers X exactly.
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=8, with_depth=True)
    Xs, Tbbs, _ = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    Xb, _, info = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs,
                                   depth_weight=1.0, depth_sigma_m=0.003)
    assert np.linalg.norm(Xb[:3, 3] - sc.X_true[:3, 3]) * 1000 < 0.1
    assert info["final_reproj_px"] < 1e-2


def test_depth_corrects_focal_scale_bias():
    # Pixels were generated with the TRUE K. Hand the solver a 3%-too-large
    # focal length: monocular reprojection-only BA then drifts to a scale-wrong
    # X (the classic planar-PnP failure — translation along the optical axis is
    # the least-constrained DOF). The metric depth residual pulls the
    # translation scale back toward truth, halving the bias at the shipped
    # depth_weight=2.0.
    sc = syn.make_scenario(n_poses=18, pixel_noise=0.0, seed=7, with_depth=True)
    K_wrong = sc.K.copy()
    K_wrong[0, 0] *= 1.03
    K_wrong[1, 1] *= 1.03

    Xoff, _, _ = hs.bundle_adjust(sc.samples, K_wrong, None, sc.board_pts,
                                  sc.X_true, sc.Tbb_true, depth_weight=0.0)
    Xon, _, _ = hs.bundle_adjust(sc.samples, K_wrong, None, sc.board_pts,
                                 sc.X_true, sc.Tbb_true,
                                 depth_weight=2.0, depth_sigma_m=0.003)
    err_off = np.linalg.norm(Xoff[:3, 3] - sc.X_true[:3, 3])
    err_on = np.linalg.norm(Xon[:3, 3] - sc.X_true[:3, 3])
    assert err_off > 0.005, f"monocular not biased enough to be a fair test: {err_off*1000:.2f}mm"
    assert err_on < err_off, f"depth made it worse: on={err_on*1000:.2f} off={err_off*1000:.2f}mm"
    assert err_on < 0.6 * err_off, f"depth didn't help enough: on={err_on*1000:.2f} off={err_off*1000:.2f}mm"


def test_depth_solve_robust_to_depth_noise():
    # With correct K but realistic 2mm FFS depth noise, the shipped solve
    # (depth on) must still recover X to within a couple mm — i.e. the depth
    # weight isn't so high that stereo speckle corrupts the rotation/lateral
    # that reprojection nails.
    sc = syn.make_scenario(n_poses=18, pixel_noise=0.3, seed=4,
                           with_depth=True, depth_noise=0.002)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts)   # default depth_weight
    err_mm = np.linalg.norm(res.X[:3, 3] - sc.X_true[:3, 3]) * 1000
    assert err_mm < 2.0, f"X off by {err_mm:.2f}mm under 2mm depth noise"
    assert tf.rotation_angle_deg(res.X[:3, :3], sc.X_true[:3, :3]) < 0.3


def test_depth_weight_noop_when_no_depth_present():
    # Samples without obs_xyz_cam + depth_weight>0 must behave EXACTLY like
    # monocular (graceful fallback when FFS is unavailable at capture).
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.3, seed=8)  # with_depth=False
    Xs, Tbbs, _ = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    Xa, _, _ = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs,
                                depth_weight=0.0)
    Xb, _, _ = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs,
                                depth_weight=5.0, depth_sigma_m=0.003)
    np.testing.assert_allclose(Xa, Xb, atol=1e-12)


def test_depth_validity_mask_excludes_bad_corners():
    # Corrupt a few corners' depth but mark them invalid -> result must match
    # the all-good case (invalid entries never enter the residual).
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=8, with_depth=True)
    Xs, Tbbs, _ = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    Xgood, _, _ = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs,
                                   depth_weight=1.0, depth_sigma_m=0.003)
    for s in sc.samples:
        s.obs_xyz_cam = s.obs_xyz_cam.copy()
        s.obs_xyz_valid = s.obs_xyz_valid.copy()
        s.obs_xyz_cam[0] = [9.9, 9.9, 9.9]   # garbage
        s.obs_xyz_valid[0] = False           # but flagged invalid
    Xmask, _, _ = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs,
                                   depth_weight=1.0, depth_sigma_m=0.003)
    np.testing.assert_allclose(Xgood, Xmask, atol=1e-9)


def test_solve_reports_depth_metrics_when_present():
    sc = syn.make_scenario(n_poses=16, pixel_noise=0.3, seed=11, with_depth=True)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts,
                   depth_weight=1.0, depth_sigma_m=0.003)
    # depth-grounded metric surfaces in the (all-sample) residual block
    assert "depth_point_rmse_mm" in res.metrics
    assert res.metrics["depth_point_rmse_mm"] is not None
    assert res.metrics["n_depth_corners"] > 0


def test_solve_omits_depth_metrics_when_absent():
    sc = syn.make_scenario(n_poses=16, pixel_noise=0.3, seed=11)  # no depth
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts)
    assert res.metrics.get("depth_point_rmse_mm") is None


def test_depth_residual_tolerates_nan_rows_without_mask():
    # deproject_corners writes NaN rows for holes; even if the validity mask is
    # somehow absent, a NaN must never reach least_squares ("Residuals are not
    # finite"). The residual ANDs validity with finiteness, so the solve runs.
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=8, with_depth=True)
    for s in sc.samples:
        s.obs_xyz_cam = s.obs_xyz_cam.copy()
        s.obs_xyz_cam[0] = [np.nan, np.nan, np.nan]   # a hole
        s.obs_xyz_valid = None                         # mask dropped
    Xs, Tbbs, _ = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    Xb, _, _ = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs,
                                depth_weight=1.0, depth_sigma_m=0.003)
    assert np.all(np.isfinite(Xb))
    assert np.linalg.norm(Xb[:3, 3] - sc.X_true[:3, 3]) * 1000 < 0.5
