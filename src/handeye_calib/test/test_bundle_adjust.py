import numpy as np
from handeye_calib import synthetic as syn, transforms as tf, handeye_solve as hs


def test_ba_beats_seed_under_noise():
    sc = syn.make_scenario(n_poses=18, pixel_noise=0.4, seed=7)
    Xs, Tbbs, _ = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    Xb, Tbbb, info = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs)
    seed_err = np.linalg.norm(Xs[:3, 3] - sc.X_true[:3, 3]) * 1000
    ba_err = np.linalg.norm(Xb[:3, 3] - sc.X_true[:3, 3]) * 1000
    assert ba_err <= seed_err + 1e-6
    assert ba_err < 1.0, f"{ba_err} mm"                       # sub-mm under 0.4px noise
    assert tf.rotation_angle_deg(Xb[:3, :3], sc.X_true[:3, :3]) < 0.2
    assert info["final_reproj_px"] < 0.6


def test_ba_exact_when_noiseless():
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=8)
    Xs, Tbbs, _ = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    Xb, _, info = hs.bundle_adjust(sc.samples, sc.K, None, sc.board_pts, Xs, Tbbs)
    assert np.linalg.norm(Xb[:3, 3] - sc.X_true[:3, 3]) * 1000 < 0.05
    assert info["final_reproj_px"] < 1e-3
