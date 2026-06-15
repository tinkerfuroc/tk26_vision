import numpy as np
from handeye_calib import synthetic as syn, transforms as tf, handeye_solve as hs


def test_seed_recovers_truth_noiseless():
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=2)
    X, Tbb, per_method = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    dt = np.linalg.norm(X[:3, 3] - sc.X_true[:3, 3]) * 1000
    dr = tf.rotation_angle_deg(X[:3, :3], sc.X_true[:3, :3])
    assert dt < 2.0, f"{dt} mm"       # linear seed: a couple mm even noiseless
    assert dr < 0.5, f"{dr} deg"
    assert len(per_method) >= 3        # several OpenCV methods tried


def test_seed_picks_lowest_reproj():
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.3, seed=4)
    X, Tbb, per_method = hs.seed_handeye(sc.samples, sc.K, None, sc.board_pts)
    best = min(per_method, key=lambda m: m["reproj_px"])
    np.testing.assert_allclose(X, best["X"], atol=1e-9)
