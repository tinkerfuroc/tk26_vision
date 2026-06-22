import numpy as np
from handeye_calib import synthetic as syn
from handeye_calib import handeye_model as hm


def test_scenario_shapes_and_consistency():
    sc = syn.make_scenario(n_poses=12, pixel_noise=0.0, seed=0)
    assert len(sc.samples) == 12
    # With zero noise, observed pixels must equal reprojection through ground-truth X, Tbb.
    from handeye_calib import transforms as tf
    for s in sc.samples:
        T_cam_board = tf.invert(s.T_base_eef @ sc.X_true) @ sc.Tbb_true
        px = hm.project_corners(sc.board_pts[s.corner_idx], T_cam_board, sc.K)
        np.testing.assert_allclose(px, s.obs_px, atol=1e-6)


def test_pnp_pose_matches_truth_noiseless():
    sc = syn.make_scenario(n_poses=8, pixel_noise=0.0, seed=1)
    from handeye_calib import transforms as tf
    for s in sc.samples:
        T_cam_board_true = tf.invert(s.T_base_eef @ sc.X_true) @ sc.Tbb_true
        np.testing.assert_allclose(s.T_cam_board, T_cam_board_true, atol=1e-3)
