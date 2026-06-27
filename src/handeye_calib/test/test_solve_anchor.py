import numpy as np
from handeye_calib import synthetic as syn, handeye_solve as hs, transforms as tf


def test_seed_from_board_anchor_recovers_X_with_zero_rotation_diversity():
    # Near pure-translation set: AX=XB is rank-deficient, but a known board
    # pose in base determines X from a single pose, so the anchor seed recovers
    # X even here.
    sc = syn.make_scenario(n_poses=8, pixel_noise=0.0, seed=3, rot_range=0.02)
    X_seed, Tbb_seed = hs.seed_from_board_anchor(sc.samples, sc.Tbb_true)
    dt = np.linalg.norm(X_seed[:3, 3] - sc.X_true[:3, 3]) * 1000.0
    dr = tf.rotation_angle_deg(X_seed[:3, :3], sc.X_true[:3, :3])
    assert dt < 1.0 and dr < 0.2          # sub-mm / sub-0.2deg from an exact anchor
    assert np.allclose(Tbb_seed, sc.Tbb_true)


def test_average_board_anchors_reports_scatter():
    sc = syn.make_scenario(n_poses=4, pixel_noise=0.0, seed=1)
    rng = np.random.default_rng(0)
    # Three noisy observations of the same true board pose.
    obs = []
    for _ in range(3):
        noise = tf.T_from_vec(np.concatenate([
            rng.normal(0, np.radians(0.3), 3), rng.normal(0, 0.004, 3)]))
        obs.append(sc.Tbb_true @ noise)
    mean, scatter = hs.average_board_anchors(obs)
    assert scatter["n"] == 3
    assert 0.0 < scatter["trans_mm"] < 20.0
    assert 0.0 < scatter["rot_deg"] < 2.0
    # mean is close to truth (noise averages partly down)
    assert np.linalg.norm(mean[:3, 3] - sc.Tbb_true[:3, 3]) * 1000.0 < 10.0
