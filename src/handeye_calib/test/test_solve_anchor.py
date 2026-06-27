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


def test_anchor_rescues_degenerate_solve():
    # Low rotation diversity => calibrateHandEye seed is poorly conditioned.
    # Pin BOTH solves to TSAI only: on this degenerate set TSAI lands in a
    # wrong basin its bundle-adjust can't escape (~1.5 m off), so the closed-
    # form-only solve fails deterministically and the multi-start board-anchor
    # branch is what rescues it. (The default 5-method best-of would let a
    # stronger method recover, masking the rescue path under test.)
    methods = {"TSAI": hs._METHODS["TSAI"]}
    sc = syn.make_scenario(n_poses=12, pixel_noise=0.3, seed=7, rot_range=0.05)
    plain = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25,
                     reject_sigma=None, methods=methods)
    # Simulate a realistic (slightly noisy) head anchor: ~5 mm / 0.3 deg off.
    rng = np.random.default_rng(2)
    anchor = sc.Tbb_true @ tf.T_from_vec(np.concatenate([
        rng.normal(0, np.radians(0.3), 3), rng.normal(0, 0.005, 3)]))
    assisted = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25,
                        reject_sigma=None, anchor_Tbb=anchor, methods=methods)
    err_plain = np.linalg.norm(plain.X[:3, 3] - sc.X_true[:3, 3])
    err_assisted = np.linalg.norm(assisted.X[:3, 3] - sc.X_true[:3, 3])
    # The anchor-assisted X is dramatically better on a degenerate set, and
    # within the head's ~1 cm floor (NOT necessarily the 3 mm gate).
    assert err_assisted < err_plain
    assert err_assisted < 0.012


def test_solve_default_no_anchor_is_unchanged():
    # anchor_Tbb=None must reproduce the historical clean-data PASS.
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25,
                   rng_seed=0, reject_sigma=None)
    assert res.status == "PASS"


def test_observability_flags_low_diversity_and_passes_diverse():
    diverse = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=4, rot_range=0.6)
    degen = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=4, rot_range=0.01)
    o_div = hs.rotation_observability(diverse.samples)
    o_deg = hs.rotation_observability(degen.samples)
    assert o_div["ok"] is True
    assert o_deg["ok"] is False
    assert o_deg["second_singular"] <= o_div["second_singular"]
