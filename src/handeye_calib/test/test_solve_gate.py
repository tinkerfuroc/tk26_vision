import numpy as np
from handeye_calib import synthetic as syn, handeye_solve as hs


def test_solve_passes_gate_on_clean_data():
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25, rng_seed=0)
    assert res.status == "PASS"
    assert res.heldout_metrics["trans_rmse_m"] < 0.003
    assert res.heldout_metrics["rot_rmse_rad"] < 0.00873
    assert res.heldout_metrics["reproj_px"] < 1.5


def test_gate_thresholds():
    assert hs.gate({"trans_rmse_m": 0.002, "rot_rmse_rad": 0.005, "reproj_px": 1.0}) == "PASS"
    assert hs.gate({"trans_rmse_m": 0.005, "rot_rmse_rad": 0.005, "reproj_px": 1.0}) == "WARN"
    assert hs.gate({"trans_rmse_m": 0.02, "rot_rmse_rad": 0.005, "reproj_px": 1.0}) == "FAIL"


def test_split_is_deterministic_and_disjoint():
    sc = syn.make_scenario(n_poses=10, pixel_noise=0.0, seed=0)
    tr1, te1 = hs.split_train_test(sc.samples, 0.3, rng_seed=5)
    tr2, te2 = hs.split_train_test(sc.samples, 0.3, rng_seed=5)
    assert [id(s) for s in te1] == [id(s) for s in te2]      # deterministic
    assert len(tr1) + len(te1) == 10 and len(te1) == 3
