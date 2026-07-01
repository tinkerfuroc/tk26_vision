import numpy as np
from handeye_calib import synthetic as syn, handeye_solve as hs


def test_solve_passes_gate_on_clean_data():
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts)
    assert res.status == "PASS"
    assert res.metrics["trans_rmse_m"] < 0.003
    assert res.metrics["rot_rmse_rad"] < 0.00873
    assert res.metrics["reproj_px"] < 1.5


def test_gate_thresholds():
    assert hs.gate({"trans_rmse_m": 0.002, "rot_rmse_rad": 0.005, "reproj_px": 1.0}) == "PASS"
    assert hs.gate({"trans_rmse_m": 0.005, "rot_rmse_rad": 0.005, "reproj_px": 1.0}) == "WARN"
    assert hs.gate({"trans_rmse_m": 0.02, "rot_rmse_rad": 0.005, "reproj_px": 1.0}) == "FAIL"
