import numpy as np
import pytest
from handeye_calib import synthetic as syn, handeye_solve as hs, transforms as tf


def test_two_placements_recover_X():
    sc1 = syn.make_scenario(n_poses=12, pixel_noise=0.3, seed=1)
    sc2 = syn.make_scenario(n_poses=12, pixel_noise=0.3, seed=2)
    placements = [("p1", sc1.samples), ("p2", sc2.samples)]
    result = hs.solve_multi_placement(placements, sc1.K, None, sc1.board_pts)
    assert np.allclose(result.X, sc1.X_true, atol=5e-3)
    assert result.status in ("PASS", "WARN")


def test_three_placements_recover_X():
    sc1 = syn.make_scenario(n_poses=8, pixel_noise=0.3, seed=10)
    sc2 = syn.make_scenario(n_poses=8, pixel_noise=0.3, seed=11)
    sc3 = syn.make_scenario(n_poses=8, pixel_noise=0.3, seed=12)
    placements = [("p1", sc1.samples), ("p2", sc2.samples), ("p3", sc3.samples)]
    result = hs.solve_multi_placement(placements, sc1.K, None, sc1.board_pts)
    assert np.allclose(result.X, sc1.X_true, atol=5e-3)
    assert result.status in ("PASS", "WARN")


def test_placement_below_min_samples_raises():
    sc = syn.make_scenario(n_poses=10, pixel_noise=0.3, seed=5)
    placements = [("short", sc.samples[:5]), ("ok", sc.samples)]
    with pytest.raises(ValueError, match="short"):
        hs.solve_multi_placement(placements, sc.K, None, sc.board_pts)


def test_seed_placement_id_is_string():
    sc1 = syn.make_scenario(n_poses=12, pixel_noise=0.3, seed=3)
    sc2 = syn.make_scenario(n_poses=12, pixel_noise=0.3, seed=4)
    placements = [("alpha", sc1.samples), ("beta", sc2.samples)]
    result = hs.solve_multi_placement(placements, sc1.K, None, sc1.board_pts)
    assert isinstance(result.seed_placement_id, str)
    assert result.seed_placement_id != ""
    assert result.seed_placement_id in ("alpha", "beta")


def test_single_placement_matches_solve():
    sc = syn.make_scenario(n_poses=15, pixel_noise=0.0, seed=11)
    placements = [("only", sc.samples)]
    multi_result = hs.solve_multi_placement(placements, sc.K, None, sc.board_pts)
    single_result = hs.solve(sc.samples, sc.K, None, sc.board_pts)
    assert np.allclose(multi_result.X, single_result.X, atol=1e-4)
