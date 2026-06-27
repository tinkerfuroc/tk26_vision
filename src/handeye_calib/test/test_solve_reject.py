"""TDD for default-on per-axis MAD outlier rejection in handeye_solve.solve().

Task 7: Adds _per_sample_chain_errors + _modified_zscores helpers; changes
solve() defaults to reject_sigma=2.5, max_reject_frac=0.25; replaces the old
reproj-median reject loop with a SINGLE-worst-drop-then-RE-SOLVE per-axis
modified-z-score loop (re-solved via _solve_once each iteration so the anchor
seed, if any, is reused, and so borderline samples get a chance to fall back
under threshold after the worst outlier is removed).
"""
import numpy as np
import dataclasses
import pytest
from handeye_calib import synthetic as syn, handeye_solve as hs


def _get_rejected(res):
    """Extract rejected_indices list from SolveResult.per_method, or []."""
    for m in (res.per_method or []):
        if m.get("name") == "rejected_indices":
            return list(m.get("rejected_indices") or [])
    return []


def test_clean_data_rejects_nothing_by_default():
    """Default-on rejection (reject_sigma=2.5) is a no-op on realistic-noise data.

    KEY REGRESSION GUARD: single-worst-drop + the absolute physical floor must
    NOT over-reject legitimate samples on clean noisy data.  With pixel_noise=0.3
    the per-sample chain errors (~mm / ~0.3-0.95 deg) sit below the 10 mm / 1.5
    deg physical floor, so even when the symmetric modified z-score over-fires on
    the right-skewed upper tail (seed=11 train idx 6 had zr=2.503), the floor
    vetoes the drop.  The solve must still PASS.
    """
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts,
                   heldout_frac=0.25, rng_seed=0)   # default reject_sigma=2.5
    assert _get_rejected(res) == [], "clean noisy data must not reject any sample"
    assert res.status == "PASS"


@pytest.mark.parametrize("seed", list(range(10)))
def test_clean_data_rejects_nothing_multiseed(seed):
    """Multi-seed guard: the absolute physical floor keeps clean data zero-reject.

    Without the floor, the symmetric modified z-score over-fired on ~34/40 clean
    seeds (chain errors are non-negative right-skewed magnitudes at small n).
    The 10 mm / 3.0 deg floor must veto every one of these spurious statistical
    flags so NO clean seed drops a sample.
    """
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=seed)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, heldout_frac=0.25, rng_seed=0)
    rejected = next((m["rejected_indices"] for m in (res.per_method or [])
                     if m.get("name") == "rejected_indices"), [])
    assert rejected == [], f"seed {seed} spuriously dropped clean sample(s): {rejected}"


def test_default_rejection_catches_translation_fk_outlier():
    """Default rejection catches a sample whose FK translation is corrupted.

    Inject a ~5 cm pure-translation error into one TRAIN-split sample, leaving
    the wrist PnP (T_cam_board) untouched.  The translation chain error for that
    sample is large while the other clean samples constrain X/Tbb well, so the
    single-worst-drop loop must flag and reject it.
    """
    sc = syn.make_scenario(n_poses=22, pixel_noise=0.3, seed=7)
    # Pre-compute which original-list indices land in the train split (rng_seed=0).
    rng = np.random.default_rng(0)
    shuffled = np.arange(len(sc.samples))
    rng.shuffle(shuffled)
    n_te = max(1, int(round(len(sc.samples) * 0.2)))
    train_orig_idxs = sorted(shuffled[n_te:].tolist())
    # Corrupt the first train sample's FK translation by 5 cm.
    bad = train_orig_idxs[0]
    bad_T = sc.samples[bad].T_base_eef.copy()
    bad_T[:3, 3] += 0.05  # 5 cm pure translation shift
    corrupted = list(sc.samples)
    corrupted[bad] = dataclasses.replace(sc.samples[bad], T_base_eef=bad_T)
    # Solve with default reject_sigma=2.5 — the outlier must be caught.
    res = hs.solve(corrupted, sc.K, None, sc.board_pts, rng_seed=0)
    rejected = _get_rejected(res)
    assert len(rejected) > 0, "FK-translation outlier should have been rejected"


def test_translation_zscore_specifically_fires_on_spiked_sample():
    """Per-axis: a 5 cm FK translation spike trips the TRANSLATION z-score.

    Score the ground-truth chain over a set where exactly one sample's FK
    translation is spiked by 5 cm.  The spiked sample must be the global worst,
    its translation modified z-score must exceed 2.5, and translation (not
    rotation) must be the trigger — a pure-translation FK error leaves the
    rotation chain error untouched.
    """
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    X, Tbb = sc.X_true, sc.Tbb_true   # ground-truth chain isolates the injection
    samples = list(sc.samples)
    spike = 5
    bad_T = samples[spike].T_base_eef.copy()
    bad_T[:3, 3] += 0.05   # 5 cm pure translation
    samples[spike] = dataclasses.replace(samples[spike], T_base_eef=bad_T)
    t_e, r_e = hs._per_sample_chain_errors(X, Tbb, samples)
    zt = hs._modified_zscores(t_e)
    zr = hs._modified_zscores(r_e)
    worst = np.maximum(zt, zr)
    k = int(np.argmax(worst))
    assert k == spike, "the spiked sample must be the global worst"
    assert zt[k] > 2.5, f"translation z-score must fire on a 5cm FK spike (got {zt[k]:.2f})"
    assert zt[k] >= zr[k], "translation axis must be the trigger, not rotation"


def test_heldout_unaffected_by_rejection():
    """Held-out metrics are identical whether rejection is on or off when it's a no-op.

    On realistic-noise clean data (proven a no-op by the clean-data test), the
    solved X/Tbb and the test split are identical between the two calls, so every
    held-out metric must be numerically identical.  This proves the held-out
    evaluator is never reached by the rejection loop.
    """
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    res_on = hs.solve(sc.samples, sc.K, None, sc.board_pts,
                      heldout_frac=0.25, rng_seed=0)             # default 2.5
    res_off = hs.solve(sc.samples, sc.K, None, sc.board_pts,
                       heldout_frac=0.25, rng_seed=0, reject_sigma=None)
    np.testing.assert_allclose(
        res_on.heldout_metrics["trans_rmse_m"],
        res_off.heldout_metrics["trans_rmse_m"],
        atol=1e-12,
        err_msg="heldout trans_rmse_m must match: rejection was not a no-op")
    np.testing.assert_allclose(
        res_on.heldout_metrics["rot_rmse_rad"],
        res_off.heldout_metrics["rot_rmse_rad"],
        atol=1e-12,
        err_msg="heldout rot_rmse_rad must match: rejection was not a no-op")
