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
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts)   # default reject_sigma=2.5
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
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts)
    rejected = next((m["rejected_indices"] for m in (res.per_method or [])
                     if m.get("name") == "rejected_indices"), [])
    assert rejected == [], f"seed {seed} spuriously dropped clean sample(s): {rejected}"


def test_default_rejection_catches_translation_fk_outlier():
    """Default rejection catches a sample whose FK translation is corrupted.

    Inject a ~5 cm pure-translation error into one sample, leaving the wrist PnP
    (T_cam_board) untouched.  The translation chain error for that sample is
    large while the other clean samples constrain X/Tbb well, so the all-sample
    single-worst-drop loop must flag and reject it — and that specific sample.
    """
    sc = syn.make_scenario(n_poses=22, pixel_noise=0.3, seed=7)
    bad = 5
    bad_T = sc.samples[bad].T_base_eef.copy()
    bad_T[:3, 3] += 0.05  # 5 cm pure translation shift
    corrupted = list(sc.samples)
    corrupted[bad] = dataclasses.replace(sc.samples[bad], T_base_eef=bad_T)
    # Solve with default reject_sigma=2.5 — the outlier must be caught.
    res = hs.solve(corrupted, sc.K, None, sc.board_pts)
    rejected = _get_rejected(res)
    assert bad in rejected, f"FK-translation outlier #{bad} should be rejected; got {rejected}"


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


def test_rejection_is_noop_on_clean_data():
    """Default rejection (2.5) must be a numerical no-op on clean data.

    On realistic-noise clean data (proven zero-reject by the clean-data test),
    turning rejection on must not change X/Tbb or the residual at all — every
    metric must match the ``reject_sigma=None`` solve bit-for-bit. There is no
    train/held-out split anymore; the whole set is fit either way.
    """
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    res_on = hs.solve(sc.samples, sc.K, None, sc.board_pts)               # default 2.5
    res_off = hs.solve(sc.samples, sc.K, None, sc.board_pts, reject_sigma=None)
    np.testing.assert_allclose(
        res_on.metrics["trans_rmse_m"], res_off.metrics["trans_rmse_m"],
        atol=1e-12, err_msg="trans_rmse_m must match: rejection was not a no-op")
    np.testing.assert_allclose(
        res_on.metrics["rot_rmse_rad"], res_off.metrics["rot_rmse_rad"],
        atol=1e-12, err_msg="rot_rmse_rad must match: rejection was not a no-op")


def test_reproj_axis_rejects_mount_inconsistent_poses():
    """A pose whose camera->flange transform is inconsistent with the rest
    (mid-ring / mount-flex capture) shows up as a high REPROJECTION outlier
    even when its chain-rotation error sits below the 3 deg floor. The new
    reprojection rejection axis must catch it; clean poses must NOT be dropped;
    and rejected indices are ORIGINAL sample indices."""
    from handeye_calib import synthetic as syn, handeye_solve as hs
    from handeye_calib import transforms as tf, handeye_model as hm
    sc = syn.make_scenario(n_poses=18, pixel_noise=0.3, seed=5)
    bad = {2, 7, 11, 15}
    rng = np.random.default_rng(1)
    for i in bad:
        s = sc.samples[i]
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        # ~1.5 deg board-pose rotation (below the 3 deg chain-rot floor), with
        # obs_px reprojected to match -> clean per-frame detection, but globally
        # inconsistent => high calibration reproj.
        s.T_cam_board = s.T_cam_board @ tf.T_from_vec(
            np.concatenate([np.radians(1.5) * ax, np.zeros(3)]))
        s.obs_px = hm.project_corners(sc.board_pts[s.corner_idx], s.T_cam_board, sc.K)
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts)
    assert res.rejected_indices, "reproj-inconsistent poses should be rejected"
    assert set(res.rejected_indices).issubset(bad), \
        f"rejection wrongly dropped a clean pose: {set(res.rejected_indices) - bad}"


def test_outlier_is_rejected_regardless_of_position():
    """REGRESSION (pan-tilt parity / live-data symptom): MAD screens ALL samples,
    so an outlier is weeded out no matter where it sits in the list.

    Pre-fix, rejection ran on the train split only, so a corrupted sample that
    shuffled into the held-out 20% was never scored — it survived as a glaring
    residual in the per-sample view (the exact live-data symptom: a 26 px point
    MAD 'failed' to weed out). With no split and full-set screening, every
    position is screened. Corrupt the LAST sample (which the old train/test
    shuffle could push into held-out) and assert it is caught.
    """
    sc = syn.make_scenario(n_poses=22, pixel_noise=0.3, seed=7)
    bad = len(sc.samples) - 1
    bad_T = sc.samples[bad].T_base_eef.copy()
    bad_T[:3, 3] += 0.05   # 5 cm pure translation
    corrupted = list(sc.samples)
    corrupted[bad] = dataclasses.replace(sc.samples[bad], T_base_eef=bad_T)
    res = hs.solve(corrupted, sc.K, None, sc.board_pts)
    assert bad in (res.rejected_indices or []), (
        f"outlier #{bad} must be screened out by all-sample MAD; "
        f"got rejected={res.rejected_indices}")
    # And the per-drop diagnostic must explain WHY (residual + robust-z).
    logged = {r["idx"] for r in (res.rejection_log or [])}
    assert bad in logged, "rejection_log must record the dropped sample's residual+z"
