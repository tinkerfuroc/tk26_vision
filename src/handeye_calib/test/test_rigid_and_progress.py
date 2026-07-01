"""Rigid-mount diagnostic (AX=XB conjugacy) + live MAD progress callback."""
import dataclasses
import numpy as np
from handeye_calib import synthetic as syn, handeye_solve as hs, transforms as tf


def test_rigid_closure_passes_on_consistent_capture():
    """A synthetic capture (T_base_eef + T_cam_board consistent with ONE fixed
    X) is rigid -> closure ~0, ok True."""
    sc = syn.make_scenario(n_poses=18, pixel_noise=0.0, seed=4, rot_range=0.6)
    r = hs.rigid_closure_deg(sc.samples)
    assert r["n_pairs"] > 0
    assert r["median_deg"] < 0.5 and r["ok"] is True, r


def test_rigid_closure_flags_nonrigid_mount():
    """Perturb each pose's camera observation by a random ~2deg rotation (mount
    flex: the camera-to-flange transform is no longer constant). The flange
    motion no longer matches the camera motion angle -> high closure, ok False.
    """
    sc = syn.make_scenario(n_poses=18, pixel_noise=0.0, seed=4, rot_range=0.6)
    rng = np.random.default_rng(0)
    flexed = []
    for s in sc.samples:
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        dR = tf.T_from_vec(np.concatenate([np.radians(2.0) * ax, np.zeros(3)]))
        flexed.append(dataclasses.replace(s, T_cam_board=s.T_cam_board @ dR))
    r = hs.rigid_closure_deg(flexed)
    assert r["median_deg"] > 0.5 and r["ok"] is False, r
    assert "NON-RIGID" in r["detail"]


def test_progress_cb_streams_start_and_each_rejection():
    """solve() must fire progress_cb: one 'start' then one 'rejecting' per drop,
    each carrying the last_drop residual+z dict. Default (cb=None) is unchanged.
    """
    sc = syn.make_scenario(n_poses=22, pixel_noise=0.3, seed=7)
    bad = 5
    bad_T = sc.samples[bad].T_base_eef.copy()
    bad_T[:3, 3] += 0.05
    corrupted = list(sc.samples)
    corrupted[bad] = dataclasses.replace(sc.samples[bad], T_base_eef=bad_T)
    events = []
    res = hs.solve(corrupted, sc.K, None, sc.board_pts, progress_cb=events.append)
    phases = [e["phase"] for e in events]
    assert phases[0] == "start"
    rej_events = [e for e in events if e["phase"] == "rejecting"]
    assert len(rej_events) == len(res.rejected_indices)
    assert all(e["last_drop"] and "idx" in e["last_drop"] for e in rej_events)
    # the corrupted sample's drop is reported live
    assert bad in [e["last_drop"]["idx"] for e in rej_events]


def test_progress_cb_none_is_unchanged():
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    a = hs.solve(sc.samples, sc.K, None, sc.board_pts)
    b = hs.solve(sc.samples, sc.K, None, sc.board_pts, progress_cb=None)
    np.testing.assert_allclose(a.metrics["reproj_px"], b.metrics["reproj_px"])


def test_progress_cb_bug_cannot_break_solve():
    sc = syn.make_scenario(n_poses=20, pixel_noise=0.3, seed=11)
    def boom(_ev):
        raise RuntimeError("callback blew up")
    res = hs.solve(sc.samples, sc.K, None, sc.board_pts, progress_cb=boom)
    assert res.status in ("PASS", "WARN", "FAIL")  # solve completed regardless
