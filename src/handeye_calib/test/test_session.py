import json

import numpy as np
from handeye_calib import synthetic as syn
from handeye_calib.handeye_collect import CaptureSession


def test_session_accepts_diverse_rejects_redundant():
    sc = syn.make_scenario(n_poses=6, pixel_noise=0.0, seed=0)
    sess = CaptureSession(min_diversity_deg=30.0)
    # First sample always accepted.
    s0 = sc.samples[0]
    assert sess.try_add(s0.T_base_eef, s0.T_cam_board, s0.obs_px, s0.corner_idx,
                        n_corners=16, reproj_px=0.5, area_frac=0.3)[0] is True
    # The same flange pose again -> not diverse -> rejected.
    ok, reason = sess.try_add(s0.T_base_eef, s0.T_cam_board, s0.obs_px, s0.corner_idx,
                              n_corners=16, reproj_px=0.5, area_frac=0.3)
    assert ok is False and "diver" in reason.lower()
    assert len(sess.samples) == 1


def test_session_rejects_low_quality():
    sc = syn.make_scenario(n_poses=3, pixel_noise=0.0, seed=1)
    sess = CaptureSession(min_diversity_deg=30.0)
    s = sc.samples[0]
    ok, reason = sess.try_add(s.T_base_eef, s.T_cam_board, s.obs_px, s.corner_idx,
                              n_corners=4, reproj_px=0.5, area_frac=0.3)
    assert ok is False and "corner" in reason.lower()


def test_session_accumulates_multiple_diverse():
    sc = syn.make_scenario(n_poses=6, pixel_noise=0.0, seed=0)
    sess = CaptureSession(min_diversity_deg=30.0)
    accepted = 0
    for s in sc.samples:
        ok, _ = sess.try_add(s.T_base_eef, s.T_cam_board, s.obs_px, s.corner_idx,
                             n_corners=16, reproj_px=0.5, area_frac=0.3)
        accepted += int(ok)
    assert accepted == len(sess.samples) and len(sess.samples) >= 2


def test_session_to_json_roundtrips():
    sc = syn.make_scenario(n_poses=4, pixel_noise=0.0, seed=0)
    sess = CaptureSession(min_diversity_deg=30.0)
    s = sc.samples[0]
    sess.try_add(s.T_base_eef, s.T_cam_board, s.obs_px, s.corner_idx,
                 n_corners=16, reproj_px=0.5, area_frac=0.3)
    rec = json.loads(sess.to_json())
    assert len(rec) == 1
    r = rec[0]
    np.testing.assert_allclose(np.array(r["T_base_eef"]), s.T_base_eef, atol=1e-12)
    np.testing.assert_allclose(np.array(r["T_cam_board"]), s.T_cam_board, atol=1e-12)
    assert np.array(r["obs_px"]).shape == s.obs_px.shape
    assert np.array(r["corner_idx"]).tolist() == s.corner_idx.tolist()
    # Back-compat: no depth supplied -> stored None, serialized as null.
    assert sess.samples[0].obs_xyz_cam is None
    assert r["obs_xyz_cam"] is None and r["obs_xyz_valid"] is None


def test_session_stores_and_serializes_ffs_depth():
    sc = syn.make_scenario(n_poses=4, pixel_noise=0.0, seed=0)
    sess = CaptureSession(min_diversity_deg=30.0)
    s = sc.samples[0]
    M = len(s.corner_idx)
    xyz = np.random.default_rng(0).uniform(-0.1, 0.6, size=(M, 3))
    valid = np.ones(M, bool)
    valid[0] = False
    ok, _ = sess.try_add(s.T_base_eef, s.T_cam_board, s.obs_px, s.corner_idx,
                         n_corners=16, reproj_px=0.5, area_frac=0.3,
                         obs_xyz_cam=xyz, obs_xyz_valid=valid)
    assert ok
    stored = sess.samples[0]
    np.testing.assert_allclose(stored.obs_xyz_cam, xyz, atol=1e-12)
    assert stored.obs_xyz_valid.tolist() == valid.tolist()
    r = json.loads(sess.to_json())[0]
    np.testing.assert_allclose(np.array(r["obs_xyz_cam"]), xyz, atol=1e-12)
    assert np.array(r["obs_xyz_valid"]).tolist() == valid.tolist()
