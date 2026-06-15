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
