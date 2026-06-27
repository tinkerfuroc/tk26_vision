import numpy as np
from handeye_calib import handeye_solve as hs


def test_consensus_corners_denoises_toward_truth():
    rng = np.random.default_rng(0)
    truth = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
    ids = np.array([0, 1, 2, 3])
    frames = [(ids, truth + rng.normal(0, 0.5, truth.shape)) for _ in range(10)]
    out_ids, out_px = hs.consensus_corners(frames)
    assert list(out_ids) == [0, 1, 2, 3]
    # consensus error well below any single frame's ~0.5 px noise
    assert np.max(np.linalg.norm(out_px - truth, axis=1)) < 0.3


def test_consensus_drops_below_quorum_corners():
    ids_full = np.array([0, 1, 2, 3])
    px = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    # corner id 3 appears in only 1 of 10 frames -> below 60% quorum, dropped.
    frames = [(np.array([0, 1, 2]), px[:3]) for _ in range(9)]
    frames.append((ids_full, px))
    out_ids, out_px = hs.consensus_corners(frames)
    assert list(out_ids) == [0, 1, 2]


def test_consensus_returns_none_on_empty_frames():
    out_ids, out_px = hs.consensus_corners([])
    assert out_ids is None and out_px is None


def test_consensus_returns_none_when_no_corner_reaches_quorum():
    # each corner appears in only 1 of 5 frames -> below ceil(0.6*5)=3 quorum
    frames = [(np.array([i]), np.array([[float(i), 0.0]])) for i in range(5)]
    out_ids, out_px = hs.consensus_corners(frames)
    assert out_ids is None and out_px is None
