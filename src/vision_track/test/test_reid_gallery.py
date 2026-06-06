"""Unit tests for the pure ReIDGallery (no torch / ROS)."""
import numpy as np

from vision_track.core.reid_gallery import ReIDGallery


def _vec(*vals, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        v[i] = x
    return v


def test_first_add_becomes_anchor_and_scores_one():
    g = ReIDGallery(size=6, novelty_max=0.85)
    assert g.maybe_add(_vec(1.0)) is True
    assert len(g) == 1
    assert g.score(_vec(2.0)) > 0.99  # same direction after L2 norm


def test_novelty_gate_rejects_near_duplicate():
    g = ReIDGallery(size=6, novelty_max=0.85)
    g.maybe_add(_vec(1.0, 0.0))
    assert g.maybe_add(_vec(1.0, 0.05)) is False
    assert len(g) == 1
    assert g.maybe_add(_vec(0.0, 1.0)) is True
    assert len(g) == 2


def test_bounded_size_and_anchor_pinned():
    g = ReIDGallery(size=3, novelty_max=0.99)
    dirs = [_vec(1, 0, 0), _vec(0, 1, 0), _vec(0, 0, 1), _vec(1, 1, 0), _vec(1, 0, 1)]
    for d in dirs:
        g.maybe_add(d)
    assert len(g) == 3
    assert g.score(_vec(5, 0, 0)) > 0.99  # anchor (direction e0) still present


def test_score_is_max_over_views():
    g = ReIDGallery(size=6, novelty_max=0.99)
    g.maybe_add(_vec(1, 0, 0))
    g.maybe_add(_vec(0, 1, 0))
    assert g.score(_vec(0, 3, 0)) > 0.99


def test_top2_mean_mode_is_stricter_than_max():
    g = ReIDGallery(size=6, novelty_max=0.99, score_mode="top2_mean")
    g.maybe_add(_vec(1, 0, 0))
    g.maybe_add(_vec(0, 1, 0))
    s = g.score(_vec(0, 3, 0))
    assert 0.4 < s < 0.6


def test_empty_and_disabled_return_none():
    g = ReIDGallery(size=6)
    assert g.score(_vec(1.0)) is None
    g.maybe_add(_vec(1.0))
    g.enabled = False
    assert g.score(_vec(1.0)) is None


def test_dim_mismatch_is_skipped_not_crash():
    g = ReIDGallery(size=6, novelty_max=0.99)
    g.maybe_add(np.array([1.0, 0.0], dtype=np.float32))
    assert g.score(np.array([1.0, 0.0, 0.0], dtype=np.float32)) is None
