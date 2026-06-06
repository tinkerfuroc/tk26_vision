"""deep_score: gallery max-over-views when enabled, legacy max(avg,anchor) else."""
import numpy as np

from vision_track.core.tracking_types import TargetAppearance


def _v(*vals, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        a[i] = x
    return a


def test_legacy_fallback_when_gallery_empty():
    t = TargetAppearance(class_id=0, class_name="person")
    t.feature_history.append(_v(1, 0))         # avg == this
    t.configure_gallery(enabled=True, size=6, novelty_max=0.85, score_mode="max")
    assert t.deep_score(_v(2, 0)) > 0.99       # gallery empty -> cosine(avg, cand)


def test_gallery_used_when_populated():
    t = TargetAppearance(class_id=0, class_name="person")
    t.configure_gallery(enabled=True, size=6, novelty_max=0.99, score_mode="max")
    t.gallery.maybe_add(_v(1, 0))
    t.gallery.maybe_add(_v(0, 1))
    assert t.deep_score(_v(0, 5)) > 0.99       # matches the 2nd view, not the avg


def test_disabled_gallery_is_legacy():
    t = TargetAppearance(class_id=0, class_name="person")
    t.feature_history.append(_v(1, 0))
    t.configure_gallery(enabled=False, size=6, novelty_max=0.85, score_mode="max")
    t.gallery.maybe_add(_v(0, 1))              # ignored for scoring when disabled
    assert abs(t.deep_score(_v(0, 1))) < 0.05  # legacy cosine(avg=[1,0], [0,1]) ~0


def test_none_when_no_feature():
    t = TargetAppearance(class_id=0, class_name="person")
    assert t.deep_score(_v(1, 0)) is None      # no history, no gallery, no anchor
