"""compute_similarity's deep term must come from TargetAppearance.deep_score."""
import numpy as np

from vision_track.core.tracking_types import TargetAppearance
from vision_track.reid.reid import ReIDMatcher


def _v(*vals, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        a[i] = x
    return a


def test_compute_similarity_uses_gallery_max_view():
    t = TargetAppearance(class_id=0, class_name="person")
    t.last_seen_time = 0.0
    t.configure_gallery(enabled=True, size=6, novelty_max=0.99, score_mode="max")
    t.gallery.maybe_add(_v(1, 0))
    t.gallery.maybe_add(_v(0, 1))
    t.anchor_feature = _v(1, 0)
    cand = {"reid": _v(0, 5)}                       # aligned with gallery view 2
    sim = ReIDMatcher.compute_similarity(
        t, cand, candidate_bbox=(0, 0, 10, 20), current_time=1.0, is_person=True
    )
    assert sim > 0.0                                # gallery max view matches -> not hard-rejected
