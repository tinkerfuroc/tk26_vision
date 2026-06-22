"""_update_feature_history must also feed the multi-view gallery (novelty-gated)."""
import numpy as np

from vision_track.core.tracking_types import TargetAppearance
from vision_track.reid.appearance_manager import _update_feature_history


class _Tracker:
    def __init__(self):
        self.target_appearance = TargetAppearance(class_id=0, class_name="person")
        self.target_appearance.configure_gallery(
            enabled=True, size=6, novelty_max=0.85, score_mode="max")
        self.original_track_id = None


def _v(*vals, dim=8):
    a = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        a[i] = x
    return a


def test_distinct_views_populate_gallery():
    trk = _Tracker()
    _update_feature_history(trk, {"reid": _v(1, 0)}, 0.9, 1.0, True)
    _update_feature_history(trk, {"reid": _v(0, 1)}, 0.9, 1.0, True)   # distinct
    assert len(trk.target_appearance.gallery) == 2


def test_near_duplicate_does_not_grow_gallery():
    trk = _Tracker()
    _update_feature_history(trk, {"reid": _v(1, 0.0)}, 0.9, 1.0, True)
    _update_feature_history(trk, {"reid": _v(1, 0.02)}, 0.9, 1.0, True)  # ~dup
    assert len(trk.target_appearance.gallery) == 1
