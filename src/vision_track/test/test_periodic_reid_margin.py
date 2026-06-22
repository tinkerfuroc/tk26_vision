"""The periodic-ReID switch requires a >=0.15 similarity margin.

This is a behavioral pin on tracking_pipeline.periodic_reid_validation, which
computes ``margin_required = max(ReIDMatcher.REID_MARGIN, 0.15)``. Task 7 raised
that inline literal floor from 0.08 to 0.15.

To make the *inline literal* the binding constraint (and not merely shadow the
current ``REID_MARGIN == 0.15`` source value, which would let the test pass even
if the literal were reverted to 0.08), every test below forces
``ReIDMatcher.REID_MARGIN`` BELOW the literal (0.05). That isolates the literal:
``max(0.05, 0.15) == 0.15`` is governed entirely by the inline ``0.15``. A
candidate beating the lock by 0.10 must NOT switch (it would switch under an
0.08 literal — that is the regression this guards); a candidate beating it by
0.20 must switch.
"""
from types import SimpleNamespace

import numpy as np

from vision_track.core import tracking_pipeline as tp
from vision_track.core.tracking_types import TrackingResult


class _StubExtractor:
    def extract_features(self, frame, bbox, mask, class_id=0):
        return [1.0]  # truthy, non-empty


def _make_tracker(best_match, best_similarity):
    """A minimal duck-typed tracker for periodic_reid_validation."""
    tracker = SimpleNamespace()
    tracker.reid_verification_interval = 1
    tracker.frame_count = 1
    tracker.enable_reid = True
    tracker.target_appearance = object()
    tracker.appearance_extractor = _StubExtractor()
    tracker.reid_threshold = 0.5
    tracker._find_best_match_reid = lambda frame, results: (best_match, best_similarity)
    return tracker


def _res(track_id):
    return TrackingResult(
        track_id=track_id, bbox=(0, 0, 10, 20), mask=None,
        confidence=0.9, class_id=0, class_name="person",
    )


def _force_reid_margin_low(monkeypatch):
    # Force REID_MARGIN below the inline literal so margin_required =
    # max(REID_MARGIN, 0.15) is decided by the 0.15 literal, not the source
    # constant. This makes the assertions below sensitive ONLY to the literal:
    # if the literal were reverted to 0.08, the 0.10-margin case would switch and
    # the "does not switch" test would fail.
    monkeypatch.setattr(tp.ReIDMatcher, "REID_MARGIN", 0.05)


def test_margin_below_015_does_not_switch(monkeypatch):
    # REID_MARGIN forced to 0.05 → margin_required = max(0.05, 0.15) = 0.15.
    # current similarity 0.60, candidate 0.70 → margin 0.10.
    # 0.10 < 0.15 literal floor → no switch. (Under an 0.08 literal this WOULD
    # switch — that is the regression Task 7 guards against.)
    _force_reid_margin_low(monkeypatch)
    monkeypatch.setattr(tp.ReIDMatcher, "compute_similarity",
                        staticmethod(lambda *a, **k: 0.60))
    cand = _res(2)
    tracker = _make_tracker(best_match=cand, best_similarity=0.70)
    cur = _res(1)
    ok, switch_to = tp.periodic_reid_validation(tracker, np.zeros((20, 10, 3)), [cur, cand], cur)
    assert ok is True
    assert switch_to is None


def test_margin_at_or_above_015_switches(monkeypatch):
    # REID_MARGIN forced to 0.05 → margin_required = max(0.05, 0.15) = 0.15.
    # current 0.60, candidate 0.80 → margin 0.20 >= 0.15 and > threshold → switch.
    _force_reid_margin_low(monkeypatch)
    monkeypatch.setattr(tp.ReIDMatcher, "compute_similarity",
                        staticmethod(lambda *a, **k: 0.60))
    cand = _res(2)
    tracker = _make_tracker(best_match=cand, best_similarity=0.80)
    cur = _res(1)
    ok, switch_to = tp.periodic_reid_validation(tracker, np.zeros((20, 10, 3)), [cur, cand], cur)
    assert ok is False
    assert switch_to is cand
