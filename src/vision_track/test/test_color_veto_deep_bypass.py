# Copyright 2026 Tinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deep-gated color veto (Change 2).

The three hard color vetoes in ReIDMatcher._compute_person_similarity used to
``return 0.0`` the instant a color histogram intersection dipped below 0.40 —
force-zeroing a strong deep match (top cause of correct matches never
re-locking). The veto is now gated on deep confidence: a gallery deep cosine
>= DEEP_CONFIDENT_BYPASS (0.70) bypasses the color floor, while a low-deep
bystander (still < 0.70) is rejected exactly as before. Precision preserved.

These tests drive the real classmethod with a minimal FAKE target (no heavy
models) so the actual veto logic — including the DEEP_CONFIDENT_BYPASS gate — is
exercised end to end. Thresholds are NOT touched here.
"""
import numpy as np
import pytest

reid = pytest.importorskip("vision_track.reid.reid")
ReIDMatcher = reid.ReIDMatcher


class FakeTarget:
    """Minimal stand-in for TargetAppearance exposing only what
    _compute_person_similarity / compute_similarity touch."""

    def __init__(self, deep_value):
        self._deep_value = deep_value
        # compute_similarity reads last_seen_time for time decay; keep it recent.
        self.last_seen_time = 0.0
        # body-color path attrs (we trip the body-color veto).
        self._body_color = np.array([1.0, 0.0], dtype=np.float32)
        self.anchor_body_color = None
        # general-color + upper/lower + size paths: feed nothing so they're skipped.
        self.anchor_color_hist = None
        self.anchor_upper_color = None
        self.upper_color_history = []
        self.anchor_lower_color = None
        self.lower_color_history = []

    def deep_score(self, candidate_reid, use_gallery=True):
        return self._deep_value

    def get_body_color(self):
        return self._body_color

    def get_average_color_hist(self):
        return None

    def get_average_size(self):
        return None


def _candidate_low_body_color():
    """A candidate whose body_color histogram is disjoint from the target's,
    so histogram intersection is 0.0 (well below the 0.40 floor)."""
    return {
        "reid": np.array([0.1, 0.2, 0.3], dtype=np.float32),  # value unused; deep_score is faked
        "body_color": np.array([0.0, 1.0], dtype=np.float32),  # disjoint -> intersection 0.0
    }


def test_high_deep_low_color_not_vetoed():
    # Deep clearly confident (>= 0.70 bypass) but body color is 0.0 (< 0.40
    # floor): the veto must be bypassed and a non-zero score returned.
    target = FakeTarget(deep_value=0.90)
    sim = ReIDMatcher.compute_similarity(
        target,
        _candidate_low_body_color(),
        candidate_bbox=(0, 0, 50, 100),
        current_time=0.0,
        is_person=True,
    )
    assert sim > 0.0


def test_low_deep_low_color_vetoed():
    # Deep below the 0.70 bypass but above MIN_REID_SIMILARITY_RAW (0.40) so the
    # raw-deep floor does NOT reject; the color veto must still fire -> 0.0.
    assert 0.40 <= 0.50 < ReIDMatcher.DEEP_CONFIDENT_BYPASS  # sanity on the band
    target = FakeTarget(deep_value=0.50)
    sim = ReIDMatcher.compute_similarity(
        target,
        _candidate_low_body_color(),
        candidate_bbox=(0, 0, 50, 100),
        current_time=0.0,
        is_person=True,
    )
    assert sim == 0.0


def test_thresholds_unchanged():
    # Guard the constants this change is contractually NOT allowed to move.
    assert ReIDMatcher.DEEP_CONFIDENT_BYPASS == 0.70
    assert ReIDMatcher.MIN_REID_SIMILARITY_RAW == 0.40
    assert ReIDMatcher.MIN_BODY_COLOR_SIMILARITY == 0.40
    assert ReIDMatcher.MIN_UPPER_SIMILARITY == 0.40
    assert ReIDMatcher.MIN_LOWER_SIMILARITY == 0.40
