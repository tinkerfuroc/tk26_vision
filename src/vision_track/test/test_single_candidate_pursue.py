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
"""Phase 3 / Option B: lone-candidate PURSUE floor in _single_candidate_guard.

A lone returning operator scoring ReID ~0.55-0.71 with occasional dips used to be
discarded by the hard-coded 0.72 wall before it could accumulate anything. The
pursue floor (default reid_threshold 0.55) keeps the look-alike IN PLAY (so its
good frames can accumulate) without authorising a commit — the commit bar is
enforced downstream in _confirm_reid_candidate.
"""
from types import SimpleNamespace

from vision_track.reid.reid_search import _single_candidate_guard


def _lone(sim):
    """One scored candidate tuple (result, similarity, features, raw_cosine)."""
    r = SimpleNamespace(track_id=7, class_id=0)
    return [(r, sim, {}, sim)]


def test_lone_above_floor_is_pursued():
    """sim 0.60 >= pursue_floor 0.55 → guard True (kept in play, not discarded)."""
    assert _single_candidate_guard(True, _lone(0.60), 0.60, 0.55) is True


def test_lone_below_floor_is_discarded():
    """sim 0.50 < pursue_floor 0.55 → guard False (discarded)."""
    assert _single_candidate_guard(True, _lone(0.50), 0.50, 0.55) is False


def test_lone_well_above_floor_is_pursued():
    """sim 0.80 >= pursue_floor → True (unchanged behaviour for confident frames)."""
    assert _single_candidate_guard(True, _lone(0.80), 0.80, 0.55) is True


def test_lone_in_old_dead_zone_now_pursued():
    """sim 0.65 was previously discarded by the 0.72 wall; now pursued."""
    assert _single_candidate_guard(True, _lone(0.65), 0.65, 0.55) is True


def test_multi_candidate_unaffected():
    """The guard only applies to the lone case; >1 candidate always passes."""
    r1 = SimpleNamespace(track_id=7, class_id=0)
    r2 = SimpleNamespace(track_id=8, class_id=0)
    scores = [(r1, 0.60, {}, 0.60), (r2, 0.40, {}, 0.40)]
    assert _single_candidate_guard(True, scores, 0.60, 0.55) is True


def test_non_person_unaffected():
    """Non-person target → guard always True (no single-person logic)."""
    assert _single_candidate_guard(False, _lone(0.50), 0.50, 0.55) is True


def test_custom_floor_respected():
    """The floor is threaded in, not hard-coded: 0.65 floor rejects 0.60."""
    assert _single_candidate_guard(True, _lone(0.60), 0.60, 0.65) is False
    assert _single_candidate_guard(True, _lone(0.66), 0.66, 0.65) is True
