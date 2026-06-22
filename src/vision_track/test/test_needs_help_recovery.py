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
"""Issue 1 pipeline tests: relaxed lone-candidate recovery while latched in
NEEDS_HELP.

ROOT CAUSE: while latched in NEEDS_HELP a lone returner that scores in the
[reid_threshold, single_person_commit_bar) dead band (e.g. 0.65 vs the 0.72
strict bar) is PURSUED every frame but never a confirm HIT — so it never arms
pending and never commits → stuck in NEEDS_HELP forever (only a manual
wave/reseed recovers).

THE ESCAPE HATCH (precision-bounded): ONLY while in_needs_help AND exactly one
person is visible (num_candidates == 1), relax the lone commit bar to
single_person_commit_bar_help (0.62) and require needs_help_confirm_frames (12)
confirm hits within the last needs_help_commit_window (16) frames, then commit
(id-swap), which clears the latch. OUTSIDE that gate behavior is byte-for-byte
unchanged: the strict 0.72 bar still holds for a lone candidate when not
latched, and the relaxation NEVER fires for num_candidates > 1.
"""
from types import SimpleNamespace

import pytest

import vision_track.core.tracking_pipeline as TP
from vision_track.core.tracking_types import TrackerState
from vision_track.reid.reid import ReIDMatcher


class _PersonRegistry:
    """Minimal registry stub: records clear/register calls, no-ops otherwise."""

    def __init__(self):
        self.cleared = 0
        self.registered = []

    def clear(self):
        self.cleared += 1

    def register_person(self, pid, appearance):
        self.registered.append(pid)

    def clear_temporary_ids(self):
        pass

    def get_person(self, pid):
        return None


def make_tracker(in_needs_help=False):
    """Stub tracker exposing every attribute _confirm_reid_candidate touches.

    appearance_extractor is a sentinel (non-None) so _confirm_reid_candidate
    takes the feature path; ReIDMatcher.compute_similarity is monkeypatched
    per-test to feed the controlled sim sequence, so the feature content is
    irrelevant. The Issue-1 knobs (in_needs_help + the three help-gate params)
    are exposed alongside the existing Phase-3 surface.
    """
    extractor = SimpleNamespace(
        extract_features=lambda *a, **k: {"reid": [0.0]},
        extract_features_batch=lambda *a, **k: [{"reid": [0.0]}],
    )
    tracker = SimpleNamespace(
        enable_reid=True,
        frames_lost=0,
        max_frames_lost=10_000,
        state=None,
        frame_count=0,
        target_track_id=3,
        original_track_id=3,
        target_class_id=0,
        target_class_name="person",
        target_appearance=SimpleNamespace(class_id=0, class_name="person"),
        appearance_extractor=extractor,
        person_registry=_PersonRegistry(),
        is_occluded=False,
        pre_occlusion_appearance=None,
        last_camera_motion_time=-1e9,         # no post-shake extra
        reid_threshold=0.55,
        reid_confirmation_frames=12,
        reid_preconfirm_frames=3,
        consecutive_reid_frames=0,
        pending_reid_match=None,
        reid_fit_streak=0,
        reid_fit_id=None,
        last_reid_switch_time=-1e9,
        reid_switch_cooldown=1.0,
        single_person_pursue_floor=0.55,
        single_person_commit_bar=0.72,
        provisional_commit_window=18,
        reid_confirm_window=[],
        # --- Issue 1: NEEDS_HELP relaxed-recovery knobs ---
        in_needs_help=in_needs_help,
        single_person_commit_bar_help=0.62,
        needs_help_confirm_frames=12,
        needs_help_commit_window=16,
        _with_original_id=lambda r: r,
    )
    return tracker


def _match(track_id=7):
    return SimpleNamespace(track_id=track_id, class_id=0,
                           class_name="person", bbox=(0, 0, 10, 10), mask=None)


def _drive(monkeypatch, tracker, sim_sequence, match, num_candidates=1):
    """Run _confirm_reid_candidate once per sim in the sequence.

    compute_similarity returns the scripted per-frame sim so the commit bar
    inside _confirm_reid_candidate sees exactly that value. Stops the moment a
    commit (id-swap) happens, mirroring the live loop where the swapped id is
    next picked up by track_by_id.
    """
    box = {"i": 0}

    def fake_sim(appearance, features, bbox, t, is_person=True, use_gallery=False):
        return sim_sequence[box["i"]]

    monkeypatch.setattr(ReIDMatcher, "compute_similarity", staticmethod(fake_sim))

    for i in range(len(sim_sequence)):
        box["i"] = i
        tracker.frame_count += 1
        prev = tracker.target_track_id
        TP._confirm_reid_candidate(
            tracker, frame=None, reid_match=match,
            best_similarity=sim_sequence[i], num_candidates=num_candidates,
        )
        if tracker.target_track_id != prev:
            # Commit happened this frame; stop driving.
            break


def test_needs_help_lone_relaxed_commits(monkeypatch):
    """in_needs_help=True, lone candidate, sim 0.65 (>= 0.62 help bar but <
    strict 0.72): after >=3 arming frames + 12 hits within a 16-frame window the
    id-swap commits (clearing the latch upstream)."""
    tracker = make_tracker(in_needs_help=True)
    match = _match(track_id=7)
    sims = [0.65] * 16
    _drive(monkeypatch, tracker, sims, match, num_candidates=1)

    # The relaxed help path committed the id-swap.
    assert tracker.target_track_id == 7
    assert tracker.state == TrackerState.TRACKING


def test_needs_help_lone_below_help_bar_never_commits(monkeypatch):
    """in_needs_help=True, lone candidate, sim 0.60 (< 0.62 help bar): never a
    confirm hit → never commits even after many frames."""
    tracker = make_tracker(in_needs_help=True)
    match = _match(track_id=7)
    sims = [0.60] * 40
    _drive(monkeypatch, tracker, sims, match, num_candidates=1)

    assert tracker.target_track_id == 3  # never committed
    assert tracker.pending_reid_match is None  # never even armed


def test_not_in_help_lone_strict_bar_holds(monkeypatch):
    """in_needs_help=False, lone candidate, sim 0.65 (< strict 0.72): the strict
    path holds the commit bar at 0.72, so 0.65 is never a hit → no commit."""
    tracker = make_tracker(in_needs_help=False)
    match = _match(track_id=7)
    sims = [0.65] * 40
    _drive(monkeypatch, tracker, sims, match, num_candidates=1)

    assert tracker.target_track_id == 3  # strict bar held; no swap
    assert tracker.pending_reid_match is None  # never armed at the relaxed bar


def test_needs_help_multi_candidate_does_not_relax(monkeypatch):
    """in_needs_help=True but num_candidates==2: the relaxed lone path is NOT
    used. The multi commit bar is reid_threshold (0.55), so 0.65 frames DO clear
    it — but the relaxation (sub-0.72 lone bar) is the thing under test, and the
    KEY invariant is that the relaxed LONE path never fires when >1 person is
    visible. Drive sub-help-bar (0.58) sims so that, were the help path wrongly
    taken (bar 0.62), there would be no hit; multi path (bar 0.55) commits and we
    assert that commit went through the MULTI bar, not the relaxed lone bar."""
    tracker = make_tracker(in_needs_help=True)
    match = _match(track_id=7)
    # 0.58 is >= reid_threshold (0.55, the MULTI commit bar) but < the help bar
    # (0.62). With num_candidates==2 the relaxed lone path must NOT engage; the
    # commit (if any) must come from the multi bar = reid_threshold.
    sims = [0.58] * 18
    _drive(monkeypatch, tracker, sims, match, num_candidates=2)

    # Multi commit bar (reid_threshold) was used → committed at 0.58.
    assert tracker.target_track_id == 7
    # Sanity: had the relaxed LONE help bar (0.62) been applied here, 0.58 would
    # never have cleared it and no commit would occur. The commit proves the
    # multi path (not the relaxed lone path) governed this case.


def test_needs_help_lone_sub_072_uses_help_bar_not_strict(monkeypatch):
    """Cross-check: the SAME sim 0.65 that COMMITS while in_needs_help=True must
    NOT commit while in_needs_help=False — proving the relaxation is gated on the
    latch, not always-on."""
    helped = make_tracker(in_needs_help=True)
    strict = make_tracker(in_needs_help=False)
    match = _match(track_id=7)
    sims = [0.65] * 16
    _drive(monkeypatch, helped, sims, match, num_candidates=1)
    _drive(monkeypatch, strict, list(sims), match, num_candidates=1)

    assert helped.target_track_id == 7   # relaxed path committed
    assert strict.target_track_id == 3   # strict path held


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
