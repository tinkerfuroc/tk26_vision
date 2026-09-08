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
"""build_debug_state: pure, defensive snapshot of tracker state for the dashboard."""
import json
from types import SimpleNamespace

from vision_track.core.debug_state import build_debug_state


class _Gal:
    """Real-shaped gallery double (class-level __len__, like ReIDGallery)."""

    version = 5

    def __len__(self):
        return 2


def _bare(**over):
    t = SimpleNamespace(
        last_lock_decision=SimpleNamespace(state="reidentifying"),
        frames_lost=12,
        target_track_id=3,
        original_track_id=3,
        last_results=[
            SimpleNamespace(class_id=0, track_id=3, bbox=(10, 10, 50, 120)),
            SimpleNamespace(class_id=0, track_id=7, bbox=(60, 12, 110, 130)),
            SimpleNamespace(class_id=39, track_id=9, bbox=(0, 0, 5, 5)),   # not a person
            SimpleNamespace(class_id=0, track_id=-1, bbox=(0, 0, 9, 9)),   # untracked det
            SimpleNamespace(class_id=0, track_id=11, bbox=None),           # boxless track
        ],
        last_debug_scores={3: 0.81, 7: 0.44},
        target_appearance=SimpleNamespace(gallery=_Gal()),
    )
    for k, v in over.items():
        setattr(t, k, v)
    return t


def _kw(**over):
    kw = dict(ts=123.0, target_lost=True, reacquisition_state=1,
              time_since_seen=0.8, awaiting_help=False,
              active_help_after_sec=5.0, active_help_timeout_sec=20.0)
    kw.update(over)
    return kw


def test_full_snapshot():
    d = build_debug_state(_bare(), **_kw())
    assert d["ts"] == 123.0 and d["target_lost"] is True
    assert d["fsm_state"] == "reidentifying"
    assert d["reacquisition_state"] == 1 and d["frames_lost"] == 12
    assert d["awaiting_help"] is False and d["active_help_timeout_sec"] == 20.0
    assert d["active_help_after_sec"] == 5.0 and isinstance(d["active_help_after_sec"], float)
    assert d["target_track_id"] == 3 and d["original_track_id"] == 3
    # persons with a real track id only; scores joined on id; bbox may be None
    assert d["candidates"] == [
        {"id": 3, "bbox": [10, 10, 50, 120], "score": 0.81},
        {"id": 7, "bbox": [60, 12, 110, 130], "score": 0.44},
        {"id": 11, "bbox": None, "score": None},
    ]
    assert d["best_sim"] == 0.81 and d["second_sim"] == 0.44
    assert d["gallery_len"] == 2 and d["gallery_version"] == 5
    json.dumps(d)   # the module's core contract: JSON-serializable


def test_defensive_on_bare_tracker():
    t = SimpleNamespace()  # nothing set at all
    d = build_debug_state(t, **_kw(target_lost=False, reacquisition_state=0))
    assert d["fsm_state"] is None and d["frames_lost"] == 0
    assert d["candidates"] == [] and d["best_sim"] is None and d["second_sim"] is None
    assert d["gallery_len"] == 0 and d["gallery_version"] == 0


def test_no_scores_yields_nulls():
    d = build_debug_state(_bare(last_debug_scores={}), **_kw())
    assert d["candidates"][0]["score"] is None
    assert d["best_sim"] is None and d["second_sim"] is None


def _init_payload(tracker, *, search_started_ts):
    """Reconstruct the exact 'initializing' phase payload that
    PersonTrackNode._publish_phase_debug_state emits (build_debug_state + the
    node's phase overrides). Mirrors the node so the contract is unit-tested
    without ROS."""
    state = build_debug_state(tracker, **_kw())
    state["fsm_state"] = "initializing"
    state["candidates"] = []
    state["best_sim"] = None
    state["second_sim"] = None
    state["search_started_ts"] = search_started_ts
    return state


def test_initializing_payload_carries_search_started_ts():
    # Even with stale candidates/scores on the tracker, the init phase payload
    # must show empty candidates, null sims, and a wall-clock search anchor so
    # the dashboard renders an alive "Searching…" timer instead of dead "—"s.
    d = _init_payload(_bare(), search_started_ts=123.5)
    assert d["fsm_state"] == "initializing"
    assert d["candidates"] == []
    assert d["best_sim"] is None and d["second_sim"] is None
    assert d["search_started_ts"] == 123.5
    json.dumps(d)   # still JSON-serializable


def test_idle_payload_has_no_search_timer():
    # Between goals (idle) there is no search anchor — None so the webui doesn't
    # surface a stale elapsed timer.
    d = _init_payload(_bare(), search_started_ts=None)
    d["fsm_state"] = "idle"
    assert d["search_started_ts"] is None
