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
"""TrackWebNode.wave() auto-reseed: a single detected waver resumes tracking
without a manual click; multiple wavers stay manual (operator picks)."""
from types import SimpleNamespace

import pytest

TrackWebNode = pytest.importorskip("vision_track.track_web").TrackWebNode
wave = TrackWebNode.wave


def _box(x, y, w, h):
    return SimpleNamespace(x_offset=x, y_offset=y, width=w, height=h)


def _node(resp, reseed_ret, on_reseed=None):
    def _reseed(box):
        if on_reseed is not None:
            on_reseed(box)
        return reseed_ret
    return SimpleNamespace(
        _wave_cli=None,
        _call=lambda cli, req, timeout=0, name="": (resp, None),
        reseed=_reseed,
    )


def test_single_waver_auto_reseeds():
    resp = SimpleNamespace(status=0, waving_boxes=[_box(10, 20, 30, 40)],
                           waving_persons=[])
    node = _node(resp, {"success": True, "target_track_id": 7, "message": "ok"})
    out = wave(node)
    assert out["auto_reseeded"] is True
    assert out["reseed"]["target_track_id"] == 7


def test_single_waver_auto_reseed_inherits_gate_via_shared_service():
    """Issue 2: the waving auto-reseed calls node.reseed (the same
    ~/reseed_target service that hits _apply_reseed), so it inherits the reseed
    confirmation gate by construction — no separate instant-lock path exists.

    A 'success' from reseed_target now means 'accepted, confirming' (probation
    armed), not 'locked' — the lock commits only after the per-frame appearance
    confirmation (covered by test_reseed_probation)."""
    resp = SimpleNamespace(status=0, waving_boxes=[_box(10, 20, 30, 40)],
                           waving_persons=[])
    seen = []
    node = _node(resp, {"success": True, "target_track_id": 7, "message": "ok"},
                 on_reseed=lambda box: seen.append(box))
    out = wave(node)
    # The single waver routed through the shared reseed service exactly once.
    assert out["auto_reseeded"] is True
    assert len(seen) == 1
    # The box passed to reseed is the waver's box (x1,y1,x2,y2) — the same
    # plumbing a manual dashboard click uses — so both triggers share
    # _apply_reseed's probation gate.
    assert seen[0] == [10, 20, 40, 60]


def test_multiple_wavers_stay_manual():
    resp = SimpleNamespace(
        status=0, waving_boxes=[_box(10, 20, 30, 40), _box(80, 20, 30, 40)],
        waving_persons=[])
    # reseed must NOT be auto-called when the waver is ambiguous
    node = _node(resp, {"success": True},
                 on_reseed=lambda b: pytest.fail("must not auto-reseed >1 waver"))
    out = wave(node)
    assert "auto_reseeded" not in out
    assert len(out["boxes"]) == 2


def test_no_waver_no_reseed():
    resp = SimpleNamespace(status=1, waving_boxes=[], waving_persons=[])
    node = _node(resp, {"success": True},
                 on_reseed=lambda b: pytest.fail("must not reseed with no waver"))
    out = wave(node)
    assert "auto_reseeded" not in out
    assert out["boxes"] == []


def test_service_error_surfaces():
    node = SimpleNamespace(
        _wave_cli=None,
        _call=lambda cli, req, timeout=0, name="": (None, "service unavailable"),
        reseed=lambda b: pytest.fail("must not reseed on service error"))
    out = wave(node)
    assert out["status"] == -1
    assert "error" in out
