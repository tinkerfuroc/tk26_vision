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
"""Spec B interface smoke tests: generated types have the new fields/constants.

Requires a built + sourced ROS workspace (``source install/setup.bash``). When
the ``tinker_vision_msgs_26`` generated types are not importable (e.g. the
pure-python ``PYTHONPATH=$(pwd)`` suite run with no workspace sourced) the whole
module skips cleanly. After an interface change + rebuild, verify codegen with:
``source <ws>/install/setup.bash && pytest test/test_active_reid_interfaces.py``.
"""
import pytest

try:  # generated types live only in the built + sourced workspace
    from tinker_vision_msgs_26 import action as _action
    from tinker_vision_msgs_26 import srv as _srv
    _HAVE_MSGS = True
except ImportError:  # pure-python suite run (PYTHONPATH=$(pwd), no workspace sourced)
    _action = _srv = None
    _HAVE_MSGS = False

# Module-level skip (keeps all tests *collected*, so unsourced runs report
# "N skipped" with exit 0 in every invocation). When the workspace IS sourced,
# the attribute access inside each test fails loudly if a field/type/constant is
# missing -- so this still guards the interface contract, it does not mask it.
pytestmark = pytest.mark.skipif(
    not _HAVE_MSGS,
    reason="requires a built+sourced ROS workspace (source install/setup.bash)")


def test_trackperson_feedback_has_reacq_state():
    tp = _action.TrackPerson
    fb = tp.Feedback()
    assert hasattr(fb, "reacquisition_state")
    assert tp.Feedback.REACQ_TRACKING == 0
    assert tp.Feedback.REACQ_PASSIVE == 1
    assert tp.Feedback.REACQ_NEEDS_HELP == 2


def test_reseed_target_srv_shape():
    srv = _srv.ReseedTarget
    req, resp = srv.Request(), srv.Response()
    assert hasattr(req, "bbox") and hasattr(req, "frame_id")
    assert hasattr(resp, "success")
    assert hasattr(resp, "target_track_id")
    assert hasattr(resp, "message")


def test_detectwaving_has_waving_boxes():
    srv = _srv.DetectWaving
    assert hasattr(srv.Response(), "waving_boxes")
