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
"""Contract for the ~/reacq_state heartbeat sentinel.

The tracker publishes a permanently-on low-rate UInt8 heartbeat that mirrors
the TrackPerson feedback reacquisition enum (0 TRACKING / 1 PASSIVE /
2 NEEDS_HELP) and reports 255 INACTIVE whenever no TrackPerson goal is active.
The navigation follow executive consumes this to distinguish a "wait, don't
abort" period (PASSIVE/NEEDS_HELP) from tracker shutdown (INACTIVE), with no
dependency on tinker_vision_msgs_26. INACTIVE must stay a value the enum never
takes (0/1/2), so the consumer can branch on it unambiguously.
"""
from vision_track.core.reacq_state import (
    REACQ_NEEDS_HELP, REACQ_PASSIVE, REACQ_TRACKING,
)
from vision_track.person_track_node import REACQ_INACTIVE


def test_reacq_inactive_sentinel():
    assert REACQ_INACTIVE == 255


def test_reacq_inactive_distinct_from_active_enum():
    # The sentinel must never collide with a live feedback value, or the
    # consumer can't tell "no goal" from a real tracking/passive/help state.
    assert REACQ_INACTIVE not in (REACQ_TRACKING, REACQ_PASSIVE, REACQ_NEEDS_HELP)


def test_reacq_inactive_fits_uint8():
    # Published as std_msgs/UInt8 — the sentinel must be a valid byte.
    assert 0 <= REACQ_INACTIVE <= 255
