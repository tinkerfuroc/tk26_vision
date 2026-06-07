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
"""Unit tests for the pure reacquisition-state hysteresis."""
from vision_track.core.reacq_state import (
    REACQ_TRACKING, REACQ_PASSIVE, REACQ_NEEDS_HELP, reacq_state,
)


def test_tracked_is_tracking():
    assert reacq_state(tracked=True, frames_lost=0, help_after=45) == REACQ_TRACKING
    assert reacq_state(tracked=True, frames_lost=999, help_after=45) == REACQ_TRACKING


def test_lost_within_window_is_passive():
    assert reacq_state(tracked=False, frames_lost=1, help_after=45) == REACQ_PASSIVE
    assert reacq_state(tracked=False, frames_lost=44, help_after=45) == REACQ_PASSIVE


def test_lost_past_window_needs_help():
    assert reacq_state(tracked=False, frames_lost=45, help_after=45) == REACQ_NEEDS_HELP
    assert reacq_state(tracked=False, frames_lost=200, help_after=45) == REACQ_NEEDS_HELP


def test_help_after_zero_escalates_immediately_when_lost():
    assert reacq_state(tracked=False, frames_lost=0, help_after=0) == REACQ_NEEDS_HELP
