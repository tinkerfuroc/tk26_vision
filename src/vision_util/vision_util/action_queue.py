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
"""Thread-safe FIFO execution gate for accepted action goals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Protocol


class _GoalHandle(Protocol):
    """Minimal goal-handle interface required by the gate."""

    def execute(self) -> None:
        """Schedule execution of the accepted goal."""


@dataclass
class _QueuedGoal:
    goal_handle: _GoalHandle
    canceled: bool = False


class QueuedActionGate:
    """Serialize accepted goal handles in FIFO order."""

    def __init__(self) -> None:
        """Create an empty gate."""
        self._lock = Lock()
        self._active: _GoalHandle | None = None
        self._queue: deque[_QueuedGoal] = deque()
        self._cancel_intents: list[_GoalHandle] = []

    def accept(self, goal_handle: _GoalHandle) -> None:
        """Queue an accepted goal and execute it if the gate is idle."""
        with self._lock:
            self._queue.append(_QueuedGoal(goal_handle))
            next_goal = self._take_next_locked()

        if next_goal is not None:
            next_goal.execute()

    def notify_finished(self, goal_handle: _GoalHandle) -> None:
        """Release the active goal and execute the next queued goal."""
        with self._lock:
            if self._active is not goal_handle:
                return

            self._clear_cancel_intent_locked(goal_handle)
            self._active = None
            next_goal = self._take_next_locked()

        if next_goal is not None:
            next_goal.execute()

    def cancel_queued(self, goal_handle: _GoalHandle) -> bool:
        """Mark a waiting goal canceled, returning whether it was queued."""
        with self._lock:
            for queued_goal in self._queue:
                if (
                    queued_goal.goal_handle is goal_handle
                    and not queued_goal.canceled
                ):
                    queued_goal.canceled = True
                    self._cancel_intents.append(goal_handle)
                    return True
        return False

    def should_cancel(self, goal_handle: _GoalHandle) -> bool:
        """Return whether queued or rclpy cancellation was requested."""
        with self._lock:
            if any(
                tracked_goal is goal_handle
                for tracked_goal in self._cancel_intents
            ):
                return True

        return bool(getattr(goal_handle, 'is_cancel_requested', False))

    def _take_next_locked(self) -> _GoalHandle | None:
        if self._active is not None:
            return None

        if not self._queue:
            return None

        queued_goal = self._queue.popleft()
        self._active = queued_goal.goal_handle
        return self._active

    def _clear_cancel_intent_locked(self, goal_handle: _GoalHandle) -> None:
        self._cancel_intents = [
            tracked_goal
            for tracked_goal in self._cancel_intents
            if tracked_goal is not goal_handle
        ]
