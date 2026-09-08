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
"""Tests for the FIFO action-goal execution gate."""

import unittest

from vision_util.action_queue import QueuedActionGate


class FakeGoalHandle:
    """Small rclpy-free goal handle used to record execution."""

    def __init__(self, name, execution_order):
        self.name = name
        self.execution_order = execution_order
        self.execute_calls = 0
        self.is_cancel_requested = False

    def execute(self):
        self.execute_calls += 1
        self.execution_order.append(self.name)


class TestQueuedActionGate(unittest.TestCase):
    def setUp(self):
        self.execution_order = []
        self.gate = QueuedActionGate()

    def make_goal(self, name):
        return FakeGoalHandle(name, self.execution_order)

    def test_first_accepted_goal_executes_immediately(self):
        first = self.make_goal('first')

        self.gate.accept(first)

        self.assertEqual(first.execute_calls, 1)
        self.assertEqual(self.execution_order, ['first'])

    def test_later_goals_wait_and_execute_in_fifo_order(self):
        first = self.make_goal('first')
        second = self.make_goal('second')
        third = self.make_goal('third')

        self.gate.accept(first)
        self.gate.accept(second)
        self.gate.accept(third)

        self.assertEqual(self.execution_order, ['first'])

        self.gate.notify_finished(first)
        self.assertEqual(self.execution_order, ['first', 'second'])

        self.gate.notify_finished(second)
        self.assertEqual(self.execution_order, ['first', 'second', 'third'])

    def test_cancel_queued_executes_for_lifecycle_before_next_goal(self):
        first = self.make_goal('first')
        canceled = self.make_goal('canceled')
        third = self.make_goal('third')
        self.gate.accept(first)
        self.gate.accept(canceled)
        self.gate.accept(third)

        self.assertTrue(self.gate.cancel_queued(canceled))
        self.assertFalse(self.gate.cancel_queued(canceled))
        self.assertTrue(self.gate.should_cancel(canceled))

        self.gate.notify_finished(first)

        self.assertEqual(canceled.execute_calls, 1)
        self.assertEqual(third.execute_calls, 0)
        self.assertEqual(self.execution_order, ['first', 'canceled'])
        self.assertTrue(self.gate.should_cancel(canceled))

        self.gate.notify_finished(canceled)

        self.assertEqual(canceled.execute_calls, 1)
        self.assertEqual(third.execute_calls, 1)
        self.assertEqual(self.execution_order, ['first', 'canceled', 'third'])
        self.assertFalse(self.gate.should_cancel(canceled))

    def test_active_goal_cannot_be_canceled_as_queued(self):
        first = self.make_goal('first')
        self.gate.accept(first)

        self.assertFalse(self.gate.cancel_queued(first))
        self.assertEqual(first.execute_calls, 1)

    def test_should_cancel_consults_goal_handle_state(self):
        first = self.make_goal('first')
        self.gate.accept(first)

        self.assertFalse(self.gate.should_cancel(first))

        first.is_cancel_requested = True

        self.assertTrue(self.gate.should_cancel(first))

    def test_double_notify_finished_is_harmless(self):
        first = self.make_goal('first')
        second = self.make_goal('second')
        third = self.make_goal('third')
        self.gate.accept(first)
        self.gate.accept(second)
        self.gate.accept(third)

        self.gate.notify_finished(first)
        self.gate.notify_finished(first)

        self.assertEqual(second.execute_calls, 1)
        self.assertEqual(third.execute_calls, 0)

        self.gate.notify_finished(second)
        self.assertEqual(self.execution_order, ['first', 'second', 'third'])


if __name__ == '__main__':
    unittest.main()
