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

"""Unit tests for the track_web ProcessManager (fixed-allowlist supervisor).

Uses real short-lived shell commands (no ROS) so the test is fast and
hermetic: `sleep 30` for the long-lived case, `true` for the quick-exit case,
and `bogus` for the unknown-name guard.
"""

import time

import pytest

from vision_track.process_manager import ProcessManager


@pytest.fixture()
def mgr():
    """A ProcessManager with a safe, ROS-free registry; always reaped."""
    m = ProcessManager(registry={"sleeper": ["sleep", "30"], "quick": ["true"]})
    try:
        yield m
    finally:
        m.shutdown_all()


def _alive(pid):
    """Return True if pid is alive (os.kill(pid, 0) doesn't raise)."""
    import os
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_start_running_and_pid(mgr):
    st = mgr.start("sleeper")
    assert st["running"] is True
    assert isinstance(st["pid"], int)
    assert _alive(st["pid"]) is True


def test_start_is_idempotent(mgr):
    first = mgr.start("sleeper")
    second = mgr.start("sleeper")
    assert second["pid"] == first["pid"]
    assert second["running"] is True


def test_start_unknown_returns_error_no_raise(mgr):
    st = mgr.start("bogus")
    assert "error" in st
    assert st["name"] == "bogus"


def test_stop_actually_kills(mgr):
    st = mgr.start("sleeper")
    pid = st["pid"]
    stopped = mgr.stop("sleeper")
    assert stopped["running"] is False
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_quick_process_reaps_with_returncode_zero(mgr):
    mgr.start("quick")
    deadline = time.time() + 2.0
    st = mgr.status("quick")
    while st["running"] and time.time() < deadline:
        time.sleep(0.02)
        st = mgr.status("quick")
    assert st["running"] is False
    assert st["returncode"] == 0


def test_shutdown_all_stops_running(mgr):
    mgr.start("sleeper")
    mgr.shutdown_all()
    assert mgr.status("sleeper")["running"] is False
