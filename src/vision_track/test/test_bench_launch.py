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
"""track_web_bench launch file: pure-generation smoke test (no nodes started)."""
import importlib.util
from pathlib import Path

import pytest

launch = pytest.importorskip(
    "launch", reason="launch/launch_ros not importable (no ROS on path)")
pytest.importorskip(
    "launch_ros", reason="launch/launch_ros not importable (no ROS on path)")


def _load_ld():
    path = Path(__file__).resolve().parents[1] / "launch" / "track_web_bench.launch.py"
    spec = importlib.util.spec_from_file_location("track_web_bench_launch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_launch_description()


def _on_exit_nodes(entities):
    """Return the Node actions nested in the OnProcessExit event handler's
    on_exit list — the canonical post-cleanup bench-node declaration."""
    from launch.actions import RegisterEventHandler
    from launch_ros.actions import Node
    for e in entities:
        if isinstance(e, RegisterEventHandler):
            handler = e.event_handler
            # OnProcessExit (Humble) stores exit actions as a name-mangled attr
            # on its base class OnActionEventBase; try all known layouts.
            on_exit = getattr(handler, "_OnActionEventBase__actions_on_event", None)
            if on_exit is None:
                on_exit = getattr(handler, "_OnProcessExit__actions_on_exit", None)
            if on_exit is None:
                on_exit = getattr(handler, "_OnProcessExit__on_exit", None)
            if on_exit and not callable(on_exit):
                return [a for a in on_exit if isinstance(a, Node)]
    return []


def test_generates_expected_entities():
    from launch.actions import DeclareLaunchArgument

    ld = _load_ld()
    args = [e for e in ld.entities if isinstance(e, DeclareLaunchArgument)]
    # The post-cleanup branch is the canonical set of bench nodes.
    nodes = _on_exit_nodes(ld.entities)
    assert {a.name for a in args} == {
        "bind", "port", "with_waving", "perf_logging", "kill_stale"}
    defaults = {a.name: "".join(s.text for s in a.default_value) for a in args}
    assert defaults == {"bind": "0.0.0.0", "port": "8766", "with_waving": "true",
                        "perf_logging": "false", "kill_stale": "true"}
    assert len(nodes) == 3
    # exactly one node (the waving server) is conditional (with_waving)
    assert sum(1 for n in nodes if n.condition is not None) == 1


def test_declares_kill_stale_cleanup_guard():
    """A cleanup ExecuteProcess gated on kill_stale runs the three narrow
    pkills before the nodes; it must be scoped to lib/<pkg>/ exec paths."""
    from launch.actions import ExecuteProcess

    from launch_ros.actions import Node

    ld = _load_ld()
    # In Humble, Node inherits ExecuteProcess; exclude Node to isolate the
    # plain ExecuteProcess cleanup action.
    cleanups = [e for e in ld.entities
                if isinstance(e, ExecuteProcess) and not isinstance(e, Node)]
    assert len(cleanups) == 1, "expected exactly one cleanup ExecuteProcess"
    cleanup = cleanups[0]
    # gated behind the kill_stale launch arg
    assert cleanup.condition is not None
    # narrow, lib/<pkg>/-scoped patterns — never a bare exec name
    cmd_text = " ".join(
        "".join(s.text for s in part) if not isinstance(part, str) else part
        for part in cleanup.cmd)
    for pat in ("lib/vision_track/person_track_server",
                "lib/vision_track/track_web",
                "lib/tk_vision_specialized/waving_person_server"):
        assert pat in cmd_text, f"missing narrow pattern: {pat}"
    # SIGTERM, never SIGKILL
    assert "-9" not in cmd_text and "pkill" in cmd_text


def test_generate_is_side_effect_free():
    """generate_launch_description() must not spawn pkill at build time —
    the structural test calls it; the kill runs only when launch executes."""
    import subprocess
    import unittest.mock as mock

    with mock.patch.object(subprocess, "Popen", side_effect=AssertionError(
            "generate_launch_description must not spawn a subprocess")), \
            mock.patch.object(subprocess, "run", side_effect=AssertionError(
                "generate_launch_description must not run a subprocess")), \
            mock.patch.object(subprocess, "call", side_effect=AssertionError(
                "generate_launch_description must not call a subprocess")):
        ld = _load_ld()
    assert ld is not None
