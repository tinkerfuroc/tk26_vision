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


def test_generates_expected_entities():
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node

    ld = _load_ld()
    args = [e for e in ld.entities if isinstance(e, DeclareLaunchArgument)]
    nodes = [e for e in ld.entities if isinstance(e, Node)]
    assert {a.name for a in args} == {"bind", "port", "with_waving"}
    defaults = {a.name: "".join(s.text for s in a.default_value) for a in args}
    assert defaults == {"bind": "0.0.0.0", "port": "8766", "with_waving": "true"}
    assert len(nodes) == 3
    # exactly one node (the waving server) is conditional
    assert sum(1 for n in nodes if n.condition is not None) == 1
