"""Exercise generalist frame selection without loading inference models."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


def select_frame(node, camera="realsense"):
    source = Path(__file__).parents[1] / "object_detection_generalist/generalist_node.py"
    tree = ast.parse(source.read_text())
    cls = next(item for item in tree.body
               if isinstance(item, ast.ClassDef) and item.name == "GeneralistDetectionNode")
    method = next(item for item in cls.body
                  if isinstance(item, ast.FunctionDef) and item.name == "_wait_for_recent_frame")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"copy": copy}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["_wait_for_recent_frame"](node, camera)


@pytest.mark.parametrize("provider_timeout", [False, True])
def test_available_frames_are_valid_regardless_of_age(provider_timeout):
    bundle = SimpleNamespace(color_msg={"stamp": 1}, depth_msg={"stamp": 2})
    intake = SimpleNamespace(
        latest=lambda: None if provider_timeout else bundle,
        latest_cached=lambda: bundle,
    )
    result = select_frame(SimpleNamespace(_camera_intakes={"realsense": intake}))
    assert result == ({"stamp": 1}, {"stamp": 2})
    assert result[0] is not bundle.color_msg


def test_absent_data_cannot_be_fabricated():
    intake = SimpleNamespace(latest=lambda: None, latest_cached=lambda: None)
    assert select_frame(SimpleNamespace(_camera_intakes={"realsense": intake})) is None
    assert select_frame(SimpleNamespace(_camera_intakes={})) is None
