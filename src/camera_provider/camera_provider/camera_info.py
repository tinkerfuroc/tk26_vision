"""Small helpers for validating camera-provider intrinsics."""
from __future__ import annotations

import math
from typing import Any, Optional


def camera_info_is_valid(info: Any) -> bool:
    """Return whether a CameraInfo contains usable pinhole intrinsics."""
    if info is None:
        return False
    try:
        if int(info.width) <= 0 or int(info.height) <= 0:
            return False
        values = getattr(info, 'k', None)
        if values is None or len(values) != 9:
            return False
        numbers = tuple(float(value) for value in values)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return (
        all(math.isfinite(value) for value in numbers)
        and numbers[0] > 0.0
        and numbers[4] > 0.0
    )


def select_camera_info(
    depth_info: Any,
    color_info: Any,
    *,
    prefer_depth: bool = True,
) -> Optional[Any]:
    """Select the first valid depth/color CameraInfo without truth-testing K."""
    candidates = (
        (depth_info, color_info)
        if prefer_depth
        else (color_info, depth_info)
    )
    for info in candidates:
        if camera_info_is_valid(info):
            return info
    return None
