"""Normalize color Image messages to BGR ndarrays.

The Orbbec publishes ``rgb8`` (verified live 2026-06-07); other drivers publish
``bgr8``. Every cv2-side consumer in this package works in BGR, so decode +
normalize HERE, once. Duck-typed (needs only encoding/width/height/step/data)
so it unit-tests without ROS.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def decode_color_msg(msg) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Color Image msg -> (BGR HxWx3 uint8 array, None) or (None, reason)."""
    if msg.step != msg.width * 3:
        return None, f"unexpected step {msg.step} for width {msg.width}"
    try:
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
    except Exception as exc:
        return None, f"decode failed: {exc}"
    if msg.encoding == "bgr8":
        return buf, None
    if msg.encoding == "rgb8":
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR), None
    return None, f"unsupported color encoding {msg.encoding!r}"
