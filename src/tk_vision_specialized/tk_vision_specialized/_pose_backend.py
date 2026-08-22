"""Pose-estimation backend for waving detection (MediaPipe Tasks API).

Wraps ``mediapipe.tasks.python.vision.PoseLandmarker`` (mediapipe >= 1.0)
behind the small surface ``waving_person_server`` needs, returning landmarks
shaped like the legacy ``mp.solutions.pose`` output so the ``is_waving``
heuristic is untouched.

Import-light on purpose: mediapipe, numpy, cv2 — no rclpy.

GPU delegate notes (Ubuntu only): creation takes 3–6 s once and mediapipe
prints an ``Unable to initialize EGL`` probe error even when it succeeds.
We verify with a warm-up ``detect`` and fall back to CPU on any failure.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

POSE_MODEL_FILENAME = "pose_landmarker_full.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


class PoseLandmarkIdx(enum.IntEnum):
    """BlazePose landmark indices used by ``is_waving`` (33-point topology)."""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16


# BlazePose skeleton (same edge list as the legacy mp.solutions.pose.POSE_CONNECTIONS).
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


_VALID_DELEGATES = ("gpu", "cpu")


def _create_landmarker(model_path: str, delegate: str, min_conf: float):
    """Build a Tasks PoseLandmarker (IMAGE mode, one pose). Separated for tests."""
    from mediapipe.tasks.python import BaseOptions, vision
    deleg = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=deleg),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_conf,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def _to_mp_image(rgb: np.ndarray):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class PoseBackend:
    """Single-person pose landmarks for one RGB crop at a time.

    ``delegate='gpu'`` tries the GPU delegate and silently rebuilds on CPU if
    creation or a warm-up inference fails; ``active_delegate`` /
    ``fallback_reason`` tell the caller what happened so it can log once.
    """

    def __init__(self, model_path: str, delegate: str = "gpu",
                 min_detection_confidence: float = 0.5):
        if delegate not in _VALID_DELEGATES:
            raise ValueError(f"pose delegate must be one of {_VALID_DELEGATES}, got {delegate!r}")
        self.model_path = model_path
        self.active_delegate: str = delegate
        self.fallback_reason: Optional[str] = None
        self._lm = None
        if delegate == "gpu":
            try:
                lm = _create_landmarker(model_path, "gpu", min_detection_confidence)
                try:
                    lm.detect(_to_mp_image(np.zeros((256, 256, 3), np.uint8)))  # warm-up / probe
                except Exception:
                    lm.close()
                    raise
                self._lm = lm
            except Exception as exc:  # noqa: BLE001 — any failure means "use CPU"
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                self.active_delegate = "cpu"
        if self._lm is None:
            self._lm = _create_landmarker(model_path, "cpu", min_detection_confidence)

    def process(self, rgb: np.ndarray) -> Optional[list]:
        """Return 33 normalized ``Landmark`` for the first pose, or ``None``."""
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"process expects HxWx3 uint8 RGB, got {rgb.shape} {rgb.dtype}")
        result = self._lm.detect(_to_mp_image(rgb))
        if not result.pose_landmarks:
            return None
        return [Landmark(p.x, p.y, p.z, float(p.visibility or 0.0))
                for p in result.pose_landmarks[0]]

    def close(self) -> None:
        if self._lm is not None:
            self._lm.close()
            self._lm = None

    def __del__(self):  # best effort; explicit close() preferred
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def draw_pose(bgr: np.ndarray, landmarks, connections=POSE_CONNECTIONS) -> None:
    """Draw joints + skeleton onto ``bgr`` in place (no-op for ``None``)."""
    if landmarks is None or bgr.size == 0:
        return
    h, w = bgr.shape[:2]
    pts = [(int(round(lm.x * w)), int(round(lm.y * h))) for lm in landmarks]
    for a, b in connections:
        cv2.line(bgr, pts[a], pts[b], (255, 255, 255), 2)
    for (x, y), lm in zip(pts, landmarks):
        color = (0, 255, 0) if lm.visibility >= 0.5 else (0, 0, 255)
        cv2.circle(bgr, (x, y), 3, color, -1)
