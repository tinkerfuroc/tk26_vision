"""Unit tests for _pose_backend that do not need a real model file.

The mediapipe Tasks landmarker is replaced with a fake so these run anywhere.
Real-model behaviour is covered by test_pose_parity.py.
"""
import numpy as np
import pytest

from tk_vision_specialized import _pose_backend as pb
from tk_vision_specialized._pose_backend import (
    Landmark, PoseBackend, PoseLandmarkIdx, POSE_CONNECTIONS, draw_pose,
)


class _FakeNormLm:
    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _FakeResult:
    def __init__(self, poses):
        self.pose_landmarks = poses


class _FakeLandmarker:
    created = []

    def __init__(self, delegate, poses):
        self.delegate = delegate
        self.poses = poses
        self.closed = False
        _FakeLandmarker.created.append(self)

    def detect(self, _image):
        return _FakeResult(self.poses)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_mp(monkeypatch):
    """Patch PoseBackend's factory so no model file / libmediapipe is needed."""
    calls = {"gpu_raises": False, "poses": [[_FakeNormLm(0.1, 0.2, 0.0, 0.9)] * 33]}

    def _create(model_path, delegate, min_conf):
        if delegate == "gpu" and calls["gpu_raises"]:
            raise RuntimeError("no EGL")
        return _FakeLandmarker(delegate, calls["poses"])

    monkeypatch.setattr(pb, "_create_landmarker", _create)
    monkeypatch.setattr(pb, "_to_mp_image", lambda rgb: rgb)
    _FakeLandmarker.created.clear()
    return calls


def test_enum_matches_blazepose_indices():
    assert PoseLandmarkIdx.NOSE == 0
    assert PoseLandmarkIdx.LEFT_SHOULDER == 11 and PoseLandmarkIdx.RIGHT_SHOULDER == 12
    assert PoseLandmarkIdx.LEFT_ELBOW == 13 and PoseLandmarkIdx.RIGHT_ELBOW == 14
    assert PoseLandmarkIdx.LEFT_WRIST == 15 and PoseLandmarkIdx.RIGHT_WRIST == 16
    assert len(POSE_CONNECTIONS) == 35
    assert all(0 <= a < 33 and 0 <= b < 33 for a, b in POSE_CONNECTIONS)


def test_gpu_first_success(fake_mp):
    be = PoseBackend("dummy.task", delegate="gpu")
    assert be.active_delegate == "gpu"
    assert be.fallback_reason is None


def test_gpu_failure_falls_back_to_cpu(fake_mp):
    fake_mp["gpu_raises"] = True
    be = PoseBackend("dummy.task", delegate="gpu")
    assert be.active_delegate == "cpu"
    assert "no EGL" in be.fallback_reason
    # the failed GPU attempt must not leak an open landmarker
    assert all(l.closed for l in _FakeLandmarker.created if l.delegate == "gpu")


def test_cpu_requested_never_tries_gpu(fake_mp):
    fake_mp["gpu_raises"] = True  # would raise if attempted
    be = PoseBackend("dummy.task", delegate="cpu")
    assert be.active_delegate == "cpu" and be.fallback_reason is None
    assert [l.delegate for l in _FakeLandmarker.created] == ["cpu"]


def test_invalid_delegate_rejected(fake_mp):
    with pytest.raises(ValueError):
        PoseBackend("dummy.task", delegate="tpu")


def test_process_returns_landmark_list_indexable_by_enum(fake_mp):
    be = PoseBackend("dummy.task", delegate="cpu")
    lms = be.process(np.zeros((64, 64, 3), np.uint8))
    assert len(lms) == 33
    lm = lms[PoseLandmarkIdx.RIGHT_WRIST]
    assert isinstance(lm, Landmark)
    assert (lm.x, lm.y, lm.visibility) == (0.1, 0.2, 0.9)


def test_process_returns_none_when_no_pose(fake_mp):
    fake_mp["poses"] = []
    be = PoseBackend("dummy.task", delegate="cpu")
    assert be.process(np.zeros((64, 64, 3), np.uint8)) is None


def test_process_rejects_non_rgb_uint8(fake_mp):
    be = PoseBackend("dummy.task", delegate="cpu")
    with pytest.raises(ValueError):
        be.process(np.zeros((64, 64), np.uint8))


def test_close_is_idempotent(fake_mp):
    be = PoseBackend("dummy.task", delegate="cpu")
    be.close(); be.close()
    assert _FakeLandmarker.created[-1].closed


def test_draw_pose_modifies_image_and_tolerates_none():
    img = np.zeros((100, 80, 3), np.uint8)
    lms = [Landmark(0.5, 0.5, 0.0, 1.0)] * 33
    draw_pose(img, lms)
    assert img.any()
    untouched = np.zeros((10, 10, 3), np.uint8)
    draw_pose(untouched, None)
    assert not untouched.any()
