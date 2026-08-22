"""Regression gate: the Tasks-API backend must reproduce the mediapipe 0.10.9
Solutions verdicts recorded in fixtures/pose_parity (see its README).

Needs the real model file in the weights cache (scripts/download_models.py);
skips — loudly — if it is absent so weight-less CI doesn't fail, but T0 on the
robot must have it so the skip never masks a regression.
"""
import json
import logging
from pathlib import Path

import cv2
import pytest

from vision_util.weights_cache import find_cached
from tk_vision_specialized._pose_backend import PoseBackend, PoseLandmarkIdx
from tk_vision_specialized.waving_person_server import DetectWavingPersonsNode

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pose_parity"
EXPECTED = json.loads((FIXTURE_DIR / "expected_0.10.9.json").read_text())
JOINTS = [PoseLandmarkIdx.NOSE, PoseLandmarkIdx.LEFT_SHOULDER, PoseLandmarkIdx.RIGHT_SHOULDER,
          PoseLandmarkIdx.LEFT_ELBOW, PoseLandmarkIdx.RIGHT_ELBOW,
          PoseLandmarkIdx.LEFT_WRIST, PoseLandmarkIdx.RIGHT_WRIST]

MODEL = find_cached("pose_landmarker_full.task")
pytestmark = pytest.mark.skipif(
    MODEL is None,
    reason="pose_landmarker_full.task not in weights cache — run scripts/download_models.py")


class _Stub:
    MIN_VISIBILITY = DetectWavingPersonsNode.MIN_VISIBILITY
    ELBOW_TOL_NORM = DetectWavingPersonsNode.ELBOW_TOL_NORM

    def get_logger(self):
        return logging.getLogger("parity")


def _verdict(landmarks, roi):
    return DetectWavingPersonsNode.is_waving(_Stub(), landmarks, roi)


def _crops():
    for entry in EXPECTED["crops"]:
        bgr = cv2.imread(str(FIXTURE_DIR / entry["file"]))
        assert bgr is not None, entry["file"]
        yield entry, bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def cpu_backend():
    be = PoseBackend(str(MODEL), delegate="cpu")
    yield be
    be.close()


@pytest.fixture(scope="module")
def gpu_backend():
    be = PoseBackend(str(MODEL), delegate="gpu")
    if be.active_delegate != "gpu":
        be.close()
        pytest.skip(f"GPU delegate unavailable: {be.fallback_reason}")
    yield be
    be.close()


def _check(backend, y_tol, vis_tol):
    mismatches = []
    for entry, bgr, rgb in _crops():
        lms = backend.process(rgb)
        detected = lms is not None
        if detected != entry["detected"]:
            mismatches.append(f"{entry['file']}: detected {detected} != {entry['detected']}")
            continue
        verdict = _verdict(lms, bgr)
        if verdict != entry["is_waving"]:
            mismatches.append(f"{entry['file']}: is_waving {verdict} != {entry['is_waving']}")
        if detected:
            for j in JOINTS:
                ex, ey, ez, ev = entry["landmarks"][int(j)]
                dy, dv = abs(lms[j].y - ey), abs(lms[j].visibility - ev)
                if dy > y_tol or dv > vis_tol:
                    mismatches.append(f"{entry['file']} {j.name}: dy={dy:.4f} dvis={dv:.4f}")
    assert not mismatches, "\n".join(mismatches)


def test_fixture_is_legacy():
    assert EXPECTED["mediapipe_version"] == "0.10.9"
    assert len(EXPECTED["crops"]) >= 6
    assert any(c["detected"] for c in EXPECTED["crops"])


def test_cpu_parity(cpu_backend):
    _check(cpu_backend, y_tol=0.01, vis_tol=0.05)


def test_gpu_parity(gpu_backend):
    # fp16 GPU path drifts more in coordinates; verdicts must still be identical
    _check(gpu_backend, y_tol=0.05, vis_tol=0.15)


def test_gpu_fallback_keeps_parity(monkeypatch):
    from tk_vision_specialized import _pose_backend as pb
    real = pb._create_landmarker

    def _gpu_breaks(model_path, delegate, min_conf):
        if delegate == "gpu":
            raise RuntimeError("forced GPU failure")
        return real(model_path, delegate, min_conf)

    monkeypatch.setattr(pb, "_create_landmarker", _gpu_breaks)
    be = PoseBackend(str(MODEL), delegate="gpu")
    try:
        assert be.active_delegate == "cpu" and "forced" in be.fallback_reason
        _check(be, y_tol=0.01, vis_tol=0.05)
    finally:
        be.close()
