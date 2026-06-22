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
"""Issue 3: gate ReID gallery admission on BOTH bbox w/h ratio AND mask-fill.

The operator stands throughout, so only clean, UPRIGHT, well-segmented operator
views should enrich the gallery. Admission requires BOTH:
  - bbox width/height <= ``gallery_max_aspect_ratio`` (default 0.5) — upright
    (taller-than-wide) only; a square/wide box is rejected.
  - mask-fill (mask_pixels / bbox_area, the ``mask_coverage`` feature) >
    ``gallery_min_mask_fill`` (default 0.35) — a merged/garbage/occlusion-
    inflated box has the operator mask as a thin slice and is rejected.

``crop_quality_ok`` ANDs every check (False if any fails), so the two gates
compose automatically. These tests pin (1) the combined behaviour via direct
``crop_quality_ok`` calls and (2) that ``update_appearance`` wires the per-call
gate from the tracker attrs rather than the shared ``DEFAULT_GATE``.
"""
import time
from types import SimpleNamespace

import numpy as np
import pytest

import vision_track.reid.appearance_manager as AM
from vision_track.reid.quality import crop_quality_ok

# The NEW gate the implementation must apply: upright w/h <= 0.5 AND fill > 0.35.
NEW_GATE = dict(
    min_crop_h=80,
    min_blur_var=50.0,
    min_mask_coverage=0.35,
    max_aspect_ratio=0.5,
)

UPRIGHT = 0.4   # w/h of a clean standing person (admit)
SQUARE = 1.0    # w/h of an occlusion/clipping-collapsed box (reject on aspect)


def test_upright_clean_admits():
    """Both gates satisfied: tall box + good fill -> admit."""
    assert crop_quality_ok(
        crop_h=300, crop_w=120, mask_coverage=0.45, blur_var=100,
        aspect_ratio=UPRIGHT, **NEW_GATE,
    ) is True


def test_square_rejected_by_aspect():
    """A square (w/h=1.0) box is now REJECTED even with good fill (operator is
    upright; a square box is occluded/merged/non-standing)."""
    assert crop_quality_ok(
        crop_h=200, crop_w=200, mask_coverage=0.45, blur_var=100,
        aspect_ratio=SQUARE, **NEW_GATE,
    ) is False


def test_upright_but_low_fill_rejected():
    """Tall box but a thin operator slice -> low fill -> reject (mask-fill gate)."""
    assert crop_quality_ok(
        crop_h=300, crop_w=120, mask_coverage=0.20, blur_var=100,
        aspect_ratio=UPRIGHT, **NEW_GATE,
    ) is False


def test_none_coverage_upright_admits():
    """No seg mask this frame -> not rejected on fill, but still must be upright."""
    assert crop_quality_ok(
        crop_h=300, crop_w=120, mask_coverage=None, blur_var=100,
        aspect_ratio=UPRIGHT, **NEW_GATE,
    ) is True


def test_none_coverage_square_rejected():
    """No mask doesn't bypass the aspect gate: a square maskless box still rejects."""
    assert crop_quality_ok(
        crop_h=200, crop_w=200, mask_coverage=None, blur_var=100,
        aspect_ratio=SQUARE, **NEW_GATE,
    ) is False


def test_aspect_boundary_is_strict():
    """Aspect gate is ``>`` reject: exactly 0.5 admits, 0.51 rejects."""
    assert crop_quality_ok(
        crop_h=300, crop_w=150, mask_coverage=0.45, blur_var=100,
        aspect_ratio=0.5, **NEW_GATE,
    ) is True
    assert crop_quality_ok(
        crop_h=300, crop_w=153, mask_coverage=0.45, blur_var=100,
        aspect_ratio=0.51, **NEW_GATE,
    ) is False


def test_mask_fill_boundary_is_strict():
    """Mask-fill gate is strict ``<=``: exactly 0.35 rejects, 0.36 admits (upright)."""
    assert crop_quality_ok(
        crop_h=300, crop_w=120, mask_coverage=0.35, blur_var=100,
        aspect_ratio=UPRIGHT, **NEW_GATE,
    ) is False
    assert crop_quality_ok(
        crop_h=300, crop_w=120, mask_coverage=0.36, blur_var=100,
        aspect_ratio=UPRIGHT, **NEW_GATE,
    ) is True


def _make_capture_tracker():
    """A minimal SimpleNamespace tracker that drives update_appearance into the
    gate (and, on a rejecting capture, the motion-refresh branch).
    """
    extractor = SimpleNamespace(
        extract_features=lambda *a, **k: {
            "mask_coverage": np.array([0.45], dtype=np.float32),
            "reid": np.array([0.0], dtype=np.float32),
        },
    )
    appearance = SimpleNamespace(
        position_history=[],
        last_seen_time=time.time(),
        velocity=(0.0, 0.0),
    )
    return SimpleNamespace(
        appearance_extractor=extractor,
        target_appearance=appearance,
        original_track_id=None,        # skip person_registry.update_person
        person_registry=SimpleNamespace(update_person=lambda *a, **k: None),
        keep_gallery_thumbs=False,
        feature_refresh_interval=1.5,
        gallery_min_mask_fill=0.35,
        gallery_max_aspect_ratio=0.5,
    )


def test_update_appearance_wires_gate_from_tracker(monkeypatch):
    """update_appearance must call crop_quality_ok with the tracker's mask-fill
    + aspect gate, NOT the shared DEFAULT_GATE (0.4 / 0.9).
    """
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return False  # take the cheap motion-refresh branch

    monkeypatch.setattr(AM, "crop_quality_ok", capture)

    tracker = _make_capture_tracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = SimpleNamespace(
        bbox=(100, 50, 300, 450), mask=None, class_id=0, class_name="person",
    )

    AM.update_appearance(tracker, frame, result, similarity=0.5)

    assert captured["min_mask_coverage"] == 0.35
    assert captured["max_aspect_ratio"] == 0.5
    # The untouched keys still fall back to DEFAULT_GATE.
    assert captured["min_crop_h"] == 80
    assert captured["min_blur_var"] == 50.0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
