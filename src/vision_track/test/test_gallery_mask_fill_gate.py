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
"""Issue 3: gate ReID gallery admission on MASK-FILL, not bbox aspect ratio.

The operator stands throughout, so only clean, well-segmented operator views
should enrich the gallery. The OLD admission gate rejected on aspect ratio
(w/h <= 0.9), which wrongly drops a genuinely-upright operator whose bbox is
square-ish from occlusion / box-clipping (w/h ~= 1.0). The robust signal is
mask-fill = mask_pixels / bbox_area (the ``mask_coverage`` feature): a clean
view fills its box even when square (admit); a merged/garbage/occlusion-
inflated box has the operator mask as a thin slice (low fill -> reject).

These tests pin (1) the NEW gate defaults via direct ``crop_quality_ok``
calls and (2) that ``update_appearance`` wires the per-call gate from tracker
attrs (mask-fill 0.35, relaxed aspect 2.0) rather than ``DEFAULT_GATE``.
"""
import time
from types import SimpleNamespace

import numpy as np
import pytest

import vision_track.reid.appearance_manager as AM
from vision_track.reid.quality import crop_quality_ok

# The NEW gate the implementation must apply (mask-fill 0.35, relaxed aspect).
NEW_GATE = dict(
    min_crop_h=80,
    min_blur_var=50.0,
    min_mask_coverage=0.35,
    max_aspect_ratio=2.0,
)


def test_square_but_clean_admits():
    """REGRESSION GUARD: a square (w/h=1.0) but well-segmented crop now passes.

    Under the OLD ``max_aspect_ratio=0.9`` this was rejected — exactly the
    occluded-but-upright operator view the gallery needs most.
    """
    assert crop_quality_ok(
        crop_h=200, crop_w=200, mask_coverage=0.45, blur_var=100,
        aspect_ratio=1.0, **NEW_GATE,
    ) is True


def test_low_fill_rejected():
    """Thin operator slice in an inflated box -> low fill -> reject."""
    assert crop_quality_ok(
        crop_h=200, crop_w=200, mask_coverage=0.20, blur_var=100,
        aspect_ratio=1.0, **NEW_GATE,
    ) is False


def test_none_coverage_admits():
    """No seg mask this frame -> not rejected on fill."""
    assert crop_quality_ok(
        crop_h=200, crop_w=200, mask_coverage=None, blur_var=100,
        aspect_ratio=1.0, **NEW_GATE,
    ) is True


def test_mask_fill_boundary_is_strict():
    """Gate is strict ``<=``: exactly 0.35 rejects, 0.36 admits."""
    assert crop_quality_ok(
        crop_h=200, crop_w=200, mask_coverage=0.35, blur_var=100,
        aspect_ratio=1.0, **NEW_GATE,
    ) is False
    assert crop_quality_ok(
        crop_h=200, crop_w=200, mask_coverage=0.36, blur_var=100,
        aspect_ratio=1.0, **NEW_GATE,
    ) is True


def test_very_wide_degenerate_still_rejected():
    """The relaxed aspect backstop (2.0) still catches a 2.5 wide box."""
    assert crop_quality_ok(
        crop_h=200, crop_w=500, mask_coverage=0.45, blur_var=100,
        aspect_ratio=2.5, **NEW_GATE,
    ) is False


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
        gallery_max_aspect_ratio=2.0,
    )


def test_update_appearance_wires_gate_from_tracker(monkeypatch):
    """update_appearance must call crop_quality_ok with the tracker's mask-fill
    + relaxed-aspect gate, NOT the shared DEFAULT_GATE (0.4 / 0.9).
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
    assert captured["max_aspect_ratio"] == 2.0
    # The untouched keys still fall back to DEFAULT_GATE.
    assert captured["min_crop_h"] == 80
    assert captured["min_blur_var"] == 50.0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
