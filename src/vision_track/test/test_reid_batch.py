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
"""Batched ReID forward equivalence tests (Phase 3, Task 1).

The pure crop-stacking shape test runs without a model. The batch==sequential
equivalence and AppearanceExtractor tests are torch-gated and skip cleanly when
torch or the OSNet backbone is unavailable.
"""
import numpy as np
import pytest

from vision_track.reid.reid import PersonReIDModel


def _make_crop(h, w, seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


def test_stack_crops_shape_varying_sizes():
    crops = [_make_crop(200, 90, 1), _make_crop(50, 30, 2), _make_crop(400, 150, 3)]
    tensor = PersonReIDModel._stack_crops(crops)  # CPU torch.Tensor [K,3,256,128]
    assert tuple(tensor.shape) == (3, 3, 256, 128)
    assert tensor.dtype.is_floating_point


def test_stack_crops_empty():
    tensor = PersonReIDModel._stack_crops([])
    assert tuple(tensor.shape) == (0, 3, 256, 128)


def _model_or_skip():
    pytest.importorskip("torch")
    try:
        m = PersonReIDModel(device="cpu")
    except Exception as e:
        pytest.skip(f"ReID model unavailable: {e}")
    if not getattr(m, "use_deep_features", False) or m.backbone is None:
        pytest.skip("ReID backbone did not load")
    return m


def test_batch_equivalence_matches_sequential():
    m = _model_or_skip()
    crops = [_make_crop(180, 80, k) for k in range(5)]
    seq = np.stack([m.extract_features(c) for c in crops], axis=0)
    batched = m.extract_features_batch(crops)
    assert batched.shape == seq.shape
    # Eval-mode forward is deterministic; allow small fp accumulation differences.
    np.testing.assert_allclose(batched, seq, atol=1e-4, rtol=0)


def test_batch_empty_returns_zero_rows():
    m = _model_or_skip()
    out = m.extract_features_batch([])
    assert out.shape == (0, m.feature_dim)


def test_appearance_extractor_batch_matches_loop():
    pytest.importorskip("torch")
    from vision_track.reid.reid import AppearanceExtractor
    try:
        ae = AppearanceExtractor(device="cpu")
    except Exception as e:
        pytest.skip(f"AppearanceExtractor unavailable: {e}")
    if not getattr(ae.person_reid, "use_deep_features", False):
        pytest.skip("ReID backbone did not load")
    frame = _make_crop(480, 640, 99)
    bboxes = [(10, 10, 90, 210), (200, 20, 280, 220), (400, 30, 470, 230)]
    looped = [ae.extract_features(frame, b, None, class_id=0) for b in bboxes]
    batched = ae.extract_features_batch(frame, bboxes, [None] * 3, [0] * 3)
    assert len(batched) == len(looped)
    for got, exp in zip(batched, looped):
        assert set(got.keys()) == set(exp.keys())
        np.testing.assert_allclose(got["reid"], exp["reid"], atol=1e-4, rtol=0)
        np.testing.assert_allclose(got["body_color"], exp["body_color"], atol=1e-6, rtol=0)
