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
"""fp16 ReID forward tolerance tests (Phase 3, Task 3).

Torch+CUDA-gated: fp16 half-precision forward only runs on CUDA, so these tests
skip cleanly when torch/CUDA/the OSNet backbone are unavailable. They assert the
returned embedding stays float32 + L2-normalized (the embedding interface is
unchanged) and that fp16 and fp32 embeddings of the same crop stay within
cosine > 0.999 so identity gating / thresholds are unaffected.
"""
import numpy as np
import pytest


def _crop(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _model(fp16):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("fp16 path requires CUDA")
    from vision_track.reid.reid import PersonReIDModel
    try:
        m = PersonReIDModel(device="cuda", fp16=fp16)
    except Exception as e:
        pytest.skip(f"ReID model unavailable: {e}")
    if not getattr(m, "use_deep_features", False) or m.backbone is None:
        pytest.skip("ReID backbone did not load")
    return m


def test_fp16_output_is_l2_normalized_float32():
    m = _model(fp16=True)
    v = m.extract_features(_crop(200, 90, 1))
    assert v.dtype == np.float32
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3


def test_fp16_close_to_fp32():
    m16 = _model(fp16=True)
    m32 = _model(fp16=False)
    crop = _crop(220, 100, 7)
    v16 = m16.extract_features(crop)
    v32 = m32.extract_features(crop)
    # cosine similarity between fp16 and fp32 embeddings of the same crop
    cos = float(np.dot(v16, v32) / (np.linalg.norm(v16) * np.linalg.norm(v32) + 1e-9))
    assert cos > 0.999


def test_fp16_batch_output_is_l2_normalized_float32():
    m = _model(fp16=True)
    crops = [_crop(180, 80, k) for k in range(4)]
    out = m.extract_features_batch(crops)
    assert out.dtype == np.float32
    assert out.shape == (4, m.feature_dim)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


def test_fp16_batch_close_to_fp32():
    m16 = _model(fp16=True)
    m32 = _model(fp16=False)
    crops = [_crop(190, 85, k + 11) for k in range(4)]
    o16 = m16.extract_features_batch(crops)
    o32 = m32.extract_features_batch(crops)
    for i in range(len(crops)):
        cos = float(
            np.dot(o16[i], o32[i])
            / (np.linalg.norm(o16[i]) * np.linalg.norm(o32[i]) + 1e-9)
        )
        assert cos > 0.999, f"row {i} cosine {cos:.5f} <= 0.999"
