"""Key-gated offline test of the live-person waving prompt.

Runs the real VLM provider chain against the three workspace ground-truth
images and asserts the number of LIVE wavers found. Skipped when no provider
key (OPENROUTER_API_KEY / DASHSCOPE_API_KEY) is configured, so it never runs
in offline CI. The counts are the contract; the VLM is non-deterministic, so a
single spurious miss can be re-run. Downscales the 4096-px source frames to a
camera-like width because encode_data_url does not resize.

Image paths resolve from $TK_WS_ROOT (default /home/tinker/tk25_ws); the test
skips per-image if the file is absent.
"""

from __future__ import annotations

import os

import cv2
import pytest

from tk_vision_specialized._waving_vlm import (
    build_provider_models,
    has_provider_key,
    request_waving_persons_chain,
)

WS_ROOT = os.environ.get('TK_WS_ROOT', '/home/tinker/tk25_ws')

# (filename, expected live-waver count). See the design doc's ground-truth
# table: printed mural figures must be excluded; real overlapping wavers kept.
CASES = [
    ('waving_background', 0),
    ('waving_real_and_background', 2),
    ('waving_two_hands', 2),
]


def _model_for(provider):
    return 'qwen3-vl-plus' if provider == 'qwen' else 'google/gemini-2.5-pro'


def _chain():
    return build_provider_models('qwen', 'gemini', has_key=has_provider_key,
                                 model_for=_model_for)


def _load(name, max_w=1280):
    path = os.path.join(WS_ROOT, name)
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f'ground-truth image not found: {path}')
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (max_w, int(round(h * scale))))
    return img


pytestmark = pytest.mark.skipif(
    not _chain(),
    reason='no VLM provider key (OPENROUTER_API_KEY / DASHSCOPE_API_KEY)')


@pytest.mark.parametrize('name,expected', CASES)
def test_live_person_waving_counts(name, expected):
    img = _load(name)
    res = request_waving_persons_chain(
        img, provider_models=_chain(), timeout_s=30.0, max_retries=3)
    assert len(res.boxes) == expected, (
        f'{name}: expected {expected} live waver(s), got {len(res.boxes)} '
        f'(provider={res.provider}, boxes={res.boxes})')
