"""Unit tests for _waving_vlm.py — pure decoders, key resolution, and the
provider chain. No network: the OpenAI client is monkeypatched per-test."""

from __future__ import annotations

from tk_vision_specialized._waving_vlm import (
    WavingVlmResult,
    decode_box_xyxy,
    select_boxes,
)


def test_decode_box_xyxy_scales_and_clamps():
    # 0-1000 normalized -> pixels on a 1000x500 frame; x by width, y by height.
    assert decode_box_xyxy([100, 200, 300, 400], 1000, 500) == (100, 100, 300, 200)
    # out-of-range clamps to image bounds (w-1, h-1).
    assert decode_box_xyxy([0, 0, 1000, 1000], 1000, 500) == (0, 0, 999, 499)


def test_decode_box_xyxy_swaps_inverted_corners():
    assert decode_box_xyxy([300, 400, 100, 200], 1000, 500) == (100, 100, 300, 200)


def test_decode_box_xyxy_rejects_degenerate_and_malformed():
    assert decode_box_xyxy([500, 500, 500, 500], 1000, 500) is None  # zero area
    assert decode_box_xyxy([10], 1000, 500) is None                  # wrong length
    assert decode_box_xyxy('nope', 1000, 500) is None                # wrong type


def test_select_boxes_keeps_only_waving_with_decodable_box():
    parsed = {'persons': [
        {'box_2d': [100, 200, 300, 400], 'waving': True},
        {'box_2d': [0, 0, 100, 100], 'waving': False},     # dropped: not waving
        {'box_2d': [500, 500, 500, 500], 'waving': True},  # dropped: zero area
    ]}
    res = select_boxes(parsed, 1000, 500)
    assert isinstance(res, WavingVlmResult)
    assert res.boxes == [(100, 100, 300, 200)]
    assert res.error is None


def test_select_boxes_clean_empty_when_no_wavers():
    res = select_boxes({'persons': []}, 1000, 500)
    assert res.boxes == []
    assert res.error is None  # clean empty is terminal, not an error
