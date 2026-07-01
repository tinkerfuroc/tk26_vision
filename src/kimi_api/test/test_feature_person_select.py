"""Unit tests for feature_recognition.select_best_person_idx.

2026-07-01 incident: feature_extraction_service picked a 17x23 px person
detection through a distant doorway over the actual foreground person,
because the foreground person's mask centroid was pulled off-center by
being cut off at the bottom of frame (very close to the camera), while the
tiny background person happened to sit almost exactly at the pixel center.
Pure image-center-distance selection has no notion of apparent size, so it
picked the background blob. These tests reproduce that scenario and lock in
the size-gate + depth-weighted fix.
"""
from kimi_api.feature_recognition import select_best_person_idx


FRAME_W, FRAME_H = 1280, 720


def test_tiny_centered_background_person_loses_to_large_offcenter_foreground():
    # Foreground: large box, cut off at the bottom of frame (mask centroid
    # pulled up/off-center), no reliable depth reading.
    foreground = (380, 280, 900, 720)
    # Background: tiny box almost exactly at image center.
    background = (520, 416, 537, 439)
    bboxes = [background, foreground]
    depths = [None, None]
    assert select_best_person_idx(bboxes, depths, FRAME_W, FRAME_H) == 1


def test_closer_person_preferred_when_depth_known():
    left = (100, 200, 400, 620)   # 3 m away
    right = (900, 200, 1200, 620)  # 1 m away, roughly symmetric offset
    bboxes = [left, right]
    depths = [3.0, 1.0]
    assert select_best_person_idx(bboxes, depths, FRAME_W, FRAME_H) == 1


def test_falls_back_to_pure_offset_when_depth_unknown():
    centered = (590, 200, 690, 620)
    off_to_side = (1100, 200, 1200, 620)
    bboxes = [centered, off_to_side]
    depths = [None, None]
    assert select_best_person_idx(bboxes, depths, FRAME_W, FRAME_H) == 0


def test_all_candidates_too_small_returns_negative_one():
    bboxes = [(520, 416, 537, 439), (600, 300, 615, 320)]
    depths = [None, None]
    assert select_best_person_idx(bboxes, depths, FRAME_W, FRAME_H) == -1


def test_single_large_candidate_is_selected():
    bboxes = [(400, 100, 900, 700)]
    depths = [None]
    assert select_best_person_idx(bboxes, depths, FRAME_W, FRAME_H) == 0


def test_zero_depth_sentinel_does_not_beat_real_close_subject():
    # Real 2026-07-01 replay against object_detection_generalist logs: a
    # depth-hole detection reports centroid.z == 0.0 (a "valid but zero"
    # point from the upstream pipeline, not None) rather than a genuine
    # 0 m reading. An additive offset+depth score treated that sentinel as
    # "closer than anything real" and picked the small, off-center, no-data
    # detection over the large, clearly-centered, real-depth (0.75 m)
    # foreground subject. Locks in that a real depth reading never loses to
    # a bogus zero.
    foreground = (315, 64, 947, 707)   # dominant, centered, depth = 0.75 m
    background = (372, 298, 444, 543)  # smaller, off-center, depth = 0.0 (invalid)
    bboxes = [foreground, background]
    depths = [0.75, 0.0]
    assert select_best_person_idx(bboxes, depths, FRAME_W, FRAME_H) == 0
