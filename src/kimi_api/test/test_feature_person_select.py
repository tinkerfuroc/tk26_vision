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


def test_large_centered_background_crowd_loses_to_dominant_offcenter_guest():
    # 2026-07-02 incident (vision_log/20260702_070449, extraction call
    # 09:17:58): at the arena door the guest fills the frame but is cut off
    # at the bottom, so her mask bbox centers well off the optical center
    # (offset ~0.27 of the half-diagonal). A crowd member ~3 m behind the
    # barrier is 156 px tall — 21.7% of frame height, past the 15% size
    # gate — and sits almost exactly on the optical center (offset ~0.03).
    # The 2026-07-01 gate+centermost fix therefore still picked him: the
    # offset difference (0.24) is far outside DEPTH_TIE_EPS, so depth was
    # never consulted. Crowd bbox is the real logged "Selected" bbox; the
    # guest bbox is read off the paired generalist-detection overlay.
    crowd = (628, 290, 689, 446)   # h = 156 px (21.7%), depth ~3.0 m
    guest = (617, 250, 968, 720)   # h = 470 px (65%), depth ~0.9 m
    bboxes = [crowd, guest]
    depths = [2.97, 0.9]
    assert select_best_person_idx(bboxes, depths, FRAME_W, FRAME_H) == 1


def test_comparably_sized_candidates_still_ranked_by_centering():
    # Size dominance must not override centering among near-equals: two
    # people at similar scale (height ratio 0.85 — inside HEIGHT_TIE_FRAC)
    # keep the centermost-wins ranking even though one is slightly shorter.
    centered_shorter = (590, 220, 690, 560)   # h = 340
    taller_offside = (1050, 180, 1200, 580)   # h = 400
    bboxes = [centered_shorter, taller_offside]
    depths = [None, None]
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
