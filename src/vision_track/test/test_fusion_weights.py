"""Pure-logic tests for the recalibrated ReID fusion weights + thresholds.

ROS-free: imports only ReIDMatcher (numpy-only class constants + the dynamic
weight-normalization invariant). No model construction, no rclpy.
"""
import numpy as np

from vision_track.reid.reid import ReIDMatcher


def test_person_weights_sum_to_one_after_normalization():
    w = np.array([
        ReIDMatcher.WEIGHT_REID,
        ReIDMatcher.WEIGHT_BODY_COLOR,
        ReIDMatcher.WEIGHT_COLOR,
        ReIDMatcher.WEIGHT_UPPER,
        ReIDMatcher.WEIGHT_LOWER,
    ])
    assert abs(float(np.sum(w / np.sum(w))) - 1.0) < 1e-9


def test_deep_term_dominates_after_reweight():
    # Phase 1: with a trained backbone the deep term must dominate color.
    assert ReIDMatcher.WEIGHT_REID >= 0.70
    color_total = (
        ReIDMatcher.WEIGHT_BODY_COLOR
        + ReIDMatcher.WEIGHT_COLOR
        + ReIDMatcher.WEIGHT_UPPER
        + ReIDMatcher.WEIGHT_LOWER
    )
    assert ReIDMatcher.WEIGHT_REID > color_total


def test_raw_reid_floor_raised_for_trained_backbone():
    # Trained OSNet separates same/different far better than the random head,
    # so the raw-cosine floor can be raised from the legacy 0.60.
    assert ReIDMatcher.MIN_REID_SIMILARITY_RAW >= 0.30
    assert ReIDMatcher.MIN_REID_SIMILARITY_RAW <= 0.55


def test_color_hard_floors_relaxed():
    # Color is now a backup cue, not a gate — its hard floors must not
    # hard-reject a true match on lighting/clothing variation.
    assert ReIDMatcher.MIN_BODY_COLOR_SIMILARITY <= 0.45
    assert ReIDMatcher.MIN_UPPER_SIMILARITY <= 0.45
    assert ReIDMatcher.MIN_LOWER_SIMILARITY <= 0.45
