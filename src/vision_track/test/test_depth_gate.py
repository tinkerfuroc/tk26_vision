"""Tests for the pure depth-consistency (crosser-rejection) predicate."""
import numpy as np

from vision_track.core.depth_gate import (
    is_depth_consistent,
    roi_median_depth,
    should_reject_candidate,
)


class TestIsDepthConsistent:
    def test_same_depth_consistent(self):
        assert is_depth_consistent(3.0, 3.0, jump_threshold=0.6) is True

    def test_small_toward_camera_jump_consistent(self):
        # operator at 3.0 m, candidate at 2.5 m → 0.5 m nearer < 0.6 threshold
        assert is_depth_consistent(2.5, 3.0, jump_threshold=0.6) is True

    def test_large_toward_camera_jump_rejected(self):
        # candidate 1.0 m vs operator 3.0 m → 2.0 m nearer, a crosser
        assert is_depth_consistent(1.0, 3.0, jump_threshold=0.6) is False

    def test_farther_candidate_always_consistent(self):
        # moving AWAY from the camera is never a crosser cue
        assert is_depth_consistent(5.0, 3.0, jump_threshold=0.6) is True

    def test_no_operator_depth_passes(self):
        # unknown operator depth → cannot gate → permissive
        assert is_depth_consistent(1.0, None, jump_threshold=0.6) is True

    def test_invalid_candidate_depth_passes(self):
        assert is_depth_consistent(0.0, 3.0, jump_threshold=0.6) is True
        assert is_depth_consistent(float("nan"), 3.0, jump_threshold=0.6) is True

    def test_threshold_boundary_inclusive(self):
        # exactly at the threshold is still consistent (reject only beyond)
        assert is_depth_consistent(2.4, 3.0, jump_threshold=0.6) is True
        assert is_depth_consistent(2.39, 3.0, jump_threshold=0.6) is False


class TestRoiMedianDepth:
    def _depth(self, H, W, val_m):
        return np.full((H, W), int(val_m * 1000), dtype=np.uint16)

    def test_constant_roi(self):
        d = self._depth(100, 100, 2.5)
        m = roi_median_depth(d, (10, 10, 60, 60), min_depth=0.1, max_depth=10.0)
        assert abs(m - 2.5) < 1e-3

    def test_excludes_zero_and_out_of_range(self):
        d = self._depth(100, 100, 2.5)
        d[10:20, 10:20] = 0          # invalid
        d[20:30, 10:20] = 11000      # out of range
        m = roi_median_depth(d, (10, 10, 60, 60), min_depth=0.1, max_depth=10.0)
        assert abs(m - 2.5) < 1e-3   # median over valid only

    def test_all_invalid_returns_none(self):
        d = np.zeros((100, 100), dtype=np.uint16)
        m = roi_median_depth(d, (10, 10, 60, 60), min_depth=0.1, max_depth=10.0)
        assert m is None


class TestShouldRejectCandidate:
    def test_rejects_toward_camera_crosser(self):
        # operator 3.0 m, candidate 1.0 m, threshold 0.6 → reject
        assert should_reject_candidate(
            candidate_depth=1.0, operator_depth=3.0, jump_threshold=0.6
        ) is True

    def test_keeps_consistent_candidate(self):
        assert should_reject_candidate(
            candidate_depth=2.8, operator_depth=3.0, jump_threshold=0.6
        ) is False

    def test_no_operator_depth_keeps(self):
        assert should_reject_candidate(
            candidate_depth=1.0, operator_depth=None, jump_threshold=0.6
        ) is False
