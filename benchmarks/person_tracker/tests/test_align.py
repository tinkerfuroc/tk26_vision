"""Tests for ptbench.common.align — prediction stream alignment to GT."""
import pytest

from ptbench.common.align import PredFrame, align_pred_to_gt
from ptbench.common.schema import GtFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gt(t_ns: int, present: bool = True) -> GtFrame:
    bbox = (10, 10, 100, 100) if present else None
    return GtFrame(t_ns=t_ns, present=present, bbox=bbox, centroid_3d=None)


def pred(t_ns: int, lost: bool = False, tid: int = 1, xyz=None) -> PredFrame:
    return PredFrame(t_ns=t_ns, target_lost=lost, target_track_id=tid, point_xyz=xyz)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlignPredToGt:
    def test_exact_match(self):
        gts = [gt(1000), gt(2000)]
        preds = [pred(1000), pred(2000)]
        result = align_pred_to_gt(preds, gts, tol_ms=50.0)
        assert len(result) == 2
        g0, p0 = result[0]
        assert g0.t_ns == 1000
        assert p0 is not None and p0.t_ns == 1000

    def test_within_tolerance_matched(self):
        """Pred 30ms away from GT should be matched at tol=50ms."""
        gts = [gt(1_000_000_000)]  # 1.0 s in ns
        preds = [pred(1_030_000_000)]  # 30 ms away
        result = align_pred_to_gt(preds, gts, tol_ms=50.0)
        g, p = result[0]
        assert p is not None
        assert p.t_ns == 1_030_000_000

    def test_outside_tolerance_not_matched(self):
        """Pred 60ms away from GT should not be matched at tol=50ms."""
        gts = [gt(1_000_000_000)]
        preds = [pred(1_060_000_000)]  # 60 ms away
        result = align_pred_to_gt(preds, gts, tol_ms=50.0)
        g, p = result[0]
        assert p is None

    def test_exact_tol_boundary_included(self):
        """Pred exactly at tol_ms boundary is included (<=)."""
        gts = [gt(1_000_000_000)]
        preds = [pred(1_050_000_000)]  # exactly 50 ms away
        result = align_pred_to_gt(preds, gts, tol_ms=50.0)
        g, p = result[0]
        assert p is not None

    def test_nearest_of_two_picked(self):
        """When two preds are within tol, the nearest one is picked."""
        gts = [gt(1_000_000_000)]
        preds = [
            pred(1_020_000_000, tid=1),  # 20 ms away
            pred(1_040_000_000, tid=2),  # 40 ms away
        ]
        result = align_pred_to_gt(preds, gts, tol_ms=50.0)
        g, p = result[0]
        assert p is not None
        assert p.target_track_id == 1  # nearest

    def test_empty_preds_all_none(self):
        """No predictions → all GT frames get None."""
        gts = [gt(1000), gt(2000), gt(3000)]
        result = align_pred_to_gt([], gts, tol_ms=50.0)
        assert len(result) == 3
        for g, p in result:
            assert p is None

    def test_empty_gt_empty_result(self):
        """No GT frames → empty result list."""
        preds = [pred(1000), pred(2000)]
        result = align_pred_to_gt(preds, [], tol_ms=50.0)
        assert result == []

    def test_both_empty(self):
        result = align_pred_to_gt([], [], tol_ms=50.0)
        assert result == []

    def test_multiple_gt_frames_each_gets_nearest(self):
        """Each GT frame independently finds its nearest pred."""
        gts = [gt(1_000_000_000), gt(2_000_000_000)]
        preds = [pred(1_010_000_000, tid=10), pred(2_020_000_000, tid=20)]
        result = align_pred_to_gt(preds, gts, tol_ms=50.0)
        assert result[0][1].target_track_id == 10
        assert result[1][1].target_track_id == 20

    def test_out_of_order_preds_tolerated(self):
        """Preds in non-chronological order should still match correctly."""
        gts = [gt(1_000_000_000), gt(2_000_000_000)]
        preds = [pred(2_010_000_000, tid=20), pred(1_010_000_000, tid=10)]
        result = align_pred_to_gt(preds, gts, tol_ms=50.0)
        assert result[0][1].target_track_id == 10
        assert result[1][1].target_track_id == 20

    def test_pred_reuse_allowed(self):
        """Same pred can be the nearest for multiple GT frames (no exclusive assignment)."""
        # Two GT frames 10ms apart, one pred in the middle 5ms from each
        gts = [gt(1_000_000_000), gt(1_010_000_000)]
        preds = [pred(1_005_000_000, tid=99)]
        result = align_pred_to_gt(preds, gts, tol_ms=50.0)
        # Both within 50ms → both get the pred (5ms from each)
        assert result[0][1] is not None
        assert result[1][1] is not None

    def test_absent_gt_frames_also_aligned(self):
        """GT frames with present=False also get matched preds."""
        gts = [gt(1000, present=False)]
        preds = [pred(1010)]
        result = align_pred_to_gt(preds, gts, tol_ms=50.0)
        g, p = result[0]
        assert g.present is False
        assert p is not None

    def test_preserves_gt_order(self):
        """Output list preserves the GT frame order."""
        gts = [gt(t) for t in [100, 200, 300, 400, 500]]
        preds = [pred(t) for t in [100, 200, 300, 400, 500]]
        result = align_pred_to_gt(preds, gts)
        for i, (g, p) in enumerate(result):
            assert g.t_ns == gts[i].t_ns
