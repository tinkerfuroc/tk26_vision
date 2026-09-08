"""Unit tests for ptbench.tpt_bench.metrics (pure, synthetic boxes)."""

from __future__ import annotations

import math

import pytest

from ptbench.tpt_bench.metrics import compute_tpt_metrics, iou


# ---------------------------------------------------------------- iou ----------


def test_iou_identical_boxes():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # A=(0,0,10,10) area 100; B=(5,5,15,15) area 100; inter=(5,5,10,10)=25
    # union = 100+100-25 = 175 ; iou = 25/175
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25.0 / 175.0)


def test_iou_none_returns_zero():
    assert iou(None, (0, 0, 1, 1)) == 0.0
    assert iou((0, 0, 1, 1), None) == 0.0
    assert iou(None, None) == 0.0


def test_iou_half_overlap():
    # A=(0,0,10,10); B=(0,0,10,5): inter=50, union=100 => 0.5
    assert iou((0, 0, 10, 10), (0, 0, 10, 5)) == pytest.approx(0.5)


# --------------------------------------------------- compute_tpt_metrics ------


def test_perfect_tracking():
    gt = [(0, 0, 10, 10), (5, 5, 15, 15), (1, 1, 11, 11)]
    pred = list(gt)  # exact match every frame
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f_score"] == pytest.approx(1.0)
    assert m["ao"] == pytest.approx(1.0)
    assert m["amr"] == pytest.approx(1.0)


def test_all_miss_zero_overlap():
    gt = [(0, 0, 10, 10), (0, 0, 10, 10)]
    pred = [(100, 100, 110, 110), (100, 100, 110, 110)]  # nowhere near gt
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f_score"] == 0.0
    assert m["ao"] == 0.0
    assert m["amr"] == 0.0


def test_partial_overlap_below_threshold_is_incorrect():
    # iou = 25/175 ~= 0.143 < 0.5 => not correct
    gt = [(0, 0, 10, 10)]
    pred = [(5, 5, 15, 15)]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    # but AO still counts the raw overlap
    assert m["ao"] == pytest.approx(25.0 / 175.0)


def test_partial_overlap_above_threshold_is_correct():
    # iou = 0.5 exactly, thr=0.5 => correct (>=)
    gt = [(0, 0, 10, 10)]
    pred = [(0, 0, 10, 5)]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["ao"] == pytest.approx(0.5)


def test_false_positive_on_absent_frame_lowers_precision():
    # frame0: gt present, correct. frame1: gt absent but pred present => FP.
    gt = [(0, 0, 10, 10), None]
    pred = [(0, 0, 10, 10), (50, 50, 60, 60)]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    # 2 predictions, 1 correct => precision 0.5
    assert m["precision"] == pytest.approx(0.5)
    # 1 gt present, 1 correct => recall 1.0
    assert m["recall"] == pytest.approx(1.0)
    f = 2 * 0.5 * 1.0 / (0.5 + 1.0)
    assert m["f_score"] == pytest.approx(f)


def test_recall_with_missed_present_frame():
    # frame0 correct, frame1 gt present but no pred (None) => missed
    gt = [(0, 0, 10, 10), (0, 0, 10, 10)]
    pred = [(0, 0, 10, 10), None]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    # 1 prediction, 1 correct => precision 1.0
    assert m["precision"] == pytest.approx(1.0)
    # 2 gt present, 1 correct => recall 0.5
    assert m["recall"] == pytest.approx(0.5)
    # AO: frame0 iou 1.0, frame1 pred None => 0.0 ; mean over 2 present = 0.5
    assert m["ao"] == pytest.approx(0.5)


def test_ao_averaging_over_present_frames_only():
    # frame0 iou 1.0 (present), frame1 absent (excluded), frame2 iou 0.5 present
    gt = [(0, 0, 10, 10), None, (0, 0, 10, 10)]
    pred = [(0, 0, 10, 10), (50, 50, 60, 60), (0, 0, 10, 5)]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    # AO averages over the 2 present frames: (1.0 + 0.5) / 2 = 0.75
    assert m["ao"] == pytest.approx(0.75)


def test_amr_drops_low_confidence_wrong_pred_to_restore_precision():
    # frame0: correct pred, high confidence
    # frame1: gt absent, wrong pred (FP), LOW confidence
    # Global precision = 0.5, but raising the score threshold drops the FP and
    # restores precision to 1.0 with recall 1.0 => AMR = 1.0.
    gt = [(0, 0, 10, 10), None]
    pred = [(0, 0, 10, 10), (50, 50, 60, 60)]
    scores = [0.9, 0.2]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5, scores=scores)
    assert m["precision"] == pytest.approx(0.5)  # before thresholding
    assert m["amr"] == pytest.approx(1.0)         # after dropping low-conf FP


def test_amr_zero_when_never_correct():
    gt = [(0, 0, 10, 10)]
    pred = [(50, 50, 60, 60)]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5, scores=[0.9])
    assert m["amr"] == 0.0


def test_amr_default_scores_collapse_to_recall_when_precise():
    # No scores => all equal; if global precision already 1.0, AMR == recall.
    gt = [(0, 0, 10, 10), (0, 0, 10, 10)]
    pred = [(0, 0, 10, 10), None]  # 1 pred, 1 correct => precision 1.0
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["amr"] == pytest.approx(0.5)


def test_amr_partial_threshold_intermediate_recall():
    # Two correct preds (high conf) + one FP (mid conf). Sweeping threshold:
    #   keep-all: precision 2/3 (not 1.0)
    #   drop FP : precision 1.0, recall 2/2 = 1.0
    gt = [(0, 0, 10, 10), (0, 0, 10, 10), None]
    pred = [(0, 0, 10, 10), (0, 0, 10, 10), (50, 50, 60, 60)]
    scores = [0.9, 0.8, 0.5]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5, scores=scores)
    assert m["amr"] == pytest.approx(1.0)


def test_empty_input_returns_zeros():
    m = compute_tpt_metrics([], [], iou_thr=0.5)
    assert m == {
        "precision": 0.0,
        "recall": 0.0,
        "f_score": 0.0,
        "ao": 0.0,
        "amr": 0.0,
    }


def test_all_absent_gt_returns_zeros_no_raise():
    gt = [None, None]
    pred = [None, None]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0
    assert m["ao"] == 0.0
    assert m["amr"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_tpt_metrics([(0, 0, 1, 1)], [], iou_thr=0.5)


def test_scores_length_mismatch_raises():
    with pytest.raises(ValueError, match="align"):
        compute_tpt_metrics(
            [(0, 0, 1, 1)], [(0, 0, 1, 1)], iou_thr=0.5, scores=[1.0, 2.0]
        )


def test_all_outputs_are_plain_floats():
    gt = [(0, 0, 10, 10)]
    pred = [(0, 0, 10, 10)]
    m = compute_tpt_metrics(gt, pred, iou_thr=0.5)
    for k, v in m.items():
        assert type(v) is float, f"{k} is {type(v)}"
        assert not math.isnan(v)
