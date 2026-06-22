"""TPT-Bench tracking metrics (pure python).

Implements the evaluation metrics defined in the TPT-Bench paper
(arXiv 2505.07446, Sec. 4.1). All functions operate on aligned per-frame
``(gt_bbox | None, pred_bbox | None)`` pairs, where each box is xyxy in pixels
and ``None`` means "absent" (no ground-truth target, or no prediction).

Definitions (paper notation in brackets):

* **Precision** [Tracking Precision, TP] — correct predictions ÷ frames that
  carry a prediction. A prediction is *correct* iff the ground-truth target is
  present in that frame AND ``IoU(pred, gt) >= iou_thr``. Predicting a box on
  an absent-GT frame is therefore a false positive that lowers precision.
* **Recall** [Tracking Recall, TR] — correct predictions ÷ frames where the
  ground-truth target is present.
* **F-score** — harmonic mean of precision and recall (0 if either is 0).
* **AO** (Average Overlap) — mean IoU over frames where the GT is present,
  taking IoU = 0 when no prediction was made. This is the paper's accuracy at
  overlap threshold ``tau_Omega = 0``.
* **AMR** (Average Max Recall at 100% precision) — the paper defines AMR as the
  recall achievable while precision stays at exactly 1.0, averaged over the IoU
  thresholds. Our tracker emits only a coarse per-frame confidence, so we
  approximate AMR at the single supplied ``iou_thr`` as: sweep the confidence
  threshold over the distinct prediction scores; among thresholds that retain
  precision == 1.0 (after suppressing predictions below the threshold), report
  the maximum recall. If no prediction is ever correct, AMR = 0.0. This matches
  the paper's "max recall while TP == 1" formulation restricted to one overlap
  threshold (we do not average across the threshold grid because the runner
  scores at a single IoU). See the module docstring of the runner for how
  scores are produced.

Empty input never raises — all metrics return 0.0.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

Box = Optional[Tuple[float, float, float, float]]


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two xyxy boxes.

    Returns 0.0 if either box is ``None`` or has non-positive area.
    """
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _amr(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    scores: Sequence[float],
    iou_thr: float,
) -> float:
    """Max recall achievable while precision stays at 1.0, sweeping score thr.

    For each candidate threshold ``t`` (each distinct prediction score, plus a
    "keep everything" baseline), suppress predictions whose score < t, then
    recompute precision/recall over the surviving predictions. Among thresholds
    yielding precision == 1.0, return the maximum recall. 0.0 if none qualify.
    """
    n_gt_present = sum(1 for g in gt_boxes if g is not None)
    if n_gt_present == 0:
        return 0.0

    # Candidate thresholds: each distinct score => "keep preds with score >= t".
    # Sorting ascending and including a sub-minimum baseline lets us start from
    # "keep all predictions" and progressively drop the lowest-confidence ones.
    pred_scores = [
        s for s, p in zip(scores, pred_boxes) if p is not None
    ]
    if not pred_scores:
        return 0.0
    thresholds = sorted(set(pred_scores))
    # Add a baseline strictly below the minimum so "keep all" is evaluated.
    thresholds = [thresholds[0] - 1.0] + thresholds

    best_recall = 0.0
    found_precise = False
    for thr in thresholds:
        n_pred = 0
        n_correct = 0
        for g, p, s in zip(gt_boxes, pred_boxes, scores):
            if p is None or s < thr:
                continue  # prediction suppressed
            n_pred += 1
            if g is not None and iou(g, p) >= iou_thr:
                n_correct += 1
        if n_pred == 0:
            continue
        precision = n_correct / n_pred
        if precision >= 1.0:
            recall = n_correct / n_gt_present
            best_recall = max(best_recall, recall)
            found_precise = True

    return float(best_recall) if found_precise else 0.0


def compute_tpt_metrics(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    iou_thr: float = 0.5,
    scores: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Compute TPT-Bench metrics over aligned per-frame boxes.

    Args:
        gt_boxes: per-frame ground-truth boxes (xyxy) or ``None`` if absent.
        pred_boxes: per-frame predicted boxes (xyxy) or ``None`` if no pred.
        iou_thr: IoU threshold for a prediction to count as correct.
        scores: optional per-frame prediction confidence (used only by AMR).
            Defaults to all-equal (1.0), which makes AMR collapse to recall iff
            global precision is already 1.0, else 0.0.

    Returns:
        Dict with ``precision``, ``recall``, ``f_score``, ``ao``, ``amr`` as
        plain floats. Never raises; returns all-zeros on empty input.
    """
    if len(gt_boxes) != len(pred_boxes):
        raise ValueError(
            f"gt_boxes ({len(gt_boxes)}) and pred_boxes ({len(pred_boxes)}) "
            "must be the same length"
        )
    n = len(gt_boxes)
    if scores is None:
        scores = [1.0] * n
    elif len(scores) != n:
        raise ValueError(
            f"scores ({len(scores)}) must align with boxes ({n})"
        )

    if n == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f_score": 0.0,
            "ao": 0.0,
            "amr": 0.0,
        }

    n_pred = 0          # frames carrying a prediction
    n_gt_present = 0     # frames where target exists
    n_correct = 0       # correct predictions (gt present & IoU >= thr)
    overlap_sum = 0.0    # sum of IoU over gt-present frames (pred absent => 0)

    for g, p in zip(gt_boxes, pred_boxes):
        if p is not None:
            n_pred += 1
        if g is not None:
            n_gt_present += 1
            ov = iou(g, p)  # iou handles p is None => 0.0
            overlap_sum += ov
            if p is not None and ov >= iou_thr:
                n_correct += 1

    precision = _safe_div(n_correct, n_pred)
    recall = _safe_div(n_correct, n_gt_present)
    f_score = (
        _safe_div(2.0 * precision * recall, precision + recall)
        if (precision > 0 and recall > 0)
        else 0.0
    )
    ao = _safe_div(overlap_sum, n_gt_present)
    amr = _amr(gt_boxes, pred_boxes, scores, iou_thr)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f_score": float(f_score),
        "ao": float(ao),
        "amr": float(amr),
    }
