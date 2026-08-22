#!/usr/bin/env python3
"""Offline RGB-only regression harness for the waving-person gesture pipeline.

Runs YOLO-seg + MediaPipe Pose + the `is_waving` heuristic against the
labelled images under `detect_waving_test/{waving,not_waving}/` and reports
accuracy, FP/FN lists, and per-image overlays so we can eyeball
misclassifications.

Run BEFORE and AFTER fixing `waving_person_server.py`:

    # baseline (buggy elbow-shoulder clause + no visibility filter)
    python3 debug_waving_pipeline.py --legacy-is-waving --out-dir /tmp/wave_audit_before

    # post-fix
    python3 debug_waving_pipeline.py --out-dir /tmp/wave_audit_after

Compare `results.csv` and stdout summary between the two runs.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]


# Lazy-import heavy deps so --help works without them.
def _lazy_imports():
    global YOLO, PoseBackend, PoseLandmarkIdx, draw_pose, find_cached
    from ultralytics import YOLO  # noqa: PLC0415
    sys.path.insert(0, str(REPO / "src" / "tk_vision_specialized"))
    sys.path.insert(0, str(REPO / "src" / "vision_util"))
    from tk_vision_specialized._pose_backend import (  # noqa: PLC0415
        PoseBackend, PoseLandmarkIdx, draw_pose,
    )
    from vision_util.weights_cache import find_cached  # noqa: PLC0415
    return YOLO, PoseBackend


# ----- gesture predicates -----------------------------------------------------

def is_waving_legacy(landmarks, img_h):
    """The pre-fix predicate verbatim from waving_person_server.py.
    Reproduces both bugs: (B1) normalized vs pixel mix, (B2) no visibility."""
    PL = PoseLandmarkIdx
    rh, re, rs = landmarks[PL.RIGHT_WRIST], landmarks[PL.RIGHT_ELBOW], landmarks[PL.RIGHT_SHOULDER]
    lh, le, ls = landmarks[PL.LEFT_WRIST],  landmarks[PL.LEFT_ELBOW],  landmarks[PL.LEFT_SHOULDER]
    rh_above_sh = rh.y <= rs.y
    lh_above_sh = lh.y <= ls.y
    rh_above_el = rh.y < re.y
    lh_above_el = lh.y < le.y
    re_above_sh = re.y <= (rs.y + int(img_h * 0.1))   # bug: pixel int added to normalized
    le_above_sh = le.y <= (ls.y + int(img_h * 0.1))   # bug
    return (rh_above_sh or lh_above_sh
            or (rh_above_el and re_above_sh)
            or (lh_above_el and le_above_sh))


def is_waving_fixed(landmarks, img_h):
    """Audit-fixed predicate — must match `is_waving` in waving_person_server.py.
    Per-side visibility gate + normalized tolerance. Tuned 2026-05-04 against
    expanded GT set (41 images)."""
    PL = PoseLandmarkIdx
    rh, re, rs = landmarks[PL.RIGHT_WRIST], landmarks[PL.RIGHT_ELBOW], landmarks[PL.RIGHT_SHOULDER]
    lh, le, ls = landmarks[PL.LEFT_WRIST],  landmarks[PL.LEFT_ELBOW],  landmarks[PL.LEFT_SHOULDER]
    MIN_VIS, SHOULDER_TOL, ELBOW_TOL = 0.5, 0.1, 0.1
    right_visible = min(rh.visibility, re.visibility, rs.visibility) >= MIN_VIS
    left_visible = min(lh.visibility, le.visibility, ls.visibility) >= MIN_VIS
    if not (right_visible or left_visible):
        return False
    right_wave = right_visible and (
        rh.y <= rs.y + SHOULDER_TOL
        or (rh.y < re.y and re.y <= rs.y + ELBOW_TOL)
    )
    left_wave = left_visible and (
        lh.y <= ls.y + SHOULDER_TOL
        or (lh.y < le.y and le.y <= ls.y + ELBOW_TOL)
    )
    return right_wave or left_wave


# ----- per-image evaluation ---------------------------------------------------

def evaluate_image(img_path, label, yolo, pose, predicate, conf_thresh, out_overlay_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return {"image": img_path.name, "label": label, "prediction": None,
                "n_persons": 0, "n_visible_persons": 0, "max_visibility": 0.0,
                "latency_ms": 0.0, "error": "imread_failed"}

    t0 = time.perf_counter()
    results = yolo(img, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes

    overlay = img.copy()
    n_persons = 0
    n_visible = 0
    max_vis = 0.0
    image_pred = False
    person_verdicts = []  # (bbox, verdict, mean_visibility)

    if boxes is not None:
        for i, box in enumerate(boxes):
            cls_name = yolo.names[int(box.cls[0])]
            if cls_name != "person":
                continue
            n_persons += 1
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            lms = pose.process(rgb_roi)

            verdict = False
            mean_v = 0.0
            if lms is not None:
                PL = PoseLandmarkIdx
                ks = (PL.RIGHT_WRIST, PL.RIGHT_ELBOW, PL.RIGHT_SHOULDER,
                      PL.LEFT_WRIST,  PL.LEFT_ELBOW,  PL.LEFT_SHOULDER)
                vis = [lms[k].visibility for k in ks]
                mean_v = float(np.mean(vis))
                max_vis = max(max_vis, mean_v)
                if mean_v >= 0.5:
                    n_visible += 1
                verdict = predicate(lms, roi.shape[0])
                # Draw landmarks on overlay (use roi coordinates remapped)
                draw_landmarks_on_overlay(overlay, roi, lms, x1, y1)

            person_verdicts.append(((x1, y1, x2, y2), verdict, mean_v))
            if verdict:
                image_pred = True

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Render overlay annotations
    for (x1, y1, x2, y2), verdict, mean_v in person_verdicts:
        color = (0, 0, 255) if verdict else (0, 255, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        label_str = f"{'WAVE' if verdict else 'still'} v={mean_v:.2f}"
        cv2.putText(overlay, label_str, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    correct = (image_pred == label)
    banner_color = (0, 200, 0) if correct else (0, 0, 200)
    cv2.putText(overlay,
                f"GT={'WAVE' if label else 'still'} | PRED={'WAVE' if image_pred else 'still'} "
                f"({'OK' if correct else 'MISS'})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)
    out_overlay_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_overlay_dir / img_path.name), overlay)

    return {
        "image": img_path.name,
        "label": int(label),
        "prediction": int(image_pred),
        "n_persons": n_persons,
        "n_visible_persons": n_visible,
        "max_visibility": round(max_vis, 3),
        "latency_ms": round(elapsed_ms, 1),
        "error": "",
    }


def draw_landmarks_on_overlay(overlay, roi, pose_landmarks, x_off, y_off):
    """Draw MediaPipe landmarks back onto the full overlay image."""
    h, w = roi.shape[:2]
    # draw_pose draws into a copy of the roi, then we paste it
    roi_copy = roi.copy()
    draw_pose(roi_copy, pose_landmarks)
    overlay[y_off:y_off + h, x_off:x_off + w] = roi_copy


# ----- top-level --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/home/tinker/tk25_ws/detect_waving_test")
    p.add_argument("--model", default="/home/tinker/tk25_ws/yolo11m-seg.pt")
    p.add_argument("--min-person-conf", type=float, default=0.4)
    p.add_argument("--out-dir", default="/tmp/wave_audit")
    p.add_argument("--legacy-is-waving", action="store_true",
                   help="use the pre-fix is_waving (bugs B1+B2) for baseline measurement")
    args = p.parse_args()

    YOLO, PoseBackend = _lazy_imports()
    yolo = YOLO(args.model)
    pose_model = find_cached("pose_landmarker_full.task")
    if pose_model is None:
        raise SystemExit(
            "pose_landmarker_full.task missing from the weights cache; "
            "run scripts/download_models.py first.")
    pose = PoseBackend(str(pose_model), delegate="gpu")
    predicate = is_waving_legacy if args.legacy_is_waving else is_waving_fixed

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_str, label in (("waving", True), ("not_waving", False)):
        sub = data_dir / label_str
        if not sub.is_dir():
            print(f"WARN: {sub} missing, skipping", file=sys.stderr)
            continue
        for img_path in sorted(sub.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            print(f"  [{label_str}] {img_path.name}", file=sys.stderr)
            row = evaluate_image(img_path, label, yolo, pose,
                                 predicate, args.min_person_conf, overlay_dir)
            rows.append(row)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate
    tp = sum(1 for r in rows if r["label"] == 1 and r["prediction"] == 1)
    tn = sum(1 for r in rows if r["label"] == 0 and r["prediction"] == 0)
    fp = sum(1 for r in rows if r["label"] == 0 and r["prediction"] == 1)
    fn = sum(1 for r in rows if r["label"] == 1 and r["prediction"] == 0)
    total = len(rows)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    fp_list = [r["image"] for r in rows if r["label"] == 0 and r["prediction"] == 1]
    fn_list = [r["image"] for r in rows if r["label"] == 1 and r["prediction"] == 0]

    print()
    print(f"Predicate:        {'LEGACY (buggy)' if args.legacy_is_waving else 'FIXED'}")
    print(f"Total images:     {total}")
    print(f"Accuracy:         {acc:.3f} ({tp + tn}/{total})")
    print(f"Precision (wave): {prec:.3f}")
    print(f"Recall    (wave): {rec:.3f}")
    print(f"Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"FP (predicted wave, GT=still):  {fp_list}")
    print(f"FN (predicted still, GT=wave):  {fn_list}")
    print(f"Per-row CSV:    {csv_path}")
    print(f"Overlays:       {overlay_dir}")


if __name__ == "__main__":
    main()
