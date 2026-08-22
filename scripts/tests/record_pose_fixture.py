#!/usr/bin/env python3
"""Record the legacy mediapipe 0.10.9 Solutions pose output on person crops.

One-off, run BEFORE upgrading mediapipe. Produces the fixture that
``test_pose_parity.py`` replays against the Tasks-API adapter.

    python scripts/tests/record_pose_fixture.py \
        --images <img.jpg ...> --out src/tk_vision_specialized/test/fixtures/pose_parity

Crops are produced by the node's YOLO11m-seg (conf 0.4, CPU) so they match
what waving_person_server feeds the pose model.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import mediapipe as mp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "vision_util"))
sys.path.insert(0, str(REPO / "src" / "tk_vision_specialized"))

from vision_util.weights_cache import resolve_weights  # noqa: E402
from tk_vision_specialized.waving_person_server import DetectWavingPersonsNode  # noqa: E402


class _Stub:
    """Minimal ``self`` so the node's is_waving runs without rclpy."""
    MIN_VISIBILITY = DetectWavingPersonsNode.MIN_VISIBILITY
    ELBOW_TOL_NORM = DetectWavingPersonsNode.ELBOW_TOL_NORM

    def get_logger(self):
        return logging.getLogger("record_pose_fixture")


def crop_persons(yolo, img, conf=0.4):
    res = yolo(img, conf=conf, verbose=False, device="cpu")[0]
    for box in res.boxes:
        if yolo.names[int(box.cls[0])] != "person":
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        roi = img[y1:y2, x1:x2]
        if roi.size:
            yield roi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11m-seg.pt")
    args = ap.parse_args()

    assert mp.__version__ == "0.10.9", f"fixture must be recorded on 0.10.9, got {mp.__version__}"
    from ultralytics import YOLO
    yolo = YOLO(str(resolve_weights(args.model)))
    opts = dict(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
    pose = mp.solutions.pose.Pose(**opts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stub = _Stub()
    crops = []
    n = 0
    for img_path in args.images:
        img = cv2.imread(img_path)
        assert img is not None, img_path
        for roi in crop_persons(yolo, img):
            name = f"{n:02d}.png"
            cv2.imwrite(str(out / name), roi)
            res = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            lm = res.pose_landmarks
            entry = {"file": name, "source": Path(img_path).name,
                     "detected": lm is not None, "landmarks": None,
                     "is_waving": DetectWavingPersonsNode.is_waving(stub, lm, roi)}
            if lm is not None:
                entry["landmarks"] = [[round(p.x, 6), round(p.y, 6), round(p.z, 6),
                                       round(p.visibility, 6)] for p in lm.landmark]
            crops.append(entry)
            print(f"{name}: detected={entry['detected']} waving={entry['is_waving']} ({img_path})")
            n += 1

    (out / "expected_0.10.9.json").write_text(json.dumps(
        {"mediapipe_version": mp.__version__, "solutions_options": opts, "crops": crops},
        indent=1))
    print(f"wrote {n} crops to {out}")


if __name__ == "__main__":
    main()
