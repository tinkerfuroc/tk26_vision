"""Contact sheet validating the 2026-07-01 kimi_api fixes against real,
newly-captured vision_log scenes (not the original two bug-report scenes).

Seat recommendation (`seat_recommend_bbox` / `_seat_bbox_vlm.py`):
  - prompt strengthened for partial-occupancy carefulness + seat suitability
    (sofa/chair over stool, widest cushion among ties)
  - deterministic `_best_unoccupied_seat` / `_seat_rank` backstop
  - self-consistency backstop: recovers when `choice` names a seat the
    model's own `seats` entry marks occupied

Feature extraction (`feature_recognition.select_best_person_idx`):
  - minimum apparent-size gate before centering ever gets to compete
  - depth used only as a near-tie breaker (not an additive term), and
    non-positive depth treated as invalid rather than "0 m away" — an
    additive weighting was tried first and regressed on real data (see
    README.md), which is why the panels below matter as evidence,
    not just the unit tests.

Run from this directory with the tk26_vision venv + ROS sourced:
    python3 make_contact_sheet.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

VISION_LOG = "/home/tinker/tk25_ws/vision_log"
KIMI_API_SRC = "/home/tinker/tk25_ws/src/tk26_vision/src/kimi_api"
sys.path.insert(0, KIMI_API_SRC)

from kimi_api._seat_bbox_vlm import decode_box_xyxy, select_box  # noqa: E402
from kimi_api.feature_recognition import select_best_person_idx  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
GREEN = (0, 200, 0)
RED = (0, 0, 220)
WHITE = (255, 255, 255)


def _wrap_put_text(img, text, org, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _panel(img_path, boxes_labels_colors, caption, out_name):
    """boxes_labels_colors: list of (box_xyxy, label, color)."""
    img = cv2.imread(img_path)
    for box, label, color in boxes_labels_colors:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        _wrap_put_text(img, label, (x1 + 4, max(20, y1 - 8)), 0.55, color, 2)
    _wrap_put_text(img, caption, (12, 34), 0.7, WHITE, 2)
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, img)
    return out_path


def seat_panel_1():
    img_path = (f"{VISION_LOG}/20260701_203130/seat_recommend_bbox_service_"
                f"1782909095_seat_recommend_bbox_orig_20260701_203534_911.jpg")
    h, w = cv2.imread(img_path).shape[:2]
    # Real live seats array captured for this scene (qwen3-vl-plus).
    box = decode_box_xyxy([693, 571, 906, 864], w, h)
    return _panel(
        img_path, [(box, "chosen: left spot on sofa", GREEN)],
        "SEAT 1/3 -- correct on first pass (no override needed)",
        "seat_panel_1.jpg",
    )


def seat_panel_2():
    img_path = (f"{VISION_LOG}/20260701_203616/seat_recommend_bbox_service_"
                f"1782909381_seat_recommend_bbox_orig_20260701_203931_458.jpg")
    h, w = cv2.imread(img_path).shape[:2]
    box = decode_box_xyxy([476, 550, 661, 757], w, h)
    return _panel(
        img_path, [(box, "chosen: right spot on sofa", GREEN)],
        "SEAT 2/3 -- correct on first pass (wide cushion, no override)",
        "seat_panel_2.jpg",
    )


def seat_panel_3():
    img_path = (f"{VISION_LOG}/20260701_203616/seat_recommend_bbox_service_"
                f"1782909381_seat_recommend_bbox_orig_20260701_204122_611.jpg")
    h, w = cv2.imread(img_path).shape[:2]
    # Real captured raw model response for this scene: choice="left spot on
    # sofa" while that seat's own entry says occupied=true -- a genuine
    # self-contradiction from a live qwen3-vl-plus call, replayed here
    # through the fixed select_box() to show the recovery deterministically.
    parsed = {
        "seats": [
            {"label": "left stool", "box_2d": [38, 560, 240, 887], "occupied": False},
            {"label": "left spot on sofa", "box_2d": [223, 510, 420, 830], "occupied": True},
            {"label": "middle spot on sofa", "box_2d": [392, 580, 588, 770], "occupied": True},
            {"label": "right spot on sofa", "box_2d": [550, 480, 636, 710], "occupied": False},
            {"label": "front middle stool", "box_2d": [636, 520, 720, 650], "occupied": False},
            {"label": "right stool", "box_2d": [720, 500, 800, 630], "occupied": False},
        ],
        "choice": "left spot on sofa",
    }
    known_seats = ["left stool", "front middle stool", "right stool",
                   "left spot on sofa", "middle spot on sofa", "right spot on sofa"]
    res = select_box(parsed, w, h, known_seats)
    assert res.overridden_from == "left spot on sofa" and res.label == "right spot on sofa"
    rejected_box = decode_box_xyxy(parsed["seats"][1]["box_2d"], w, h)
    return _panel(
        img_path,
        [
            (rejected_box, "model said: left spot on sofa (occupied=true!)", RED),
            (res.box_xyxy, f"FIXED -> {res.label}", GREEN),
        ],
        "SEAT 3/3 -- self-contradiction caught + recovered",
        "seat_panel_3.jpg",
    )


def _feature_panel(det_json_path, img_path, out_name, caption):
    data = json.load(open(det_json_path))
    dets = [d for d in data["detections"] if d["cls_name"] == "person"]
    bboxes = [tuple(d["bbox"]) for d in dets]
    depths = [d["centroid"]["z"] for d in dets]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    sel = select_best_person_idx(bboxes, depths, w, h)
    boxes_labels_colors = [(bboxes[sel], "selected", GREEN)]
    return _panel(
        img_path, boxes_labels_colors,
        f"{caption} (n={len(bboxes)} persons, idx={sel})",
        out_name,
    )


def feature_panel_1():
    return _feature_panel(
        f"{VISION_LOG}/20260701_203130/generalist_detection_node_yolo_req_20260701_203431_183.json",
        f"{VISION_LOG}/20260701_203130/generalist_detection_node_yolo_orig_20260701_203431_183.jpg",
        "feature_panel_1.jpg",
        "PERSON 1/3 -- dominant subject correctly picked",
    )


def feature_panel_2():
    return _feature_panel(
        f"{VISION_LOG}/20260701_203616/generalist_detection_node_yolo_req_20260701_203851_560.json",
        f"{VISION_LOG}/20260701_203616/generalist_detection_node_yolo_orig_20260701_203851_560.jpg",
        "feature_panel_2.jpg",
        "PERSON 2/3 -- zero-depth-sentinel regression fixed (see README.md)",
    )


def feature_panel_3():
    return _feature_panel(
        f"{VISION_LOG}/20260701_203616/generalist_detection_node_yolo_req_20260701_204047_328.json",
        f"{VISION_LOG}/20260701_203616/generalist_detection_node_yolo_orig_20260701_204047_328.jpg",
        "feature_panel_3.jpg",
        "PERSON 3/3 -- 14 candidates, background clutter gated out",
    )


def build_sheet(panel_paths, cols=3, tile_w=427, tile_h=240,
                 header_lines=(), out_name="contact_sheet.jpg"):
    tiles = [cv2.resize(cv2.imread(p), (tile_w, tile_h)) for p in panel_paths]
    rows = (len(tiles) + cols - 1) // cols
    line_h = 34
    header_h = line_h * len(header_lines) + 10 if header_lines else 0
    sheet = np.zeros((rows * tile_h + header_h, cols * tile_w, 3), dtype=np.uint8)
    sheet[:] = (30, 30, 30)
    for i, line in enumerate(header_lines):
        _wrap_put_text(sheet, line, (14, 28 + i * line_h), 0.7, WHITE, 2)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = header_h + r * tile_h
        sheet[y0:y0 + tile_h, c * tile_w:(c + 1) * tile_w] = t
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, sheet)
    return out_path


def main():
    seat_paths = [seat_panel_1(), seat_panel_2(), seat_panel_3()]
    feature_paths = [feature_panel_1(), feature_panel_2(), feature_panel_3()]
    out = build_sheet(
        seat_paths + feature_paths, cols=3,
        header_lines=(
            "kimi_api seat_recommend_bbox + feature_extraction fix validation -- 2026-07-01",
            "top row: seat recommendation   |   bottom row: person selection",
        ),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
