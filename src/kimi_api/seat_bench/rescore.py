"""Re-score existing results against (edited) ground truth — no VLM calls.

Reads every results/<cell>/<id>.json, re-runs score.classify against the
current dataset/<id>.gt.json, rewrites the json's 'scoring' field, and
regenerates the <id>.jpg overlay. Use after editing GT so the scoreboard
reflects corrected annotations without re-running the (expensive) VLM grid.

  python -m seat_bench.rescore
"""

from __future__ import annotations

import glob
import json
import os

import cv2

from .geometry import draw_overlay
from .paths import DATASET_DIR, RESULTS_DIR
from .score import classify


def _load_gt(sid: str):
    p = DATASET_DIR / f"{sid}.gt.json"
    return json.load(open(p)) if p.is_file() else None


def main():
    cells = sorted(os.path.basename(p) for p in glob.glob(str(RESULTS_DIR / "*"))
                   if os.path.isdir(p))
    for cell in cells:
        n = changed = 0
        for jf in sorted(glob.glob(str(RESULTS_DIR / cell / "*.json"))):
            sid = os.path.basename(jf)[:-5]
            rec = json.load(open(jf))
            gt = _load_gt(sid)
            if gt is None:
                continue
            n += 1
            old = (rec.get("scoring") or {}).get("outcome")
            outcome = classify(rec, gt)
            rec["scoring"] = outcome
            if outcome["outcome"] != old:
                changed += 1
            with open(jf, "w") as fh:
                json.dump(rec, fh, indent=2)
            img = cv2.imread(str(DATASET_DIR / f"{sid}.jpg"))
            if img is not None:
                gt_boxes = [(tuple(s["cushion_bbox"]), s["occupied"]) for s in gt["seats"]]
                hit = outcome["outcome"] == "hit"
                overlay = draw_overlay(
                    img,
                    point=tuple(rec["point_xy"]) if rec.get("point_xy") else None,
                    box=tuple(rec["box_xyxy"]) if rec.get("box_xyxy") else None,
                    gt_boxes=gt_boxes,
                    label=f"{cell}:{rec.get('chosen_label')}",
                    hit=hit,
                )
                cv2.imwrite(str(RESULTS_DIR / cell / f"{sid}.jpg"), overlay)
        print(f"[{cell}] rescored {n} scenes, {changed} outcome(s) changed")


if __name__ == "__main__":
    main()
