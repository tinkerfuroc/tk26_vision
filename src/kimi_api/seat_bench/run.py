"""Run ONE (strategy, provider) cell over every dataset scene.

Writes results/<strategy>_<provider>/<id>.json (Result + scoring outcome)
and results/<strategy>_<provider>/<id>.jpg (overlay). One subagent runs
one cell; cells are independent so they fan out concurrently.

Usage:
  python -m seat_bench.run --strategy s1 --provider qwen [--ids scene_000 ...] [--limit N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2

from .geometry import draw_overlay
from .paths import DATASET_DIR, RESULTS_DIR
from .score import classify
from .strategies import get_strategy


def _load_scene(sid: str):
    img = cv2.imread(str(DATASET_DIR / f"{sid}.jpg"))
    req = json.load(open(DATASET_DIR / f"{sid}.req.json"))
    gt_path = DATASET_DIR / f"{sid}.gt.json"
    gt = json.load(open(gt_path)) if gt_path.is_file() else None
    return img, req, gt


def _scene_ids() -> list[str]:
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(str(DATASET_DIR / "scene_*.jpg")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--provider", required=True, choices=["gemini", "qwen"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_fn = get_strategy(args.strategy)
    cell = f"{args.strategy}_{args.provider}"
    out_dir = RESULTS_DIR / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.ids or _scene_ids()
    if args.limit:
        ids = ids[: args.limit]

    n_hit = 0
    for sid in ids:
        img, req, gt = _load_scene(sid)
        if img is None:
            print(f"  {sid}: SKIP (no image)")
            continue
        res = run_fn(img, req, args.provider, logger=lambda m: None)
        rec = res.to_dict()
        if gt is not None:
            outcome = classify(rec, gt)
            rec["scoring"] = outcome
            n_hit += 1 if outcome["outcome"] == "hit" else 0
            gt_boxes = [(tuple(s["cushion_bbox"]), s["occupied"]) for s in gt["seats"]]
            hit = outcome["outcome"] == "hit"
        else:
            gt_boxes, hit = None, None
        (out_dir / f"{sid}.json").write_text(json.dumps(rec, indent=2))
        overlay = draw_overlay(
            img,
            point=tuple(rec["point_xy"]) if rec["point_xy"] else None,
            box=tuple(rec["box_xyxy"]) if rec["box_xyxy"] else None,
            gt_boxes=gt_boxes,
            label=f"{cell}:{rec['chosen_label']}",
            hit=hit,
        )
        cv2.imwrite(str(out_dir / f"{sid}.jpg"), overlay)
        tag = rec.get("scoring", {}).get("outcome", "n/a")
        print(f"  {sid}: {tag} ({rec['n_calls']} calls, {rec['elapsed_s']:.1f}s)"
              + (f" ERR {rec['error']}" if rec.get("error") else ""))

    print(f"[{cell}] done: {len(ids)} scenes, {n_hit} hits -> {out_dir}")


if __name__ == "__main__":
    main()
