"""CLI: score the vision_track tracker on a TPT-Bench sequence.

Usage::

    python -m ptbench.tpt_bench.score_cli --seq DIR [--iou 0.5] \
        [--imgsz 1280] [--conf 0.5] [--json out.json]

Loads a sequence, runs the YOLO tracker over it, aligns predictions to
ground-truth per frame index, computes the TPT-Bench metrics, prints a small
ASCII table, and optionally dumps the metrics (+ run config) as JSON.

Self-contained: does not import ``ptbench.common``. Requires the ROS workspace
sourced so ``vision_track`` is importable (see ``DOWNLOAD.md`` / ``runner.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

from .dataset import TptDatasetError, load_sequence
from .metrics import compute_tpt_metrics

_METRIC_ORDER = ("precision", "recall", "f_score", "ao", "amr")


def _format_table(metrics: Dict[str, float]) -> str:
    """Render the metrics dict as a fixed-width two-column ASCII table."""
    rows = [(name, f"{metrics.get(name, 0.0):.4f}") for name in _METRIC_ORDER]
    label_w = max(len("metric"), *(len(name) for name, _ in rows))
    value_w = max(len("value"), *(len(val) for _, val in rows))
    sep = "+" + "-" * (label_w + 2) + "+" + "-" * (value_w + 2) + "+"
    lines = [
        sep,
        f"| {'metric'.ljust(label_w)} | {'value'.ljust(value_w)} |",
        sep,
    ]
    for name, val in rows:
        lines.append(f"| {name.ljust(label_w)} | {val.rjust(value_w)} |")
    lines.append(sep)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ptbench.tpt_bench.score_cli",
        description="Score vision_track's YOLO tracker on a TPT-Bench sequence.",
    )
    parser.add_argument("--seq", required=True, help="path to a sequence directory")
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU threshold (default 0.5)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="tracker inference size (default 1280)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="tracker detection confidence threshold (default 0.5)",
    )
    parser.add_argument(
        "--json", dest="json_out", default=None,
        help="optional path to dump metrics + config as JSON",
    )
    args = parser.parse_args(argv)

    try:
        frames = load_sequence(args.seq)
    except TptDatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Deferred import: only pulls in the heavy tracker when actually running.
    from .runner import run_tracker_on_sequence

    pred_boxes, scores = run_tracker_on_sequence(
        frames, imgsz=args.imgsz, conf=args.conf
    )

    gt_boxes = [f.gt_bbox for f in frames]
    metrics = compute_tpt_metrics(
        gt_boxes, pred_boxes, iou_thr=args.iou, scores=scores
    )

    print(f"sequence: {args.seq}")
    print(f"frames:   {len(frames)}  (iou_thr={args.iou}, imgsz={args.imgsz})")
    print(_format_table(metrics))

    if args.json_out:
        payload = {
            "sequence": args.seq,
            "num_frames": len(frames),
            "iou_thr": args.iou,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "metrics": metrics,
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
