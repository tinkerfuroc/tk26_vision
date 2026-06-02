"""CLI: replay a bag through the person tracker and score it against GT.

Usage::

    python -m ptbench.replay.score_cli --bag DIR --gt GT.json \
        [--backend offline|action] [--imgsz 1280] [--conf 0.5] [--json out.json]

Loads the GT clip, runs the chosen backend over the bag to get a prediction
stream, aligns predictions to GT frames by timestamp, computes the scoreboard
metrics, scores them against the gates, prints the table, and optionally dumps
the scoreboard dict as JSON.

The align→metrics→score wiring is factored into :func:`score_preds`, which does
**not** import the tracker, so it is unit-testable with hand-built predictions.
``main`` is the only part that pulls in the (deferred) runner backends.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from ..common.align import PredFrame, align_pred_to_gt
from ..common.metrics import compute_metrics
from ..common.schema import GtClip, GtSchemaError, load_gt
from ..common.scoreboard import GateConfig, Scoreboard, score


def score_preds(
    preds: List[PredFrame],
    gt_clip: GtClip,
    throughput_hz: Optional[float] = None,
    gates: Optional[GateConfig] = None,
) -> Scoreboard:
    """Align ``preds`` to ``gt_clip.frames``, compute metrics, and score them.

    Pure glue over the committed ``common`` pipeline — no tracker import — so it
    can be exercised end-to-end (align → metrics → score) without a model.

    Args:
        preds: prediction stream from a runner backend.
        gt_clip: validated GT clip whose ``frames`` are the scoring reference.
        throughput_hz: measured tracker throughput (fed straight into the
            metrics + throughput gate); ``None`` ⇒ that row scores N/A.
        gates: PASS/WARN/FAIL thresholds.

    Returns:
        a :class:`~ptbench.common.scoreboard.Scoreboard`.
    """
    aligned = align_pred_to_gt(preds, gt_clip.frames)
    metrics = compute_metrics(aligned, throughput_hz=throughput_hz)
    return score(metrics, gates)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ptbench.replay.score_cli",
        description="Replay a rosbag through the person tracker and score it.",
    )
    parser.add_argument("--bag", required=True, help="rosbag2 directory to replay")
    parser.add_argument("--gt", required=True, help="GT annotation JSON path")
    parser.add_argument(
        "--backend",
        choices=("offline", "action"),
        default="offline",
        help="prediction backend (default offline: drive YOLOTracker in-process; "
        "action: replay onto a live /track_person server)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="tracker inference size (offline backend; default 1280)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="tracker confidence threshold (offline backend; default 0.5)",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="optional path to dump the scoreboard dict as JSON",
    )
    args = parser.parse_args(argv)

    try:
        clip = load_gt(args.gt)
    except (OSError, ValueError, GtSchemaError) as exc:
        print(f"error loading GT: {exc}", file=sys.stderr)
        return 2

    # Deferred import: only pull in the heavy runner when actually running.
    from .runner import run_action, run_offline

    if args.backend == "offline":
        preds, throughput_hz = run_offline(
            args.bag, clip, imgsz=args.imgsz, conf=args.conf
        )
    else:
        preds, throughput_hz = run_action(args.bag, clip)

    board = score_preds(preds, clip, throughput_hz=throughput_hz)

    print(f"clip:    {clip.clip_id}  (scenario={clip.scenario})")
    print(f"bag:     {args.bag}")
    print(
        f"frames:  gt={len(clip.frames)}  preds={len(preds)}  "
        f"backend={args.backend}  throughput_hz={throughput_hz:.2f}"
    )
    print(board.to_table())

    if args.json_out:
        payload = {
            "clip_id": clip.clip_id,
            "scenario": clip.scenario,
            "bag": args.bag,
            "backend": args.backend,
            "num_gt_frames": len(clip.frames),
            "num_preds": len(preds),
            "throughput_hz": throughput_hz,
            "scoreboard": board.to_dict(),
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
