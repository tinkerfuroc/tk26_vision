#!/usr/bin/env python3
"""Score vision_track on the LaSOT `person` category — reproducible benchmark run.

TPT-Bench (the originally-intended Tier-B set) is download-blocked behind
OneDrive/Baidu, so the realized external regression benchmark is LaSOT's
`person` category, which is directly downloadable from HuggingFace and is a
drop-in for the `ptbench.tpt_bench` scorer. See DOWNLOAD.md.

Paths are derived from this file's location (no hard-coded worktree path), so it
keeps working after the branch merges. The LaSOT data dir defaults to
``~/datasets/lasot`` (override with --data).

Usage:
    # core tracker (matches what person_track_server runs, minus the node FSM):
    .venv-vision-main/bin/python demo/run_lasot_person_benchmark.py
    # only the Protocol-II test split:
    ... --seqs person-1 person-5 person-10 person-12
    # FSM ablation: attach the node's LockStateMachine (depth permissive on RGB):
    ... --fsm --seqs person-1 person-5 person-10 person-12

Production faithfulness: imgsz=736 (config inference_size), conf=0.5; the
OSNet-AIN/MSMT17 ReID, fp16, and yolo_track_conf=0.15 are YOLOTracker defaults.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PTBENCH_ROOT = os.path.dirname(HERE)                              # benchmarks/person_tracker
TK26_ROOT = os.path.dirname(os.path.dirname(PTBENCH_ROOT))        # src/tk26_vision
sys.path.insert(0, PTBENCH_ROOT)
sys.path.insert(0, os.path.join(TK26_ROOT, "src", "vision_track"))

from ptbench.tpt_bench.dataset import load_sequence            # noqa: E402
from ptbench.tpt_bench.metrics import compute_tpt_metrics      # noqa: E402

TEST_SPLIT = ("person-1", "person-5", "person-10", "person-12")
IMGSZ, CONF, IOU = 736, 0.5, 0.5


def _rgb(path: str):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def _make_tracker(use_fsm: bool, no_gallery: bool = False):
    """Build a YOLOTracker; optionally attach the node's LockStateMachine.

    The plain offline path leaves lock_state_machine=None (tracker core only).
    --fsm replicates person_track_node's FSM wiring so the recovery
    publish/hysteresis gate runs — depth stays permissive (RGB-only can't drive
    the crosser-rejection gate), so this isolates the FSM's hysteresis effect.
    """
    from vision_track.track_yolo import YOLOTracker
    trk = YOLOTracker(confidence_threshold=CONF, inference_size=IMGSZ,
                      reid_gallery_enabled=not no_gallery)
    if use_fsm:
        from vision_track.core.lock_state_machine import LockStateMachine
        trk.max_frames_lost = 600
        trk.frame_rate = 15.0
        trk.max_recovery_frames = 45
        trk.provisional_high_bar = 0.72
        trk.provisional_distinct_margin = 0.10
        trk.crosser_depth_jump_m = 0.6
        trk.lock_state_machine = LockStateMachine(
            high_bar=0.72, distinct_margin=0.10, commit_frames=12, max_recovery_frames=45,
        )
    return trk


def run_sequence(frames, use_fsm: bool, no_gallery: bool = False):
    trk = _make_tracker(use_fsm, no_gallery=no_gallery)
    preds, scores = [], []
    initialized = False
    for f in frames:
        rgb = _rgb(f.image_path)
        if not initialized:
            if f.gt_bbox is None:
                preds.append(None); scores.append(0.0); continue
            x1, y1, x2, y2 = (int(round(v)) for v in f.gt_bbox)
            ok = trk.initialize_tracking(rgb, target_bbox=(x1, y1, x2, y2), target_class="person")
            initialized = True
            if ok and use_fsm and getattr(trk, "lock_state_machine", None) is not None \
                    and trk.original_track_id is not None:
                trk.lock_state_machine.start(trk.original_track_id)  # node _try_initialize
            if ok:
                preds.append((float(x1), float(y1), float(x2), float(y2))); scores.append(1.0)
            else:
                preds.append(None); scores.append(0.0)
            continue
        res = trk.update(rgb)
        if use_fsm and res is not None:  # node _handle_tracked_frame present-by-id re-step
            fsm = getattr(trk, "lock_state_machine", None)
            present = (
                not bool(getattr(trk, "last_frame_recovery", False))
                and trk.target_track_id is not None
                and res.track_id == trk.original_track_id
                and getattr(trk, "frames_lost", 0) == 0
            )
            if fsm is not None and present:
                trk.last_lock_decision = fsm.step(
                    sim_score=1.0, present=True, frames_since_loss=0,
                    num_candidates=1, distinct_margin=float("inf"), depth_consistent=True,
                )
        if res is None or res.bbox is None:
            preds.append(None); scores.append(0.0)
        else:
            preds.append(tuple(float(v) for v in res.bbox))
            cv = getattr(res, "confidence", None)
            scores.append(float(cv) if cv is not None else 1.0)
    return preds, scores


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.expanduser("~/datasets/lasot"),
                    help="dir holding person-* sequence dirs (default ~/datasets/lasot)")
    ap.add_argument("--seqs", nargs="*", default=None,
                    help="sequence names (default: all person-* found)")
    ap.add_argument("--fsm", action="store_true", help="attach the node LockStateMachine")
    ap.add_argument("--no-gallery", dest="no_gallery", action="store_true",
                    help="disable the multi-view ReID gallery (legacy avg/anchor scoring)")
    ap.add_argument("--json", dest="json_out", default=None, help="dump results JSON")
    args = ap.parse_args(argv)

    if args.seqs:
        seqs = args.seqs
    else:
        seqs = sorted(
            (os.path.basename(p) for p in glob.glob(f"{args.data}/person-*") if os.path.isdir(p)),
            key=lambda s: int(s.split("-")[1]),
        )
    if not seqs:
        print(f"no person-* sequences under {args.data!r}", file=sys.stderr)
        return 2

    results = {}
    for s in seqs:
        fr = load_sequence(os.path.join(args.data, s))
        t = time.time()
        pred, sc = run_sequence(fr, args.fsm, no_gallery=args.no_gallery)
        dt = time.time() - t
        m = compute_tpt_metrics([f.gt_bbox for f in fr], pred, iou_thr=IOU, scores=sc)
        m["frames"] = len(fr); m["throughput_hz"] = len(fr) / dt
        results[s] = m
        print(f"{s}: {len(fr)}f {m['throughput_hz']:.1f}Hz P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F={m['f_score']:.3f} AO={m['ao']:.3f} AMR={m['amr']:.3f}",
              flush=True)

    keys = ["precision", "recall", "f_score", "ao", "amr", "throughput_hz"]
    mean = {k: st.mean(results[s][k] for s in seqs) for k in keys}
    print(f"MEAN ({len(seqs)} seq, fsm={args.fsm}, gallery={not args.no_gallery}):",
          {k: round(v, 3) for k, v in mean.items()})
    if args.json_out:
        json.dump({"per_seq": results, "mean": mean,
                   "config": {"imgsz": IMGSZ, "conf": CONF, "iou": IOU, "fsm": args.fsm,
                              "gallery": not args.no_gallery}},
                  open(args.json_out, "w"), indent=2)
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
