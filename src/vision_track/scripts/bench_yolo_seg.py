#!/usr/bin/env python3
"""Benchmark YOLO-seg inference latency at the tracker's imgsz (default 736).

Conservative upper bound: this times the plain PyTorch ``.pt`` forward (fp16 on
CUDA). A TensorRT engine (see ``export_yolo_trt.py``) is only FASTER, so if a
model clears the per-frame budget here it clears it with TRT too. Use this to
decide whether a larger seg model (``yolo11m-seg`` / ``l``) is fast enough before
exporting a per-box engine on the robot.

Standalone (no ROS). Needs ``ultralytics`` + a CUDA GPU (both in
``.venv-vision-main``).

    python scripts/bench_yolo_seg.py --models yolo11s-seg.pt yolo11m-seg.pt --imgsz 736
"""
import argparse
import statistics
import time

import numpy as np


def bench_one(model_path: str, imgsz: int, half: bool, iters: int, warmup: int) -> dict:
    """Time `iters` forward passes of one model and return latency stats."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    # A representative 720p frame; ultralytics letterboxes to imgsz internally.
    frame = (np.random.rand(720, 1280, 3) * 255).astype("uint8")
    common = dict(imgsz=imgsz, half=half, device=0, verbose=False)
    for _ in range(warmup):
        model.predict(frame, **common)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(frame, **common)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "model": model_path,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * len(samples)) - 1],
        "max_fps": 1000.0 / statistics.mean(samples),
    }


def main() -> int:
    """Parse args and print a latency table for the requested models."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+",
                    default=["yolo11s-seg.pt", "yolo11m-seg.pt"],
                    help="YOLO-seg weights to benchmark (paths or names).")
    ap.add_argument("--imgsz", type=int, default=736,
                    help="Inference size — match person_track_node inference_size.")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--fp32", action="store_true", help="Disable fp16 (default fp16).")
    args = ap.parse_args()

    print(f"imgsz={args.imgsz} half={not args.fp32} iters={args.iters} warmup={args.warmup}")
    print(f"{'model':<22}{'mean_ms':>10}{'median_ms':>11}{'p95_ms':>9}{'max_fps':>9}")
    for m in args.models:
        r = bench_one(m, args.imgsz, not args.fp32, args.iters, args.warmup)
        print(f"{r['model']:<22}{r['mean_ms']:>10.2f}{r['median_ms']:>11.2f}"
              f"{r['p95_ms']:>9.2f}{r['max_fps']:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
