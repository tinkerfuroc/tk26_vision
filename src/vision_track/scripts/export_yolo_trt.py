#!/usr/bin/env python3
"""Export a YOLO seg model to a FP16 TensorRT engine (OPTIONAL top-end speedup).

The engine is RESOLUTION- and BATCH-LOCKED to the imgsz used here; the live node
MUST run YOLO at the same imgsz (person_track_node ``inference_size`` param). This
is hardware-specific (built for THIS GPU/TensorRT version) and is not portable —
re-export on each deployment box. Best-effort: if TensorRT is absent, the .pt
model continues to work unchanged (this script errors clearly; the node does not
need it).

This script is standalone (no ROS, no rclpy). It needs ``ultralytics`` (present
in ``.venv-vision-main``) plus ``tensorrt`` + a CUDA GPU at *export* time.
``tensorrt`` is NOT installed in ``.venv-vision-main`` — provision it on the
deployment workstation before running the export (Ultralytics will pull a
matching build, or install it manually for your CUDA version).

Usage:
    export_yolo_trt.py --model yolo11s-seg.pt --imgsz 736 --out yolo11s-seg.engine
Verify (manual, on the robot with live cameras):
    ros2 run vision_track person_track_server --ros-args \
        -p model_path:=/abs/path/yolo11s-seg.engine -p inference_size:=736
"""
import argparse
import importlib.util
import sys


def _tensorrt_available() -> bool:
    """Return True if the ``tensorrt`` module can be imported (no import side effect)."""
    return importlib.util.find_spec("tensorrt") is not None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a YOLO seg model to a FP16 TensorRT engine (optional, "
        "best-effort, hardware-specific). The .pt path remains the default/fallback.",
    )
    ap.add_argument("--model", default="yolo11s-seg.pt",
                    help="Path to the source YOLO .pt weights (default: yolo11s-seg.pt).")
    ap.add_argument("--imgsz", type=int, default=736,
                    help="Inference size to LOCK the engine to. The live node's "
                         "inference_size param MUST match this (default: 736).")
    ap.add_argument("--device", default="0",
                    help="CUDA device index for the export (default: 0).")
    ap.add_argument("--out", default=None,
                    help="Optional rename/copy target for the produced .engine.")
    args = ap.parse_args()

    # Preflight: fail clearly when TensorRT is absent rather than surfacing an
    # opaque ultralytics error deep inside the export. The .pt model keeps working.
    if not _tensorrt_available():
        print(
            "ERROR: the 'tensorrt' package is not importable in this Python "
            "environment.\n"
            "TensorRT export is OPTIONAL and hardware-specific — it requires "
            "TensorRT + a CUDA GPU.\n"
            "In tk26_vision, 'tensorrt' is NOT in .venv-vision-main (it lives in "
            ".venv-fs for foundation_stereo).\n"
            "Provision TensorRT for your CUDA version on the deployment box, then "
            "re-run.\n"
            "The .pt model continues to work unchanged without this engine.",
            file=sys.stderr,
        )
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ERROR: ultralytics is required to export: {exc}", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    # device accepts an int index; ultralytics also tolerates the string.
    try:
        device = int(args.device)
    except ValueError:
        device = args.device
    engine_path = model.export(format="engine", half=True, imgsz=args.imgsz, device=device)
    print(f"Exported TensorRT engine: {engine_path}")
    if args.out and args.out != str(engine_path):
        import shutil
        shutil.copyfile(engine_path, args.out)
        print(f"Copied to: {args.out}")
    print(
        "RUN: ros2 run vision_track person_track_server --ros-args "
        f"-p model_path:=<abs>/{args.out or engine_path} -p inference_size:={args.imgsz}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
