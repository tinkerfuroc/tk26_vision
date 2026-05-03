#!/usr/bin/env python3
"""Pre-warm every checkpoint the tk26_vision stack loads at runtime.

The nodes already download on-demand through ``vision_util.weights_cache``,
so this script is optional — run it once before a match / demo to avoid
paying the download cost at node-start time. It populates the *same* cache
the nodes use, so any future cold-start resolves to a local file.

Coverage:
  * Ultralytics YOLO / YOLO-seg / YOLO-World / SAM (MobileSAM) .pt weights
  * torchvision ResNet50 + ResNet18 ImageNet weights (custom ReID)
  * MediaPipe Pose landmark model (waving detection)

Not covered (API-only, no local weights):
  * Gemini 2.5 Flash via OpenRouter (object_detection_generalist, enable_vlm)
  * OpenRouter LLMs used by kimi_api
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VISION_DIR = SCRIPT_DIR.parent
# vision_util source isn't on sys.path until colcon-installed, but running
# this straight from the repo is convenient — splice it in ahead of time.
sys.path.insert(0, str(VISION_DIR / "src" / "vision_util"))

from vision_util.weights_cache import resolve_weights  # noqa: E402

# Default manifest — what the nodes declare out of the box. Extra sizes are
# opt-in; nobody wants ~1 GB of l/x variants they never load.
DEFAULT_MANIFEST = [
    "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
    "yolov8s-seg.pt", "yolov8s.pt",
    "yolov8s-worldv2.pt",
    "mobile_sam.pt",
]
EXTRA_SIZES = [
    "yolo11l-seg.pt", "yolo11x-seg.pt",
    "yolov8n-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
    "yolov8m-worldv2.pt", "yolov8l-worldv2.pt", "yolov8x-worldv2.pt",
]


def human_size(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def warm_ultralytics(names: list[str]) -> None:
    for name in names:
        path = resolve_weights(name)
        print(f"  ✓ {name:<26} {human_size(path)}  ({path})")


def warm_torchvision() -> None:
    print("warming torchvision (~/.cache/torch/hub/checkpoints)…")
    try:
        import torchvision.models as M
    except ImportError as exc:
        print(f"  ! torchvision unavailable ({exc}); skipping")
        return
    M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
    M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    print("  ✓ resnet50 + resnet18")


def warm_mediapipe() -> None:
    print("warming mediapipe pose landmark model…")
    try:
        import mediapipe as mp
    except ImportError as exc:
        print(f"  ! mediapipe unavailable ({exc}); skipping. "
              "`pip install mediapipe` in .venv-vision-main if you use "
              "tk_vision_specialized.waving_person_server.")
        return
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    pose.close()
    print("  ✓ mediapipe pose")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extra-sizes", action="store_true",
                    help="also warm l/x variants (~1 GB extra)")
    ap.add_argument("--skip-ultralytics", action="store_true")
    ap.add_argument("--skip-torchvision", action="store_true")
    ap.add_argument("--skip-mediapipe", action="store_true")
    args = ap.parse_args()

    names = list(DEFAULT_MANIFEST)
    if args.extra_sizes:
        names += EXTRA_SIZES

    if not args.skip_ultralytics:
        warm_ultralytics(names)
    if not args.skip_torchvision:
        warm_torchvision()
    if not args.skip_mediapipe:
        warm_mediapipe()

    print("all vision checkpoints staged in the shared weights cache.")


if __name__ == "__main__":
    main()
