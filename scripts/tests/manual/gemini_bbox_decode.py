#!/usr/bin/env python3
"""Manual sanity check for the VLM bbox decode path.

Loads a saved JPEG from disk, calls `vlm_bbox.request_bboxes` with the given
prompt and model, draws the returned xyxy boxes on the image, and writes an
overlay PNG + the raw bbox list next to the source image. This is the single
source of ground truth for "does Gemini's [y0, x0, y1, x1] 0-1000 convention
match what _decode_bbox assumes?". Run after OpenRouter model bumps or any
edit to `_SYSTEM_PROMPT` / `_decode_bbox`.

Usage (with the tk26 venv + ROS env active so OPENROUTER_API_KEY loads):

    python3 src/tk26_vision/scripts/tests/manual/gemini_bbox_decode.py \\
        --image src/tk26_vision/scripts/tests/manual/fixtures/scene.jpg \\
        --prompt "monitor screen"

Outputs `<image_stem>_overlay.png` and `<image_stem>_raw.json` alongside
the input. Commit representative fixtures under `manual/fixtures/` so later
runs have a regression reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from vision_util.vlm_models import vision_vlm_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True,
                        help='Path to input JPEG/PNG.')
    parser.add_argument('--prompt', required=True,
                        help='Natural-language target class.')
    parser.add_argument('--model', default=vision_vlm_model(),
                        help='OpenRouter model slug.')
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--timeout-s', type=float, default=20.0)
    parser.add_argument('--out-dir', default=None,
                        help='Write overlay + raw.json here instead of '
                             'next to the input.')
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f'error: image not found: {image_path}', file=sys.stderr)
        return 2

    img = cv2.imread(str(image_path))
    if img is None:
        print(f'error: cv2 could not decode {image_path}', file=sys.stderr)
        return 2

    from object_detection_generalist.vlm_bbox import (
        VlmBboxError,
        request_bboxes,
    )

    try:
        bboxes, raw_labels, _elapsed = request_bboxes(
            img,
            args.prompt,
            model=args.model,
            max_retries=args.max_retries,
            timeout_s=args.timeout_s,
        )
    except VlmBboxError as exc:
        print(f'VLM call failed: {exc}', file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).resolve() if args.out_dir else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    overlay_path = out_dir / f'{stem}_overlay.png'
    raw_path = out_dir / f'{stem}_raw.json'

    overlay = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 2)
        caption = raw_labels[i] if i < len(raw_labels) and raw_labels[i] else args.prompt
        cv2.putText(overlay, caption, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imwrite(str(overlay_path), overlay)

    with open(raw_path, 'w') as fp:
        json.dump({
            'image': str(image_path),
            'prompt': args.prompt,
            'model': args.model,
            'bboxes_xyxy': [list(bbox) for bbox in bboxes],
            'raw_labels': list(raw_labels),
            'n_detections': len(bboxes),
        }, fp, indent=2)

    print(f'Wrote {overlay_path}')
    print(f'Wrote {raw_path}')
    print(f'Detections: {len(bboxes)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
