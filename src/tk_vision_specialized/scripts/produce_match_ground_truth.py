#!/usr/bin/env python3
"""Generate ground truth for the object_match_all batch-size benchmark.

Runs the existing single-category VLM call (qwen_match_vlm.request_match_bboxes)
over every (scene, category) pair and writes the high-confidence predictions
to a JSON file that the benchmark scorer consumes.

This is "VLM ground truth," not human ground truth. It measures agreement
with the single-category /object_match service we trust in production.
See spec §8.3.1 for the rationale and caveat.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2

from tk_vision_specialized.qwen_match_vlm import request_match_bboxes
from tk_vision_specialized.items_map_loader import ItemsMapLoader
from tk_vision_specialized.nms import MatchRow, suppress_within_category


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--scenes-dir', required=True, type=Path,
        help='Directory containing scene_*.jpg files.',
    )
    p.add_argument(
        '--items-dir', required=True, type=Path,
        help='Directory containing items_map.yaml + reference jpgs.',
    )
    p.add_argument(
        '--provider', default='qwen', choices=['qwen'],
        help='Only qwen is supported (production single-cat path).',
    )
    p.add_argument('--vlm-model', default='qwen3-vl-plus')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--min-conf', type=float, default=0.6)
    p.add_argument('--timeout-s', type=float, default=12.0)
    p.add_argument(
        '--out', required=True, type=Path,
        help='Path to write the GT JSON.',
    )
    return p.parse_args()


def main():
    args = _parse_args()
    items = ItemsMapLoader(str(args.items_dir))
    if len(items) == 0:
        print(f'No items found in {args.items_dir}', file=sys.stderr)
        return 1

    scenes = sorted(args.scenes_dir.glob('*.jpg')) + sorted(
        args.scenes_dir.glob('*.png'),
    )
    if not scenes:
        print(f'No scenes found in {args.scenes_dir}', file=sys.stderr)
        return 1

    out: dict = {
        '_meta': {
            'provider': args.provider,
            'vlm_model': args.vlm_model,
            'top_k': args.top_k,
            'min_conf': args.min_conf,
            'items': sorted(items.keys()),
            'generated_at': (
                datetime.datetime.utcnow().isoformat() + 'Z'
            ),
        },
    }

    for scene_path in scenes:
        rgb = cv2.imread(str(scene_path))
        if rgb is None:
            print(
                f'skip unreadable scene {scene_path}', file=sys.stderr,
            )
            continue
        scene_gt: list[MatchRow] = []
        for category in items.keys():
            ref_url = items.get_data_url(category)
            boxes, confs, _labels, _elapsed = request_match_bboxes(
                rgb, ref_url, item_name=category, top_k=args.top_k,
                timeout_s=args.timeout_s, max_retries=1,
            )
            for bbox, conf in zip(boxes, confs):
                if conf >= args.min_conf:
                    scene_gt.append(MatchRow(
                        label=category, bbox=tuple(bbox), conf=conf,
                    ))
        scene_gt = suppress_within_category(scene_gt, iou_thresh=0.5)
        out[scene_path.name] = [
            {'category': r.label, 'bbox': list(r.bbox), 'conf': r.conf}
            for r in scene_gt
        ]
        print(
            f'{scene_path.name}: {len(scene_gt)} GT items',
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f'wrote {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
