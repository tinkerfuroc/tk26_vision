#!/usr/bin/env python3
"""Sweep batch_size for object_match_all and report
precision/recall/F1/latency.

Reads scenes + GT JSON, runs the configured provider's MatchClient with
each batch_size, scores against GT, writes a CSV and Markdown summary."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

from tk_vision_specialized.items_map_loader import ItemsMapLoader
from tk_vision_specialized.nms import MatchRow, iou
from tk_vision_specialized.vlm_match_client import build_match_client


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes-dir', required=True, type=Path)
    p.add_argument('--items-dir', required=True, type=Path)
    p.add_argument('--ground-truth', required=True, type=Path)
    p.add_argument(
        '--batch-sizes', type=int, nargs='+',
        default=[1, 2, 3, 5, 8],
    )
    p.add_argument(
        '--provider', default='qwen',
        choices=['qwen', 'gemini', 'both'],
    )
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out-prefix', required=True, type=Path)
    p.add_argument('--timeout-s', type=float, default=12.0)
    p.add_argument(
        '--iou-thresh', type=float, default=0.3,
        help='IoU threshold for TP scoring.',
    )
    return p.parse_args()


def _score(predictions, ground_truth, iou_thresh):
    """Standard set-matching: a prediction is TP if it has a same-label
    GT box with IoU >= thresh that hasn't been matched yet.
    Multi-prediction-to-one-GT is greedy by descending confidence."""
    preds = sorted(predictions, key=lambda r: -r.conf)
    gt_remaining = list(ground_truth)
    tp = 0
    for p in preds:
        for i, g in enumerate(gt_remaining):
            if g.label != p.label:
                continue
            if iou(p.bbox, g.bbox) >= iou_thresh:
                tp += 1
                gt_remaining.pop(i)
                break
    fp = len(preds) - tp
    fn = len(gt_remaining)
    return tp, fp, fn


def _provider_list(arg):
    if arg == 'both':
        return ['qwen', 'gemini']
    return [arg]


def main():
    args = _parse_args()
    items = ItemsMapLoader(str(args.items_dir))
    gt_raw = json.loads(args.ground_truth.read_text())
    meta = gt_raw.get('_meta', {})
    gt_items = set(meta.get('items', []))
    if gt_items and gt_items != set(items.keys()):
        print(
            f'GT items {sorted(gt_items)} differ from current items_map '
            f'{sorted(items.keys())}; regenerate GT.',
            file=sys.stderr,
        )
        return 1

    refs_all = [(k, items.get_data_url(k)) for k in items.keys()]
    scenes = sorted(
        p for p in args.scenes_dir.iterdir()
        if p.suffix.lower() in {'.jpg', '.png'}
    )

    rows: list[dict] = []

    for provider in _provider_list(args.provider):
        client = build_match_client(provider)
        for B in args.batch_sizes:
            for scene_path in scenes:
                rgb = cv2.imread(str(scene_path))
                if rgb is None:
                    continue
                gt = [
                    MatchRow(
                        label=e['category'],
                        bbox=tuple(e['bbox']),
                        conf=float(e['conf']),
                    )
                    for e in gt_raw.get(scene_path.name, [])
                ]
                for r in range(args.repeats):
                    batches = [
                        refs_all[i:i + B]
                        for i in range(0, len(refs_all), B)
                    ]
                    t0 = time.perf_counter()
                    preds: list[MatchRow] = []
                    for batch in batches:
                        try:
                            preds.extend(client.match_batch(
                                rgb, batch,
                                timeout_s=args.timeout_s,
                                max_retries=1,
                            ))
                        except Exception as exc:    # noqa: BLE001
                            print(
                                f'batch fail provider={provider} B={B} '
                                f'scene={scene_path.name}: {exc}',
                                file=sys.stderr,
                            )
                    elapsed = time.perf_counter() - t0
                    tp, fp, fn = _score(preds, gt, args.iou_thresh)
                    rows.append({
                        'scene': scene_path.name,
                        'provider': provider,
                        'batch_size': B,
                        'repeat': r,
                        'n_calls': len(batches),
                        'elapsed_s': elapsed,
                        'tp': tp, 'fp': fp, 'fn': fn,
                        'n_pred': len(preds), 'n_gt': len(gt),
                    })
                    print(
                        f'  {scene_path.name} provider={provider} B={B} '
                        f'r={r} tp={tp} fp={fp} fn={fn} {elapsed:.1f}s'
                    )

    csv_path = args.out_prefix.with_suffix('.csv')
    md_path = args.out_prefix.with_suffix('.md')
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open('w', newline='') as fh:
        if rows:
            writer = csv.DictWriter(
                fh, fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    summary: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row['provider'], row['batch_size'])
        s = summary.setdefault(
            key, {'f1s': [], 'lats': [], 'precs': [], 'recs': []},
        )
        prec = row['tp'] / max(1, row['tp'] + row['fp'])
        rec = row['tp'] / max(1, row['tp'] + row['fn'])
        f1 = (
            0.0 if prec + rec == 0
            else 2 * prec * rec / (prec + rec)
        )
        s['precs'].append(prec)
        s['recs'].append(rec)
        s['f1s'].append(f1)
        s['lats'].append(row['elapsed_s'])

    lines = [
        '# Batch-size benchmark summary',
        f'GT: {args.ground_truth.name}',
        f'Generated: {datetime.datetime.utcnow().isoformat()}Z',
        '',
        '| provider | batch_size | median F1 | '
        'median latency (s) | p95 latency (s) |',
        '|---|---|---|---|---|',
    ]
    for (provider, B), s in sorted(summary.items()):
        f1 = statistics.median(s['f1s'])
        lat_med = statistics.median(s['lats'])
        lat_p95 = sorted(s['lats'])[
            max(0, int(len(s['lats']) * 0.95) - 1)
        ]
        lines.append(
            f'| {provider} | {B} | {f1:.3f} | '
            f'{lat_med:.2f} | {lat_p95:.2f} |'
        )

    lines.append('')
    lines.append('## Recommended batch_size')
    for provider in _provider_list(args.provider):
        candidates = [
            (B, summary[(provider, B)])
            for B in args.batch_sizes
            if (provider, B) in summary
        ]
        if not candidates:
            continue
        best_B, best_s = max(
            candidates,
            key=lambda kv: (
                statistics.median(kv[1]['f1s']),
                -statistics.median(kv[1]['lats']),
            ),
        )
        lines.append(
            f'- **{provider}**: `batch_size = {best_B}` '
            f'(median F1 {statistics.median(best_s["f1s"]):.3f}, '
            f'median latency '
            f'{statistics.median(best_s["lats"]):.2f}s)'
        )

    md_path.write_text('\n'.join(lines))
    print(f'wrote {csv_path} and {md_path}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
