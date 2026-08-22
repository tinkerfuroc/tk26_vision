#!/usr/bin/env python3
"""Time the OpenRouter VLM call used by seat_recommend_bbox.

Replays saved (image, request) pairs from `vision_log/` against the same
`request_seat()` helper the production node uses (`kimi_api._seat_vlm`),
so the system prompt, response schema, model, retry loop, and reasoning
extra_body are identical to a live `/seat_recommend_bbox_service` call —
we just skip ROS, depth resolution, snap-to-horizontal, and TF.

Each pair is replayed `--trials` times. Per-call elapsed seconds are
collected (the value `request_seat` already returns), then averaged.

Usage (from workspace root):

    python3 src/tk26_vision/scripts/tests/manual/seat_recommend_vlm_bench.py \\
        --n-images 8 --trials 3

Env: `OPENROUTER_API_KEY` loaded from workspace `.env` (same path the
production node uses).

Outputs land in `vision_log/seat_recommend_vlm_bench_<YYYYmmdd_HHMMSS>/`:
    summary.csv     one row per call
    summary.md      avg / p50 / p90 / p99 over successful calls
    <imgstem>_t<idx>.json   per-call detail
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[4]
_KIMI_SRC = _REPO_ROOT / 'src' / 'tk26_vision' / 'src' / 'kimi_api'
_VISION_UTIL_SRC = _REPO_ROOT / 'src' / 'tk26_vision' / 'src' / 'vision_util'
if str(_KIMI_SRC) not in sys.path:
    sys.path.insert(0, str(_KIMI_SRC))
if str(_VISION_UTIL_SRC) not in sys.path:
    sys.path.insert(0, str(_VISION_UTIL_SRC))

from kimi_api._env import load_env, require_api_key  # noqa: E402
from kimi_api._seat_vlm import VlmSeatError, request_seat  # noqa: E402
from vision_util.vlm_models import vision_vlm_model  # noqa: E402


@dataclass
class CallResult:
    image: str
    trial: int
    model: str
    n_names: int
    n_known_seats: int
    status: str           # 'ok' | 'error'
    elapsed_s: float | None = None
    label: str = ''
    point_xy: tuple[int, int] | None = None
    n_visible_seats: int = 0
    error: str = ''

    def csv_row(self) -> dict[str, Any]:
        return {
            'image': self.image,
            'trial': self.trial,
            'model': self.model,
            'n_names': self.n_names,
            'n_known_seats': self.n_known_seats,
            'status': self.status,
            'elapsed_s': '' if self.elapsed_s is None else f'{self.elapsed_s:.3f}',
            'label': self.label,
            'point_x': '' if self.point_xy is None else self.point_xy[0],
            'point_y': '' if self.point_xy is None else self.point_xy[1],
            'n_visible_seats': self.n_visible_seats,
            'error': self.error.replace('\n', ' \\n ')[:400],
        }


def discover_pairs(vision_log_root: Path, run_glob: str) -> list[tuple[Path, Path]]:
    """Yield (orig_jpg, req_json) pairs from seat_recommend_bbox runs."""
    pairs: list[tuple[Path, Path]] = []
    for run_dir in sorted(vision_log_root.glob(run_glob)):
        if not run_dir.is_dir() or run_dir.name.startswith('seat_recommend_vlm_bench_'):
            continue
        for orig in sorted(run_dir.glob('seat_recommend_bbox_*_orig_*.jpg')):
            req = orig.with_name(
                orig.name.replace('_orig_', '_req_').replace('.jpg', '.json'),
            )
            if req.exists():
                pairs.append((orig, req))
    return pairs


def load_request(req_path: Path) -> dict | None:
    try:
        meta = json.loads(req_path.read_text())
    except Exception:
        return None
    return meta.get('request') or {}


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float('nan')
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def write_summary_md(
    path: Path,
    results: list[CallResult],
    *,
    model: str,
    timeout_s: float,
    n_images: int,
    trials: int,
) -> None:
    ok = [r for r in results if r.status == 'ok' and r.elapsed_s is not None]
    err = [r for r in results if r.status != 'ok']
    elapsed = [r.elapsed_s for r in ok]  # type: ignore[misc]

    rows = [
        '# seat_recommend_bbox VLM latency\n',
        f'- Model: `{model}`\n',
        f'- Timeout: {timeout_s:.1f} s\n',
        f'- Sample: {n_images} images x {trials} trials = {len(results)} calls\n',
        f'- OK: {len(ok)}  Errors: {len(err)}\n',
        f'- Run: {datetime.now().isoformat(timespec="seconds")}\n\n',
    ]
    if elapsed:
        rows.append('| stat | seconds |\n|---|---|\n')
        rows.append(f'| mean | {statistics.fmean(elapsed):.2f} |\n')
        rows.append(f'| stdev | {statistics.pstdev(elapsed):.2f} |\n')
        rows.append(f'| min  | {min(elapsed):.2f} |\n')
        rows.append(f'| p50  | {percentile(elapsed, 0.50):.2f} |\n')
        rows.append(f'| p90  | {percentile(elapsed, 0.90):.2f} |\n')
        rows.append(f'| p99  | {percentile(elapsed, 0.99):.2f} |\n')
        rows.append(f'| max  | {max(elapsed):.2f} |\n')
    else:
        rows.append('_no successful calls_\n')
    if err:
        rows.append('\n## Errors\n')
        for r in err[:20]:
            rows.append(f'- `{r.image}` t{r.trial}: {r.error[:200]}\n')
    path.write_text(''.join(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=vision_vlm_model(),
                        help='OpenRouter model id (matches seat_recommend_bbox default).')
    parser.add_argument('--n-images', type=int, default=8)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--timeout-s', type=float, default=30.0)
    parser.add_argument('--max-retries', type=int, default=1,
                        help='Pass-through to request_seat; 1 keeps single-call latency clean.')
    parser.add_argument('--vision-log-root', default='vision_log')
    parser.add_argument('--run-glob', default='*')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inter-call-sleep', type=float, default=0.5,
                        help='Seconds between calls; spread out provider rate limits.')
    parser.add_argument('--warmup', action='store_true',
                        help='Discard the first call before timing (cache warmup).')
    args = parser.parse_args()

    load_env()
    try:
        require_api_key()
    except RuntimeError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    rng = random.Random(args.seed)

    vision_log_root = Path(args.vision_log_root).resolve()
    if not vision_log_root.is_dir():
        print(f'error: vision-log-root does not exist: {vision_log_root}', file=sys.stderr)
        return 2

    pairs = discover_pairs(vision_log_root, args.run_glob)
    if not pairs:
        print(
            f'error: no seat_recommend_bbox orig+req pairs found under '
            f'{vision_log_root} (run-glob={args.run_glob!r})',
            file=sys.stderr,
        )
        return 2
    print(f'discovered {len(pairs)} seat_recommend_bbox pairs in {vision_log_root}')

    n = min(args.n_images, len(pairs))
    sample = rng.sample(pairs, n)
    print(f'sampling {n} pairs (seed={args.seed})')

    cache: list[tuple[Path, Any, dict]] = []  # (img_path, bgr ndarray, request dict)
    for img_path, req_path in sample:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f'  skip (cv2.imread failed): {img_path.name}')
            continue
        req = load_request(req_path)
        if req is None:
            print(f'  skip (req.json parse failed): {req_path.name}')
            continue
        cache.append((img_path, bgr, req))
    if not cache:
        print('error: no images survived loading', file=sys.stderr)
        return 2

    out_root = vision_log_root / f'seat_recommend_vlm_bench_{datetime.now():%Y%m%d_%H%M%S}'
    out_root.mkdir(parents=True, exist_ok=False)
    print(f'writing results to {out_root}')

    if args.warmup:
        wp, wbgr, wreq = cache[0]
        print(f'  warmup on {wp.name}...', end=' ', flush=True)
        t0 = time.perf_counter()
        try:
            request_seat(
                wbgr,
                wreq.get('names') or [],
                wreq.get('features') or [],
                model=args.model,
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
                known_seats=wreq.get('known_seats') or None,
            )
            print(f'done ({time.perf_counter() - t0:.2f} s)')
        except Exception as exc:  # noqa: BLE001
            print(f'failed: {exc}')
        time.sleep(args.inter_call_sleep)

    schedule: list[tuple[int, tuple[Path, Any, dict]]] = []
    for trial in range(args.trials):
        for entry in cache:
            schedule.append((trial, entry))
    rng.shuffle(schedule)
    print(f'running {len(schedule)} timed calls (shuffled)')

    results: list[CallResult] = []
    csv_path = out_root / 'summary.csv'
    csv_fields = list(
        CallResult(image='', trial=0, model='', n_names=0, n_known_seats=0,
                   status='').csv_row().keys()
    )
    with csv_path.open('w', newline='') as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        writer.writeheader()
        for i, (trial, (img_path, bgr, req)) in enumerate(schedule, 1):
            names = list(req.get('names') or [])
            features = list(req.get('features') or [])
            known_seats = list(req.get('known_seats') or [])
            r = CallResult(
                image=img_path.name,
                trial=trial,
                model=args.model,
                n_names=len(names),
                n_known_seats=len(known_seats),
                status='error',
            )
            try:
                label, point_xy, visible_seats, elapsed = request_seat(
                    bgr,
                    names,
                    features,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    known_seats=known_seats or None,
                )
                r.elapsed_s = elapsed
                r.label = label or ''
                r.point_xy = point_xy
                r.n_visible_seats = len(visible_seats)
                r.status = 'ok'
            except VlmSeatError as exc:
                r.error = f'VlmSeatError: {exc}'
            except Exception as exc:  # noqa: BLE001
                r.error = (
                    f'{type(exc).__name__}: {exc}\n'
                    f'{traceback.format_exc(limit=2)}'
                )
            results.append(r)
            writer.writerow(r.csv_row())
            csv_fh.flush()

            detail_path = out_root / f'{img_path.stem}_t{trial}.json'
            detail_path.write_text(json.dumps(asdict(r), indent=2))

            t_str = f'{r.elapsed_s:.2f}s' if r.elapsed_s is not None else '----'
            tag = 'OK' if r.status == 'ok' else 'ERR'
            print(
                f'  [{i:>3}/{len(schedule)}] {img_path.stem[-30:]:>30} '
                f't={trial} elapsed={t_str:>7}  '
                f'label={r.label[:30]!r:>32}  vs={r.n_visible_seats}  {tag}'
                f'{("  " + r.error[:100]) if r.error else ""}'
            )
            time.sleep(args.inter_call_sleep)

    write_summary_md(
        out_root / 'summary.md',
        results,
        model=args.model,
        timeout_s=args.timeout_s,
        n_images=len(cache),
        trials=args.trials,
    )
    print('\n' + (out_root / 'summary.md').read_text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
