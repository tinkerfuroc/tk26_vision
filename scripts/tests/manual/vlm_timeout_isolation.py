#!/usr/bin/env python3
"""Phase A — isolation test for the per-attempt 10 s OpenRouter timeout.

Replays saved generalist orig+req pairs from `vision_log/` against TWO retry
policies, side-by-side on the same images:

  baseline   - calls production `vlm_bbox.request_bboxes()` directly
               (timeout_s=20, max_retries=3, hard_deadline shared across attempts)
  capped     - sibling `request_bboxes_capped()` defined in this file:
               per_attempt_timeout_s=10, max_retries=3, fresh deadline per
               attempt, abandons stream + drops client on each timeout

Production code is NOT modified by this harness. Use it to confirm the
proposed semantics catch the 40 s tail without regressing success rate before
touching `vlm_bbox.py` / `generalist_node.py`.

Usage (workspace root, after `source install/setup.bash`):

    python3 src/tk26_vision/scripts/tests/manual/vlm_timeout_isolation.py \\
        [--n-images 10] [--trials 3] [--per-attempt-s 10] [--max-attempts 3] \\
        [--overall-cap-s 30] [--baseline-timeout-s 20]

Outputs:

    vision_log/vlm_timeout_isolation_<ts>/summary.csv
    vision_log/vlm_timeout_isolation_<ts>/summary.md
    vision_log/vlm_timeout_isolation_<ts>/<policy>_<imgstem>_t<idx>.json
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
from typing import Any, Callable

import cv2

# Make tk26_vision packages importable from a fresh shell.
_REPO_ROOT = Path(__file__).resolve().parents[4]
for _src_dir in (
    _REPO_ROOT / 'src' / 'tk26_vision' / 'src' / 'kimi_api',
    _REPO_ROOT / 'src' / 'tk26_vision' / 'src' / 'object_detection_generalist',
    _REPO_ROOT / 'src' / 'tk26_vision' / 'src' / 'vision_util',
):
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

from kimi_api._env import (  # noqa: E402
    base_url as openrouter_base_url,
    load_env,
    require_api_key as openrouter_api_key,
)
from kimi_api._image_utils import encode_to_data_url  # noqa: E402
from object_detection_generalist.vlm_bbox import (  # noqa: E402
    _GEMINI_SYSTEM_PROMPT,
    _RESPONSE_SCHEMA,
    request_bboxes as production_request_bboxes,
)
from vision_util.vlm_models import vision_flash_model  # noqa: E402


# ---------------------------------------------------------------------------
# Per-call result


@dataclass
class CallResult:
    policy: str
    image: str
    prompt: str
    trial: int
    status: str = 'error'        # ok | error
    error: str = ''
    total_ms: float | None = None  # full wall-clock incl. retries
    n_attempts: int = 0
    successful_attempt_ms: float | None = None
    n_detections: int | None = None

    def csv_row(self) -> dict[str, Any]:
        return {
            'policy': self.policy,
            'image': self.image,
            'prompt': self.prompt,
            'trial': self.trial,
            'status': self.status,
            'total_ms': '' if self.total_ms is None else f'{self.total_ms:.1f}',
            'attempt_ms': '' if self.successful_attempt_ms is None
                          else f'{self.successful_attempt_ms:.1f}',
            'n_attempts': self.n_attempts,
            'n_detections': '' if self.n_detections is None else self.n_detections,
            'error': self.error.replace('\n', ' \\n ')[:400],
        }


# ---------------------------------------------------------------------------
# Capped policy (under test). Implemented standalone here — does NOT modify
# production. Mirrors the production message + schema shape so we test the
# same SDK-level behavior, just with the new retry / timeout discipline.


def request_bboxes_capped(
    rgb_bgr,
    prompt: str,
    *,
    model: str,
    per_attempt_timeout_s: float,
    max_attempts: int,
    overall_cap_s: float,
    inject_latency_s: float = 0.0,
    logger=None,
) -> tuple[bool, str, int, float, float | None, str]:
    """Issue the same VLM bbox call as production, but with per-attempt cap +
    fresh client/stream per attempt + an overall wall-clock backstop.

    Returns
    -------
    (ok, raw_response_text, n_attempts, total_ms, successful_attempt_ms, error)
    """
    from openai import OpenAI

    h, w = rgb_bgr.shape[:2]
    data_url = encode_to_data_url(rgb_bgr)
    user_text = f'Target classes: {prompt}. Original image size: width={w}, height={h}.'
    messages = [
        {'role': 'system', 'content': _GEMINI_SYSTEM_PROMPT},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': user_text},
            {'type': 'image_url', 'image_url': {'url': data_url}},
        ]},
    ]
    response_format = {
        'type': 'json_schema',
        'json_schema': {
            'name': 'detections_response',
            'strict': True,
            'schema': _RESPONSE_SCHEMA,
        },
    }

    overall_t0 = time.perf_counter()
    last_error = ''
    api_key = openrouter_api_key()
    base = openrouter_base_url()

    for attempt in range(1, max_attempts + 1):
        if (time.perf_counter() - overall_t0) > overall_cap_s:
            last_error = f'overall cap {overall_cap_s}s exhausted before attempt {attempt}'
            break

        # Fresh client per attempt — guarantees no stuck connection from
        # the previous attempt's closed-mid-stream state can leak in.
        client = OpenAI(
            api_key=api_key, base_url=base,
            max_retries=0, timeout=per_attempt_timeout_s,
        )
        attempt_t0 = time.perf_counter()
        raw_parts: list[str] = []
        stream = None
        injected_this_attempt = False
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=0,
                stream=True,
            )
            for chunk in stream:
                # httpx will already enforce the per-phase read timeout, but
                # add a defensive wall-clock check too in case chunks trickle.
                if (time.perf_counter() - attempt_t0) > per_attempt_timeout_s:
                    raise TimeoutError(
                        f'attempt {attempt} exceeded {per_attempt_timeout_s}s wall-clock'
                    )
                err = getattr(chunk, 'error', None)
                if err:
                    raise RuntimeError(f'openrouter stream error: {err}')
                choices = getattr(chunk, 'choices', None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], 'delta', None)
                piece = getattr(delta, 'content', None) if delta else None
                if piece:
                    raw_parts.append(piece)
                    # Latency injection (test-only): once per attempt, after
                    # the first content chunk, sleep long enough that the
                    # wall-clock guard above fires on the next iteration.
                    if inject_latency_s > 0 and not injected_this_attempt:
                        injected_this_attempt = True
                        if logger:
                            logger(f'  attempt {attempt}: injecting {inject_latency_s}s sleep')
                        time.sleep(inject_latency_s)
            attempt_ms = (time.perf_counter() - attempt_t0) * 1000.0
            raw = ''.join(raw_parts)
            # Validate that we got a parseable JSON payload — otherwise the
            # attempt counts as a failure (matches what production retry
            # logic catches as JSONDecodeError / ValueError).
            try:
                json.loads(raw or '{}')
            except json.JSONDecodeError as exc:
                last_error = f'JSONDecodeError attempt {attempt}: {exc}'
                if logger:
                    logger(f'  attempt {attempt}: bad JSON — {exc}')
                continue
            return True, raw, attempt, (time.perf_counter() - overall_t0) * 1000.0, attempt_ms, ''
        except Exception as exc:
            last_error = f'{type(exc).__name__} attempt {attempt}: {exc}'
            if logger:
                logger(f'  attempt {attempt}: {type(exc).__name__}: {exc}')
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            try:
                client.close()
            except Exception:
                pass

    total_ms = (time.perf_counter() - overall_t0) * 1000.0
    return False, '', max_attempts, total_ms, None, last_error


# ---------------------------------------------------------------------------
# Baseline wrapper around production request_bboxes() so we measure the
# SAME function the runtime calls. We can't easily get attempt-count out of
# it, so we treat n_attempts as best-effort (parsed from metadata).


def run_baseline(
    rgb_bgr, prompt: str, *,
    model: str, timeout_s: float, max_retries: int,
) -> tuple[bool, int, float, str, int]:
    """Returns (ok, n_attempts, total_ms, error, n_detections)."""
    t0 = time.perf_counter()
    try:
        boxes, labels, elapsed_s, metadata = production_request_bboxes(
            rgb_bgr, prompt,
            model=model,
            max_retries=max_retries,
            timeout_s=timeout_s,
            stream=True,
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - t0) * 1000.0
        return False, 0, total_ms, f'{type(exc).__name__}: {exc}', 0
    total_ms = (time.perf_counter() - t0) * 1000.0
    n_attempts = len(metadata.get('attempts', []))
    err = metadata.get('error') or ''
    if err:
        return False, n_attempts, total_ms, err, len(boxes)
    return True, n_attempts, total_ms, '', len(boxes)


# ---------------------------------------------------------------------------
# Image discovery (same as vlm_provider_bench)


def discover_pairs(vision_log_root: Path, run_glob: str) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for run_dir in sorted(vision_log_root.glob(run_glob)):
        if not run_dir.is_dir() or run_dir.name.startswith(('vlm_provider_bench_',
                                                             'vlm_timeout_isolation_',
                                                             'web_image_smoke_')):
            continue
        for orig in sorted(run_dir.glob('generalist_detection_node_vlm_sam_orig_*.jpg')):
            req = orig.with_name(orig.name.replace('_orig_', '_req_').replace('.jpg', '.json'))
            if not req.exists():
                continue
            try:
                meta = json.loads(req.read_text())
                prompt = meta.get('request', {}).get('prompt', '')
            except Exception:
                continue
            if prompt:
                pairs.append((orig, prompt))
    return pairs


# ---------------------------------------------------------------------------
# Aggregation


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float('nan')
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rows: list[CallResult]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    by: dict[str, list[CallResult]] = {}
    for r in rows:
        by.setdefault(r.policy, []).append(r)
    for policy, rs in by.items():
        ok = [r for r in rs if r.status == 'ok' and r.total_ms is not None]
        totals = [r.total_ms for r in ok]
        attempts = [r.n_attempts for r in rs]
        out[policy] = {
            'n_calls': len(rs),
            'n_ok': len(ok),
            'n_err': len(rs) - len(ok),
            'success_rate': len(ok) / max(1, len(rs)),
            'total_p50': percentile(totals, 0.50),
            'total_p90': percentile(totals, 0.90),
            'total_p99': percentile(totals, 0.99),
            'total_max': max(totals) if totals else float('nan'),
            'total_mean': statistics.fmean(totals) if totals else float('nan'),
            'mean_attempts': statistics.fmean(attempts) if attempts else float('nan'),
            'max_attempts': max(attempts) if attempts else 0,
        }
    return out


def write_summary_md(path: Path, stats: dict[str, dict], *, n_images: int, trials: int,
                     per_attempt_s: float, baseline_timeout_s: float,
                     overall_cap_s: float, max_attempts: int, model: str) -> None:
    rows = []
    rows.append('# vlm_bbox per-attempt timeout — isolation test\n\n')
    rows.append(f'- Sample: {n_images} images x {trials} trials per policy\n')
    rows.append(f'- Model: `{model}`\n')
    rows.append(f'- Baseline policy: timeout_s={baseline_timeout_s}, max_retries=3 '
                f'(production `vlm_bbox.request_bboxes`)\n')
    rows.append(f'- Capped policy:  per_attempt={per_attempt_s}s, max_attempts={max_attempts}, '
                f'overall_cap={overall_cap_s}s\n')
    rows.append(f'- Run: {datetime.now().isoformat(timespec="seconds")}\n\n')
    cols = list(stats.keys())
    rows.append('| metric | ' + ' | '.join(cols) + ' |\n')
    rows.append('|---' * (len(cols) + 1) + '|\n')
    metrics = [
        ('n_calls', '{}'),
        ('n_ok', '{}'),
        ('n_err', '{}'),
        ('success_rate', '{:.2%}'),
        ('total_p50', '{:.0f} ms'),
        ('total_p90', '{:.0f} ms'),
        ('total_p99', '{:.0f} ms'),
        ('total_max', '{:.0f} ms'),
        ('total_mean', '{:.0f} ms'),
        ('mean_attempts', '{:.2f}'),
        ('max_attempts', '{}'),
    ]
    for key, fmt in metrics:
        cells = []
        for c in cols:
            v = stats[c].get(key, '')
            try:
                cells.append(fmt.format(v))
            except (ValueError, TypeError):
                cells.append(str(v))
        rows.append(f'| {key} | ' + ' | '.join(cells) + ' |\n')
    path.write_text(''.join(rows))


# ---------------------------------------------------------------------------
# Main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=vision_flash_model())
    parser.add_argument('--n-images', type=int, default=10)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--per-attempt-s', type=float, default=10.0)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--overall-cap-s', type=float, default=30.0)
    parser.add_argument('--baseline-timeout-s', type=float, default=20.0)
    parser.add_argument('--vision-log-root', default='vision_log')
    parser.add_argument('--run-glob', default='*')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inter-call-sleep', type=float, default=0.2)
    parser.add_argument('--inject-latency-s', type=float, default=0.0,
                        help='If > 0, after the first content chunk in each '
                             'capped attempt, sleep this many seconds. '
                             'Used to deterministically force the per-attempt '
                             'cap to fire. Capped policy only.')
    parser.add_argument('--policies', default='baseline,capped',
                        help='Comma-separated list of policies to run.')
    args = parser.parse_args()
    selected_policies = tuple(p.strip() for p in args.policies.split(',') if p.strip())

    load_env()
    rng = random.Random(args.seed)

    vision_log_root = Path(args.vision_log_root).resolve()
    pairs = discover_pairs(vision_log_root, args.run_glob)
    if not pairs:
        print(f'no orig+req pairs found under {vision_log_root}', file=sys.stderr)
        return 2
    n = min(args.n_images, len(pairs))
    sample = rng.sample(pairs, n)
    print(f'discovered {len(pairs)} pairs; sampling {n} (seed={args.seed})')

    # Pre-decode each image once.
    cache = []
    for img_path, prompt in sample:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f'  skip (cv2.imread failed): {img_path.name}')
            continue
        cache.append((img_path, prompt, bgr))
    if not cache:
        print('no images decoded', file=sys.stderr)
        return 2

    out_root = vision_log_root / f'vlm_timeout_isolation_{datetime.now():%Y%m%d_%H%M%S}'
    out_root.mkdir(parents=True, exist_ok=False)
    print(f'writing results to {out_root}')

    schedule: list[tuple[str, tuple[Path, str, Any], int]] = []
    for trial in range(args.trials):
        for entry in cache:
            for policy in selected_policies:
                schedule.append((policy, entry, trial))
    rng.shuffle(schedule)
    print(f'running {len(schedule)} timed calls (shuffled)')

    results: list[CallResult] = []
    csv_path = out_root / 'summary.csv'
    csv_fields = list(CallResult(policy='', image='', prompt='', trial=0).csv_row().keys())
    with csv_path.open('w', newline='') as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        writer.writeheader()
        for i, (policy, (img_path, prompt, bgr), trial) in enumerate(schedule, 1):
            r = CallResult(policy=policy, image=img_path.name, prompt=prompt, trial=trial)
            try:
                if policy == 'baseline':
                    ok, n_attempts, total_ms, err, n_dets = run_baseline(
                        bgr, prompt,
                        model=args.model,
                        timeout_s=args.baseline_timeout_s,
                        max_retries=args.max_attempts,
                    )
                    r.status = 'ok' if ok else 'error'
                    r.n_attempts = n_attempts
                    r.total_ms = total_ms
                    r.successful_attempt_ms = total_ms if ok else None
                    r.error = err
                    r.n_detections = n_dets if ok else None
                else:
                    ok, raw, n_attempts, total_ms, attempt_ms, err = request_bboxes_capped(
                        bgr, prompt,
                        model=args.model,
                        per_attempt_timeout_s=args.per_attempt_s,
                        max_attempts=args.max_attempts,
                        overall_cap_s=args.overall_cap_s,
                        inject_latency_s=args.inject_latency_s,
                    )
                    r.status = 'ok' if ok else 'error'
                    r.n_attempts = n_attempts
                    r.total_ms = total_ms
                    r.successful_attempt_ms = attempt_ms
                    r.error = err
                    if ok:
                        try:
                            payload = json.loads(raw or '{}')
                            dets = payload.get('detections')
                            if isinstance(dets, list):
                                r.n_detections = len(dets)
                        except Exception:
                            pass
            except Exception as exc:
                r.error = f'harness {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}'

            results.append(r)
            writer.writerow(r.csv_row())
            csv_fh.flush()
            detail_path = out_root / f'{policy}_{img_path.stem}_t{trial}.json'
            detail_path.write_text(json.dumps(asdict(r), indent=2))

            t = f'{r.total_ms:.0f}' if r.total_ms is not None else '----'
            print(f'  [{i:>3}/{len(schedule)}] {policy:>9} {img_path.stem[-22:]:>22} '
                  f't={trial}  total={t:>6} ms  attempts={r.n_attempts}  '
                  f'dets={r.n_detections}  {"OK" if r.status == "ok" else "ERR"}'
                  f'{("  " + r.error[:80]) if r.error else ""}')
            time.sleep(args.inter_call_sleep)

    stats = summarize(results)
    write_summary_md(out_root / 'summary.md', stats,
                     n_images=n, trials=args.trials,
                     per_attempt_s=args.per_attempt_s,
                     baseline_timeout_s=args.baseline_timeout_s,
                     overall_cap_s=args.overall_cap_s,
                     max_attempts=args.max_attempts,
                     model=args.model)
    print('\n' + (out_root / 'summary.md').read_text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
