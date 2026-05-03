#!/usr/bin/env python3
"""Compare OpenRouter vs direct Gemini latency on the generalist VLM bbox path.

Replays saved images from `vision_log/` (the orig + req JSON pairs produced by
`generalist_detection_node` when running with `enable_vlm=True` /
detection_source=`vlm_sam`) against two providers using the same image bytes,
the same system prompt, and the same response schema. Captures TTFT (first
non-empty content chunk) and total stream-completion time per call, then
aggregates a side-by-side report.

Production runtime stays on OpenRouter — this is a read-only benchmark.

Usage (from workspace root):

    pip install --user google-genai           # one-time
    python3 src/tk26_vision/scripts/tests/manual/vlm_provider_bench.py \\
        --n-images 10 --trials 3

Env vars (loaded from .env at workspace root):
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL  - openrouter route
    GEMINI_API_KEY                           - direct gemini route

Outputs land in `vision_log/vlm_provider_bench_<YYYYmmdd_HHMMSS>/`:
    summary.csv     one row per call
    summary.md      human-readable side-by-side table
    <provider>_<imgstem>_t<idx>.json   per-call detail
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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import cv2

# Make tk26_vision packages importable when running from workspace root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
for _src_dir in (
    _REPO_ROOT / 'src' / 'tk26_vision' / 'src' / 'kimi_api',
    _REPO_ROOT / 'src' / 'tk26_vision' / 'src' / 'object_detection_generalist',
):
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

from kimi_api._env import (  # noqa: E402
    base_url as openrouter_base_url,
    gemini_api_key,
    load_env,
    require_api_key as openrouter_api_key,
)
from kimi_api._image_utils import encode_to_data_url  # noqa: E402
from object_detection_generalist.vlm_bbox import (  # noqa: E402
    _GEMINI_SYSTEM_PROMPT,
    _RESPONSE_SCHEMA,
)


# ---------------------------------------------------------------------------
# Provider call result


@dataclass
class CallResult:
    provider: str
    model: str
    image: str
    prompt: str
    trial: int
    status: str               # 'ok' | 'error'
    error: str = ''
    ttft_ms: float | None = None
    total_ms: float | None = None
    n_chunks: int = 0
    n_detections: int | None = None
    output_text: str = ''      # truncated for the per-call json on disk

    def csv_row(self) -> dict[str, Any]:
        return {
            'provider': self.provider,
            'model': self.model,
            'image': self.image,
            'prompt': self.prompt,
            'trial': self.trial,
            'status': self.status,
            'ttft_ms': '' if self.ttft_ms is None else f'{self.ttft_ms:.1f}',
            'total_ms': '' if self.total_ms is None else f'{self.total_ms:.1f}',
            'n_chunks': self.n_chunks,
            'n_detections': '' if self.n_detections is None else self.n_detections,
            'error': self.error.replace('\n', ' \\n ')[:400],
        }


# ---------------------------------------------------------------------------
# Provider implementations
#
# Both call the streaming endpoint, advance TTFT only on the first chunk that
# carries non-empty text content, and return total wall-clock at iterator
# exhaustion. No retries, no fallbacks - we measure one round trip.


class OpenRouterProvider:
    name = 'openrouter'

    def __init__(self, model: str, timeout_s: float):
        from openai import OpenAI
        self.model = model
        self.timeout_s = timeout_s
        self.client = OpenAI(
            api_key=openrouter_api_key(),
            base_url=openrouter_base_url(),
            max_retries=0,
            timeout=timeout_s,
        )

    def call(self, jpeg_bytes: bytes, prompt: str, image_shape: tuple[int, int]) -> CallResult:
        import base64
        h, w = image_shape
        data_url = f'data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode()}'
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
        result = CallResult(
            provider=self.name, model=self.model, image='', prompt=prompt,
            trial=0, status='error',
        )
        t_send = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=0,
                stream=True,
            )
            try:
                for chunk in stream:
                    result.n_chunks += 1
                    err = getattr(chunk, 'error', None)
                    if err:
                        raise RuntimeError(f'openrouter stream error: {err}')
                    choices = getattr(chunk, 'choices', None) or []
                    if not choices:
                        continue
                    delta = getattr(choices[0], 'delta', None)
                    piece = getattr(delta, 'content', None) if delta else None
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t_send
                        parts.append(piece)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass
            total = time.perf_counter() - t_send
            result.ttft_ms = None if ttft is None else ttft * 1000.0
            result.total_ms = total * 1000.0
            result.output_text = ''.join(parts)
            result.status = 'ok' if ttft is not None else 'error'
            if ttft is None:
                result.error = 'stream produced no content chunks'
        except Exception as exc:
            result.error = f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}'
        return result


class DirectGeminiProvider:
    name = 'gemini'

    def __init__(self, model: str, timeout_s: float):
        from google import genai
        from google.genai import types as genai_types
        self.model = model
        self.timeout_s = timeout_s
        self.client = genai.Client(api_key=gemini_api_key())
        self._types = genai_types
        # Strip OpenAI-strict-only knobs that google-genai doesn't accept.
        self._schema = _gemini_compatible_schema(_RESPONSE_SCHEMA)

    def call(self, jpeg_bytes: bytes, prompt: str, image_shape: tuple[int, int]) -> CallResult:
        types = self._types
        h, w = image_shape
        user_text = f'Target classes: {prompt}. Original image size: width={w}, height={h}.'
        contents = [
            types.Content(role='user', parts=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type='image/jpeg'),
                types.Part.from_text(text=user_text),
            ]),
        ]
        config = types.GenerateContentConfig(
            system_instruction=_GEMINI_SYSTEM_PROMPT,
            response_mime_type='application/json',
            response_schema=self._schema,
            temperature=0.0,
        )
        result = CallResult(
            provider=self.name, model=self.model, image='', prompt=prompt,
            trial=0, status='error',
        )
        t_send = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                result.n_chunks += 1
                if (time.perf_counter() - t_send) > self.timeout_s:
                    raise TimeoutError(f'gemini stream exceeded {self.timeout_s}s')
                piece = getattr(chunk, 'text', None)
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t_send
                    parts.append(piece)
            total = time.perf_counter() - t_send
            result.ttft_ms = None if ttft is None else ttft * 1000.0
            result.total_ms = total * 1000.0
            result.output_text = ''.join(parts)
            result.status = 'ok' if ttft is not None else 'error'
            if ttft is None:
                result.error = 'stream produced no content chunks'
        except Exception as exc:
            result.error = f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}'
        return result


def _gemini_compatible_schema(schema: dict) -> dict:
    """google-genai rejects 'additionalProperties' / 'strict' style fields.

    Recursively drops them so the same shared schema works for both providers.
    """
    if isinstance(schema, dict):
        return {
            k: _gemini_compatible_schema(v)
            for k, v in schema.items()
            if k not in ('additionalProperties', 'strict')
        }
    if isinstance(schema, list):
        return [_gemini_compatible_schema(v) for v in schema]
    return schema


# ---------------------------------------------------------------------------
# Image discovery


def discover_pairs(vision_log_root: Path, run_glob: str) -> list[tuple[Path, str]]:
    """Yield (orig_jpg_path, prompt) pairs from generalist vlm_sam runs."""
    pairs: list[tuple[Path, str]] = []
    for run_dir in sorted(vision_log_root.glob(run_glob)):
        if not run_dir.is_dir() or run_dir.name.startswith('vlm_provider_bench_'):
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
            if not prompt:
                continue
            pairs.append((orig, prompt))
    return pairs


# ---------------------------------------------------------------------------
# Aggregation + reporting


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float('nan')
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(results: list[CallResult]) -> dict[str, dict]:
    """Per-provider stats over successful calls."""
    out: dict[str, dict] = {}
    by_provider: dict[str, list[CallResult]] = {}
    for r in results:
        by_provider.setdefault(r.provider, []).append(r)
    for provider, rs in by_provider.items():
        ok = [r for r in rs if r.status == 'ok' and r.ttft_ms is not None and r.total_ms is not None]
        ttfts = [r.ttft_ms for r in ok]
        totals = [r.total_ms for r in ok]
        out[provider] = {
            'model': rs[0].model if rs else '',
            'n_calls': len(rs),
            'n_ok': len(ok),
            'n_err': len(rs) - len(ok),
            'ttft_p50': percentile(ttfts, 0.50),
            'ttft_p90': percentile(ttfts, 0.90),
            'ttft_p99': percentile(ttfts, 0.99),
            'ttft_mean': statistics.fmean(ttfts) if ttfts else float('nan'),
            'total_p50': percentile(totals, 0.50),
            'total_p90': percentile(totals, 0.90),
            'total_p99': percentile(totals, 0.99),
            'total_mean': statistics.fmean(totals) if totals else float('nan'),
            'mean_chunks': statistics.fmean([r.n_chunks for r in ok]) if ok else float('nan'),
            'mean_output_bytes': statistics.fmean([len(r.output_text) for r in ok]) if ok else float('nan'),
        }
    return out


def write_summary_md(path: Path, stats: dict[str, dict], n_images: int, n_trials: int) -> None:
    rows = []
    rows.append('# OpenRouter vs Direct Gemini — generalist VLM bbox latency\n')
    rows.append(f'- Sample: {n_images} images x {n_trials} trials per provider\n')
    rows.append(f'- Run: {datetime.now().isoformat(timespec="seconds")}\n\n')
    rows.append('| metric | ' + ' | '.join(stats.keys()) + ' |\n')
    rows.append('|---' * (len(stats) + 1) + '|\n')
    metrics = [
        ('model', '{}'),
        ('n_calls', '{}'),
        ('n_ok', '{}'),
        ('n_err', '{}'),
        ('ttft_p50', '{:.0f} ms'),
        ('ttft_p90', '{:.0f} ms'),
        ('ttft_p99', '{:.0f} ms'),
        ('ttft_mean', '{:.0f} ms'),
        ('total_p50', '{:.0f} ms'),
        ('total_p90', '{:.0f} ms'),
        ('total_p99', '{:.0f} ms'),
        ('total_mean', '{:.0f} ms'),
        ('mean_chunks', '{:.1f}'),
        ('mean_output_bytes', '{:.0f}'),
    ]
    for key, fmt in metrics:
        cells = []
        for prov in stats:
            v = stats[prov].get(key, '')
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
    parser.add_argument('--openrouter-model', default='google/gemini-2.5-flash')
    parser.add_argument('--gemini-model', default='gemini-2.5-flash')
    parser.add_argument('--n-images', type=int, default=10)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--vision-log-root', default='vision_log')
    parser.add_argument('--run-glob', default='*')
    parser.add_argument('--timeout-s', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inter-call-sleep', type=float, default=0.2,
                        help='Seconds to sleep between successive calls.')
    args = parser.parse_args()

    load_env()

    rng = random.Random(args.seed)

    vision_log_root = Path(args.vision_log_root).resolve()
    if not vision_log_root.is_dir():
        print(f'error: vision-log-root does not exist: {vision_log_root}', file=sys.stderr)
        return 2

    all_pairs = discover_pairs(vision_log_root, args.run_glob)
    if not all_pairs:
        print(f'error: no generalist vlm_sam orig+req pairs found under '
              f'{vision_log_root} matching run-glob={args.run_glob!r}',
              file=sys.stderr)
        return 2
    print(f'discovered {len(all_pairs)} (orig, prompt) pairs in {vision_log_root}')

    n = min(args.n_images, len(all_pairs))
    sample = rng.sample(all_pairs, n)
    print(f'sampling {n} pairs (seed={args.seed})')

    # Pre-encode each image once and cache JPEG bytes; use the same bytes for both providers.
    cache: list[tuple[Path, str, bytes, tuple[int, int]]] = []
    for img_path, prompt in sample:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f'  skip (cv2.imread failed): {img_path.name}')
            continue
        ok, buf = cv2.imencode('.jpg', bgr)
        if not ok:
            print(f'  skip (jpeg encode failed): {img_path.name}')
            continue
        cache.append((img_path, prompt, buf.tobytes(), bgr.shape[:2]))
    if not cache:
        print('error: no images survived encoding', file=sys.stderr)
        return 2

    out_root = vision_log_root / f'vlm_provider_bench_{datetime.now():%Y%m%d_%H%M%S}'
    out_root.mkdir(parents=True, exist_ok=False)
    print(f'writing results to {out_root}')

    # Build providers (lazy imports already inside __init__).
    print(f'init OpenRouterProvider model={args.openrouter_model}')
    op = OpenRouterProvider(args.openrouter_model, args.timeout_s)
    print(f'init DirectGeminiProvider model={args.gemini_model}')
    gp = DirectGeminiProvider(args.gemini_model, args.timeout_s)
    providers = [op, gp]

    # Single warm-up call per provider on the first image.
    warm_path, warm_prompt, warm_bytes, warm_shape = cache[0]
    for p in providers:
        print(f'  warmup {p.name}...', end=' ', flush=True)
        wr = p.call(warm_bytes, warm_prompt, warm_shape)
        print(f'{wr.status} ({wr.total_ms:.0f} ms)' if wr.total_ms else wr.status,
              f'err={wr.error[:120]}' if wr.error else '')
        time.sleep(args.inter_call_sleep)

    # Build the trial schedule, shuffled.
    schedule = []
    for trial in range(args.trials):
        for entry in cache:
            for prov in providers:
                schedule.append((prov, entry, trial))
    rng.shuffle(schedule)
    print(f'running {len(schedule)} timed calls (shuffled)')

    results: list[CallResult] = []
    csv_path = out_root / 'summary.csv'
    csv_fields = list(CallResult(provider='', model='', image='', prompt='', trial=0,
                                  status='').csv_row().keys())
    with csv_path.open('w', newline='') as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        writer.writeheader()
        for i, (prov, (img_path, prompt, jpeg, shape), trial) in enumerate(schedule, 1):
            r = prov.call(jpeg, prompt, shape)
            r.image = img_path.name
            r.trial = trial
            # Lightly parse n_detections from output for a sanity column.
            try:
                payload = json.loads(r.output_text or '{}')
                dets = payload.get('detections')
                if isinstance(dets, list):
                    r.n_detections = len(dets)
            except Exception:
                pass
            results.append(r)
            writer.writerow(r.csv_row())
            csv_fh.flush()

            # Per-call json detail (truncate output to keep files small).
            detail = asdict(r)
            detail['output_text'] = r.output_text[:4096]
            detail_path = out_root / f'{r.provider}_{img_path.stem}_t{trial}.json'
            detail_path.write_text(json.dumps(detail, indent=2))

            t_str = f'{r.total_ms:.0f}/{r.ttft_ms:.0f}' if r.total_ms is not None and r.ttft_ms is not None else '----'
            print(f'  [{i:>3}/{len(schedule)}] {r.provider:>10} {img_path.stem[-22:]:>22} '
                  f't={trial} total/ttft={t_str:>12} ms  '
                  f'chunks={r.n_chunks:>3}  dets={r.n_detections}  '
                  f'{"OK" if r.status == "ok" else "ERR"}'
                  f'{("  " + r.error[:80]) if r.error else ""}')
            time.sleep(args.inter_call_sleep)

    stats = summarize(results)
    write_summary_md(out_root / 'summary.md', stats, n, args.trials)
    print('\n' + (out_root / 'summary.md').read_text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
