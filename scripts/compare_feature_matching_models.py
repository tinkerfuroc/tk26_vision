#!/usr/bin/env python3
"""Offline Gemini Pro vs Flash comparison for feature_matching service.

Replays logged feature-matching cases (vision_log) through both models via
OpenRouter and compares accuracy, latency, and parse reliability.

Usage:
    source src/tk26_vision/.venv-vision-main/bin/activate
    python3 src/tk26_vision/scripts/compare_feature_matching_models.py [-n 3]

Requires OPENROUTER_API_KEY in env or .env at repo root.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_VISION_LOG = WORKSPACE_ROOT / 'vision_log'

# vision_util source isn't on sys.path until colcon-installed, but running
# this straight from the repo is convenient — splice it in ahead of time.
sys.path.insert(0, str(SCRIPT_DIR.parent / "src" / "vision_util"))
from vision_util.vlm_models import vision_vlm_model, vision_flash_model  # noqa: E402

PRO_MODEL = vision_vlm_model()
FLASH_MODEL = vision_flash_model()
DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT_S = 30.0

# Selected cases: 1 STRONG, 2 MEDIUM, 2 WEAK.
SELECTED_CASES = {
    'feature_matching_service_1777726993_feature_matching_req_20260502_210727_666.json': 'STRONG',
    'feature_matching_service_1777714065_feature_matching_req_20260502_173724_004.json': 'MEDIUM',
    'feature_matching_service_1777714904_feature_matching_req_20260502_174844_813.json': 'MEDIUM',
    'feature_matching_service_1777635721_feature_matching_req_20260501_200049_440.json': 'WEAK',
    'feature_matching_service_1777727590_feature_matching_req_20260502_211633_250.json': 'WEAK',
}


def encode_to_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode('.jpg', img)
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return f'data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode("utf-8")}'


def _patch_result(raw, n_targets: int, n_cand: int):
    """Copy of kimi_api/_match_vlm.py:patch_result (moved there from
    feature_matching.py when the provider-fallback chain was added)."""
    if not isinstance(raw, list):
        return None, f'not a list: {raw!r}'
    if len(raw) == 0 and n_targets > 0:
        return None, 'empty list'
    patched = []
    for i in range(n_targets):
        v = raw[i] if i < len(raw) else None
        if isinstance(v, bool):
            v = i % n_cand
        elif isinstance(v, int):
            pass
        elif v is None:
            v = i % n_cand
        else:
            try:
                v = int(v)
            except (TypeError, ValueError):
                v = i % n_cand
        if v < 0 or v >= n_cand:
            v = i % n_cand
        patched.append(v)
    return patched, ''


@dataclass
class TestCase:
    name: str
    quality: str
    json_path: Path
    orig_path: Path
    ref_paths: list[Path]
    detections: list[dict]
    features_text: list[str]
    n_features: int
    n_candidates: int
    ground_truth: list[int]


@dataclass
class TrialResult:
    test_name: str
    quality: str
    model: str
    repeat: int
    latency_s: float
    parse_ok: bool
    structurally_valid: bool
    raw_matches: Optional[list]
    patched_matches: Optional[list]
    exact_match: bool
    per_ref_correct: list[bool] = field(default_factory=list)
    accuracy: float = 0.0
    raw_response: str = ''
    error: str = ''


def _filename_root(json_path: Path) -> tuple[str, str]:
    """Return (tag, ts) from `<tag>_feature_matching_req_<ts>.json`."""
    stem = json_path.stem
    marker = '_feature_matching_req_'
    idx = stem.index(marker)
    tag = stem[:idx]
    ts = stem[idx + len(marker):]
    return tag, ts


def discover_test_cases(base: Path) -> list[TestCase]:
    cases: list[TestCase] = []
    for json_path in base.rglob('*feature_matching*req*.json'):
        if json_path.name not in SELECTED_CASES:
            continue
        try:
            with json_path.open() as f:
                data = json.load(f)
        except Exception as e:
            print(f'WARN: skip {json_path.name}: {e}', file=sys.stderr)
            continue
        if data.get('vlm_status') != 0:
            continue

        tag, ts = _filename_root(json_path)
        parent = json_path.parent
        orig_path = parent / f'{tag}_feature_matching_orig_{ts}.jpg'
        n_refs = data['request']['n_references']
        ref_paths = [parent / f'{tag}_feature_matching_ref{i}_{ts}.jpg' for i in range(n_refs)]

        if not orig_path.exists():
            print(f'WARN: orig missing for {json_path.name}', file=sys.stderr)
            continue
        missing_refs = [p for p in ref_paths if not p.exists()]
        if missing_refs:
            print(f'WARN: refs missing for {json_path.name}: {missing_refs}', file=sys.stderr)
            continue

        gt_map = {m['ref']: m['cand'] for m in data['matches']}
        ground_truth = [gt_map[i] for i in range(data['request']['n_features'])]

        cases.append(TestCase(
            name=f'{parent.name}/{ts}',
            quality=SELECTED_CASES[json_path.name],
            json_path=json_path,
            orig_path=orig_path,
            ref_paths=ref_paths,
            detections=data['detections'],
            features_text=data['features_text'],
            n_features=data['request']['n_features'],
            n_candidates=data['request']['n_candidates'],
            ground_truth=ground_truth,
        ))
    cases.sort(key=lambda c: ('SWMSTRONG'.find(c.quality[0]), c.name))
    return cases


def extract_candidate_crops(orig: np.ndarray, detections: list[dict]) -> list[np.ndarray]:
    crops = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        crop = orig[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Empty crop for bbox {det['bbox']}, orig shape {orig.shape}")
        crops.append(crop)
    return crops


def build_messages(tc: TestCase, cand_crops: list[np.ndarray], ref_imgs: list[np.ndarray]) -> tuple[str, list[dict]]:
    """Replicate prompt from feature_matching.py:347-386 (image+text mode)."""
    n_feats = tc.n_features
    n_cand = tc.n_candidates

    sys_prompt = (
        f'You will be shown {n_feats} REFERENCE images of specific people, then '
        f'{n_cand} CANDIDATE crops taken from a wider scene. For each reference '
        f'(0..{n_feats - 1}), output the candidate index whose person is the SAME '
        'individual as the reference. Use clothing, hair color/length, body shape, '
        'and posture as evidence. The user may also provide a textual description '
        'per reference; treat it as a tiebreaker hint only. '
        f'Output ONLY a JSON list of length {n_feats}, e.g. "[0, 2, 1]". '
        'EVERY reference MUST be matched to a candidate. If you are uncertain, '
        'pick the candidate whose features (clothing, hair, body shape, description) '
        'are CLOSEST to the reference. NEVER use -1 or any negative number. '
        'Do not include explanations.'
    )

    reference_urls = [encode_to_data_url(img) for img in ref_imgs]
    candidate_urls = [encode_to_data_url(img) for img in cand_crops]

    user_content = []
    for i, ref_url in enumerate(reference_urls):
        user_content.append({'type': 'text', 'text': f'Reference {i}:'})
        user_content.append({'type': 'image_url', 'image_url': {'url': ref_url}})
    for j, cand_url in enumerate(candidate_urls):
        user_content.append({'type': 'text', 'text': f'Candidate {j}:'})
        user_content.append({'type': 'image_url', 'image_url': {'url': cand_url}})

    text_tail = 'Textual hints per reference:\n'
    for i, feat in enumerate(tc.features_text):
        text_tail += f'- Reference {i}: {feat or "(none)"}\n'
    text_tail += (
        f'Now output the JSON list of length {n_feats} mapping each reference '
        'to the matching candidate index.'
    )
    user_content.append({'type': 'text', 'text': text_tail})

    return sys_prompt, user_content


def call_model(
    client: OpenAI,
    model: str,
    sys_prompt: str,
    user_content: list[dict],
    timeout_s: float,
    tc: TestCase,
    repeat: int,
) -> TrialResult:
    res = TrialResult(
        test_name=tc.name,
        quality=tc.quality,
        model=model,
        repeat=repeat,
        latency_s=0.0,
        parse_ok=False,
        structurally_valid=False,
        raw_matches=None,
        patched_matches=None,
        exact_match=False,
    )

    t0 = time.perf_counter()
    try:
        completion = client.with_options(timeout=timeout_s).chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': user_content},
            ],
        )
    except Exception as e:
        res.latency_s = time.perf_counter() - t0
        res.error = f'API call failed: {e}'
        return res
    res.latency_s = time.perf_counter() - t0

    raw_text = completion.choices[0].message.content or ''
    res.raw_response = raw_text

    try:
        parsed = ast.literal_eval(raw_text.strip())
        res.parse_ok = True
        res.raw_matches = parsed if isinstance(parsed, list) else [parsed]
    except Exception as e:
        res.error = f'parse failed: {e}'
        return res

    patched, msg = _patch_result(parsed, tc.n_features, tc.n_candidates)
    if patched is None:
        res.error = f'unsalvageable: {msg}'
        return res
    res.structurally_valid = True
    res.patched_matches = patched
    res.per_ref_correct = [patched[i] == tc.ground_truth[i] for i in range(tc.n_features)]
    res.accuracy = sum(res.per_ref_correct) / tc.n_features
    res.exact_match = all(res.per_ref_correct)
    return res


def _print_per_case(results: list[TrialResult], tc: TestCase):
    print(f'\n=== {tc.name}  [{tc.quality}]  '
          f'(refs={tc.n_features}, cands={tc.n_candidates})')
    print(f'  Ground truth: {tc.ground_truth}')
    print(f'  {"Model":<20} {"Rep":<4} {"Latency":<10} {"Parse":<6} {"Valid":<6} '
          f'{"Matches":<18} {"Acc":<6}')
    for r in results:
        if r.test_name != tc.name:
            continue
        matches = str(r.patched_matches) if r.patched_matches is not None else 'FAIL'
        parse = 'OK' if r.parse_ok else 'X'
        valid = 'OK' if r.structurally_valid else 'X'
        err = f'  err: {r.error}' if r.error else ''
        print(f'  {r.model:<20} {r.repeat:<4} {r.latency_s:>7.2f}s  '
              f'{parse:<6} {valid:<6} {matches:<18} {r.accuracy:.3f}{err}')


def _consistency(results: list[TrialResult]) -> float:
    """Fraction of (case, model) groups where all repeats produced same patched_matches."""
    groups: dict[tuple, list[Optional[list]]] = {}
    for r in results:
        groups.setdefault((r.test_name, r.model), []).append(r.patched_matches)
    if not groups:
        return 0.0
    n_consistent = sum(1 for v in groups.values() if len(set(map(tuple, [m for m in v if m is not None]))) <= 1 and all(m is not None for m in v))
    return n_consistent / len(groups)


def _summary_for(results: list[TrialResult], model: str) -> dict:
    sub = [r for r in results if r.model == model]
    if not sub:
        return {}
    accs = [r.accuracy for r in sub if r.structurally_valid]
    lats = [r.latency_s for r in sub if r.error == '' or 'parse' in r.error]
    parse_ok = sum(1 for r in sub if r.parse_ok)
    valid = sum(1 for r in sub if r.structurally_valid)
    exact = sum(1 for r in sub if r.exact_match)

    # Per-model consistency
    groups: dict[str, list[Optional[list]]] = {}
    for r in sub:
        groups.setdefault(r.test_name, []).append(r.patched_matches)
    consistent = sum(
        1 for v in groups.values()
        if len(set(map(tuple, [m for m in v if m is not None]))) <= 1
        and all(m is not None for m in v)
    )
    consistency = consistent / len(groups) if groups else 0.0

    return {
        'avg_accuracy': statistics.mean(accs) if accs else 0.0,
        'exact_pct': exact / len(sub) * 100,
        'avg_latency_s': statistics.mean(lats) if lats else 0.0,
        'parse_pct': parse_ok / len(sub) * 100,
        'valid_pct': valid / len(sub) * 100,
        'consistency_pct': consistency * 100,
        'n_trials': len(sub),
    }


def print_summary(results: list[TrialResult]):
    print('\n=== OVERALL SUMMARY ===')
    print(f'  {"Model":<22} {"AvgAcc":<8} {"Exact%":<8} {"AvgLat":<10} '
          f'{"Parse%":<8} {"Valid%":<8} {"Consist%":<10} {"N":<5}')
    for model in (PRO_MODEL, FLASH_MODEL):
        s = _summary_for(results, model)
        if not s:
            continue
        print(f'  {model:<22} {s["avg_accuracy"]:<8.3f} {s["exact_pct"]:<8.1f} '
              f'{s["avg_latency_s"]:>7.2f}s   {s["parse_pct"]:<8.1f} '
              f'{s["valid_pct"]:<8.1f} {s["consistency_pct"]:<10.1f} {s["n_trials"]:<5}')

    # Per-quality breakdown
    print('\n=== PER-QUALITY ACCURACY ===')
    print(f'  {"Quality":<10} {"Model":<22} {"AvgAcc":<8} {"Exact%":<8} {"AvgLat":<10}')
    for quality in ('STRONG', 'MEDIUM', 'WEAK'):
        for model in (PRO_MODEL, FLASH_MODEL):
            sub = [r for r in results if r.model == model and r.quality == quality]
            if not sub:
                continue
            accs = [r.accuracy for r in sub if r.structurally_valid]
            lats = [r.latency_s for r in sub]
            exact = sum(1 for r in sub if r.exact_match)
            avg_acc = statistics.mean(accs) if accs else 0.0
            avg_lat = statistics.mean(lats) if lats else 0.0
            print(f'  {quality:<10} {model:<22} {avg_acc:<8.3f} '
                  f'{exact/len(sub)*100:<8.1f} {avg_lat:>7.2f}s')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-log', type=Path, default=DEFAULT_VISION_LOG)
    parser.add_argument('--repeats', '-n', type=int, default=DEFAULT_REPEATS)
    parser.add_argument('--timeout-s', type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument('--pro-model', default=PRO_MODEL)
    parser.add_argument('--flash-model', default=FLASH_MODEL)
    parser.add_argument('--json-output', type=Path, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print('ERROR: OPENROUTER_API_KEY not set. Export it or fill in .env.', file=sys.stderr)
        return 1
    base_url = os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    client = OpenAI(api_key=api_key, base_url=base_url)

    cases = discover_test_cases(args.vision_log)
    if not cases:
        print(f'ERROR: no cases discovered under {args.vision_log}', file=sys.stderr)
        return 1
    print(f'Discovered {len(cases)} test cases:')
    for c in cases:
        print(f'  [{c.quality:<6}] {c.name}  refs={c.n_features} cands={c.n_candidates} GT={c.ground_truth}')

    all_results: list[TrialResult] = []
    models = [args.pro_model, args.flash_model]

    for tc in cases:
        orig = cv2.imread(str(tc.orig_path), cv2.IMREAD_COLOR)
        ref_imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in tc.ref_paths]
        if orig is None or any(r is None for r in ref_imgs):
            print(f'ERROR: failed to load images for {tc.name}', file=sys.stderr)
            continue
        try:
            cand_crops = extract_candidate_crops(orig, tc.detections)
        except Exception as e:
            print(f'ERROR: crop extraction failed for {tc.name}: {e}', file=sys.stderr)
            continue

        sys_prompt, user_content = build_messages(tc, cand_crops, ref_imgs)
        case_results: list[TrialResult] = []

        for model in models:
            for rep in range(1, args.repeats + 1):
                print(f'  -> {tc.name} | {model} | rep {rep}/{args.repeats} ...', flush=True)
                r = call_model(client, model, sys_prompt, user_content,
                               args.timeout_s, tc, rep)
                if args.verbose:
                    print(f'     raw: {r.raw_response[:200]}')
                    if r.error:
                        print(f'     err: {r.error}')
                case_results.append(r)
                all_results.append(r)
                time.sleep(0.5)

        _print_per_case(case_results, tc)

    print_summary(all_results)

    if args.json_output:
        with args.json_output.open('w') as f:
            json.dump(
                {
                    'cases': [
                        {
                            'name': c.name,
                            'quality': c.quality,
                            'n_features': c.n_features,
                            'n_candidates': c.n_candidates,
                            'ground_truth': c.ground_truth,
                            'features_text': c.features_text,
                        }
                        for c in cases
                    ],
                    'trials': [asdict(r) for r in all_results],
                },
                f,
                indent=2,
                default=str,
            )
        print(f'\nResults written to {args.json_output}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
