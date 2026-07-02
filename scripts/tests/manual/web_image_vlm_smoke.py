#!/usr/bin/env python3
"""Web-image VLM smoke test for kimi_api / object_detection_generalist.

Standalone test harness — does NOT require a running ROS stack. Downloads a
small set of public-domain images (COCO val2017), exercises every VLM call
path in the workspace against them, and exercises the post-fix retry/timeout
defenses with mocked clients.

Live tests hit OpenRouter and require ``OPENROUTER_API_KEY``. This harness
fails fast if the key is missing — set it in your shell or via the same
``.env`` file the kimi_api ROS nodes use.

Logging
-------
The harness ALWAYS writes per-test logs and overlay images to a timestamped
subdirectory of the log root (default ``./vision_log/`` relative to CWD,
matching the convention used by the production tk26 vision nodes):

    vision_log/web_image_smoke_<YYYYmmdd_HHMMSS>/
        run.log                # stdout/stderr tee
        summary.txt            # PASS/FAIL table
        summary.json           # machine-readable summary
        <test_name>/
            req.json           # prompt, model, params, input refs
            resp.json          # raw VLM response (or error trace)
            orig.jpg           # source image (when applicable)
            overlay.jpg        # annotated overlay (bboxes/points/crops)

Usage::

    # From workspace root, after sourcing the venv + ROS:
    source src/tk26_vision/.venv-vision-main/bin/activate
    source install/setup.bash
    python3 src/tk26_vision/scripts/tests/manual/web_image_vlm_smoke.py [opts]

Options::

    --only NAME[,NAME...]    Run only the named tests
    --model MODEL            Override LLM model (default: env LLM_MODEL or
                             google/gemini-2.5-flash for cheap)
    --timeout-s SECONDS      Per-attempt VLM timeout (default 30.0)
    --max-retries N          Retries per call (default 3)
    --cache-dir PATH         Image cache directory
    --log-dir PATH           Log root (default ./vision_log/)
    --verbose                Print full responses to stdout (logged either way)

Test names: ``feature_extraction``, ``feature_matching``, ``vlm_bbox``,
``seat_recommend``, ``grocery_categorize``, ``parse_recovery``,
``parse_exhaustion``, ``timeout_fires``.

Exit code 0 on all-pass, 1 on any failure. Suitable for CI gating.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# kimi_api / object_detection_generalist imports (require ROS sourced)
# ---------------------------------------------------------------------------

try:
    from kimi_api._env import base_url, default_model, load_env, require_api_key
    from kimi_api._image_utils import encode_to_data_url
    from object_detection_generalist.vlm_bbox import request_bboxes
    from kimi_api._seat_vlm import request_seat
except ImportError as exc:
    print(
        f'[FATAL] failed to import workspace packages: {exc}\n'
        '  did you `source install/setup.bash` and activate the venv at '
        'src/tk26_vision/.venv-vision-main/?',
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Image manifest — public-domain COCO val2017 images
# ---------------------------------------------------------------------------

DEFAULT_CACHE = Path(__file__).parent / 'fixtures' / 'web_cache'

IMAGE_MANIFEST: Dict[str, str] = {
    # Manually verified content (each image has been read in and inspected
    # before commit — see the docstring of each test for which image it uses
    # and why):

    # Child face-on, drinking from a bottle — clear single human subject.
    # Used by feature_extraction and as ref 0 of feature_matching.
    'child_portrait.jpg':   'https://images.cocodataset.org/val2017/000000532058.jpg',

    # Adult chef seen from behind in a period kitchen, white apron + dark
    # pants. Visually distinct from the child. Used as ref 1 of feature_matching.
    'kitchen_chef.jpg':     'https://images.cocodataset.org/val2017/000000397133.jpg',

    # Three pedestrians on a New York sidewalk (woman walking left, man with
    # cart center, man walking right). Used as the candidates scene in
    # feature_matching — expected outcome: neither child nor chef is present
    # so result should be [-1, -1].
    'sidewalk_people.jpg':  'https://images.cocodataset.org/val2017/000000252219.jpg',

    # Empty domestic kitchen with a fruit basket (oranges + bananas) on the
    # foreground dining table, plus upper cabinets in the back. Used by
    # vlm_bbox (prompt: "banana") and as both shelf+object input for
    # grocery_categorize (cabinets stand in for shelf layers).
    'kitchen_fruit.jpg':    'https://images.cocodataset.org/val2017/000000037777.jpg',

    # Living room / dining nook with multiple wooden dining chairs around a
    # table, a TV unit on the left, and one person in the far doorway. Used
    # by seat_recommend.
    'dining_living.jpg':    'https://images.cocodataset.org/val2017/000000000139.jpg',
}


def _download_with_cache(name: str, cache_dir: Path, verbose: bool = False,
                          max_attempts: int = 4) -> Path:
    """Download (if missing) and return the local path. Retries on transient
    HTTP errors with exponential backoff."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        if verbose:
            print(f'  cache hit: {dest}')
        return dest

    url = IMAGE_MANIFEST[name]
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f'  downloading {url} -> {dest} (attempt {attempt}/{max_attempts})')
            req = urllib.request.Request(url,
                                         headers={'User-Agent': 'tk26_vision-smoke/1.0'})
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError('empty body')
            dest.write_bytes(data)
            if verbose:
                print(f'  {len(data)} bytes, sha256={hashlib.sha256(data).hexdigest()[:12]}…')
            return dest
        except Exception as exc:
            last_err = exc
            print(f'  download failed: {type(exc).__name__}: {exc}')
            if attempt < max_attempts:
                time.sleep(0.5 * (2 ** (attempt - 1)))
    raise RuntimeError(f'failed to download {url} after {max_attempts} attempts: {last_err}')


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'cv2.imread returned None for {path}')
    return img


# ---------------------------------------------------------------------------
# stdout/stderr tee — writes everything to run.log AND the terminal
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


# ---------------------------------------------------------------------------
# TestLogger — owns the per-run directory and per-test artifacts
# ---------------------------------------------------------------------------

class TestLogger:
    """Writes per-test req/resp/orig/overlay artifacts under a timestamped run dir."""

    def __init__(self, log_root: Path):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = log_root / f'web_image_smoke_{ts}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._test_dirs: Dict[str, Path] = {}
        self._artifacts: Dict[str, Dict[str, Any]] = {}

    def test_dir(self, name: str) -> Path:
        d = self._test_dirs.get(name)
        if d is None:
            d = self.run_dir / name
            d.mkdir(parents=True, exist_ok=True)
            self._test_dirs[name] = d
            self._artifacts[name] = {}
        return d

    def write_request(self, name: str, request: Dict[str, Any]) -> None:
        path = self.test_dir(name) / 'req.json'
        path.write_text(json.dumps(_jsonable(request), indent=2, default=str))
        self._artifacts[name].setdefault('files', []).append('req.json')

    def write_response(self, name: str, response: Dict[str, Any]) -> None:
        path = self.test_dir(name) / 'resp.json'
        path.write_text(json.dumps(_jsonable(response), indent=2, default=str))
        self._artifacts[name].setdefault('files', []).append('resp.json')

    def write_image(self, name: str, basename: str, img: np.ndarray) -> None:
        path = self.test_dir(name) / basename
        cv2.imwrite(str(path), img)
        self._artifacts[name].setdefault('files', []).append(basename)

    def attach_meta(self, name: str, **kw) -> None:
        self._artifacts.setdefault(name, {}).update(kw)

    def write_summary(self, results: List['TestResult']) -> None:
        summary_path = self.run_dir / 'summary.json'
        summary_path.write_text(json.dumps([{
            'name': r.name,
            'status': r.status,
            'elapsed_s': round(r.elapsed_s, 3),
            'notes': r.notes,
            'error': r.error,
            'artifacts': self._artifacts.get(r.name, {}),
        } for r in results], indent=2))

        txt_path = self.run_dir / 'summary.txt'
        with txt_path.open('w') as f:
            f.write(_format_summary(results))


def _jsonable(obj: Any) -> Any:
    """Best-effort JSON conversion. Numpy / Path / bytes turn into str."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return f'<ndarray shape={obj.shape} dtype={obj.dtype}>'
    if isinstance(obj, (bytes, bytearray)):
        return f'<bytes len={len(obj)}>'
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


# ---------------------------------------------------------------------------
# Overlay drawing helpers
# ---------------------------------------------------------------------------

def _draw_label(img: np.ndarray, text: str, xy: Tuple[int, int],
                color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    x, y = xy
    cv2.putText(img, text, (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def _draw_bboxes(img: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                 labels: Optional[List[str]] = None) -> np.ndarray:
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = labels[i] if labels else str(i)
        _draw_label(out, label, (x1, y1))
    return out


def _draw_point(img: np.ndarray, point_xy: Tuple[int, int],
                label: str = '') -> np.ndarray:
    out = img.copy()
    x, y = point_xy
    cv2.drawMarker(out, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    cv2.circle(out, (x, y), 14, (0, 0, 255), 2)
    if label:
        _draw_label(out, label, (x + 16, y), color=(0, 0, 255))
    return out


def _stack_horizontal(images: List[np.ndarray]) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        pad_h = h - im.shape[0]
        if pad_h > 0:
            im = cv2.copyMakeBorder(im, 0, pad_h, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(40, 40, 40))
        padded.append(im)
    return np.hstack(padded)


def _annotate_top(img: np.ndarray, text: str) -> np.ndarray:
    """Add a black band with text on top of the image."""
    h, w = img.shape[:2]
    band = np.zeros((30, w, 3), dtype=np.uint8)
    cv2.putText(band, text[:max(1, w // 8)], (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, img])


# ---------------------------------------------------------------------------
# Test result accounting
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    status: str   # 'PASS' | 'FAIL' | 'SKIP'
    elapsed_s: float
    notes: str = ''
    error: str = ''


def _run(name: str, fn: Callable[[], Tuple[bool, str]],
         *, results: List[TestResult], logger: TestLogger) -> None:
    t0 = time.perf_counter()
    try:
        ok, notes = fn()
        status = 'PASS' if ok else 'FAIL'
        results.append(TestResult(name, status, time.perf_counter() - t0, notes))
    except Exception as exc:
        tb = traceback.format_exc()
        # Persist the unhandled-exception trace alongside this test's logs.
        try:
            (logger.test_dir(name) / 'traceback.txt').write_text(tb)
        except Exception:
            pass
        results.append(TestResult(
            name, 'FAIL', time.perf_counter() - t0,
            notes='unhandled exception',
            error=f'{type(exc).__name__}: {exc}\n{tb}',
        ))


# ---------------------------------------------------------------------------
# Live VLM tests — replicate the production prompt construction in-process
# ---------------------------------------------------------------------------

# Free-text prompts pulled verbatim from feature_recognition.py:175-184
_FEATURE_EXTRACTION_SYS = (
    'You will be asked to extract features of one single designated person in an image,'
    ' including their gender, approximate age in years, facial features (hair length,'
    ' with or without glasses), hair color, and atleast two pieces of clothing (the more'
    ' the better, but no more than five). Output in the format of "[gender pronoun] is'
    ' [gender], [gender pronoun] are approximately [approximate age in years (give in'
    ' words, such as "twenty", not numeric numerals)] years-old, [gender pronoun] has'
    ' [hair color] hair and [facial features]. [gender pronoun] is wearing [clothing]",'
    ' do not include other information'
)


def _new_client(timeout_s: float):
    from openai import OpenAI
    return OpenAI(api_key=require_api_key(), base_url=base_url()).with_options(timeout=timeout_s)


def make_test_feature_extraction(args, logger: TestLogger) -> Callable[[], Tuple[bool, str]]:
    name = 'feature_extraction'

    def run() -> Tuple[bool, str]:
        img_path = _download_with_cache('child_portrait.jpg', args.cache_dir, args.verbose)
        img = _read_bgr(img_path)
        url = encode_to_data_url(img)

        logger.write_image(name, 'orig.jpg', img)
        logger.write_request(name, {
            'model': args.model,
            'system_prompt': _FEATURE_EXTRACTION_SYS,
            'user_text': 'extract the features of the person shown in the image.',
            'image_path': str(img_path),
            'image_shape': list(img.shape),
        })

        client = _new_client(args.timeout_s)
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {'role': 'system', 'content': _FEATURE_EXTRACTION_SYS},
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': url}},
                    {'type': 'text', 'text': 'extract the features of the person shown in the image.'},
                ]},
            ],
        )
        text = completion.choices[0].message.content or ''
        if args.verbose:
            print(f'    response: {text[:400]}')

        logger.write_response(name, {'content': text, 'model': args.model})
        logger.write_image(name, 'overlay.jpg', _annotate_top(img, f'feature: {text[:80]}'))

        if not text.strip():
            return False, 'empty response'
        return True, f'response {len(text)} chars'
    return run


def make_test_feature_matching(args, logger: TestLogger) -> Callable[[], Tuple[bool, str]]:
    name = 'feature_matching'

    def run() -> Tuple[bool, str]:
        ref_a_path = _download_with_cache('child_portrait.jpg', args.cache_dir, args.verbose)
        ref_b_path = _download_with_cache('kitchen_chef.jpg', args.cache_dir, args.verbose)
        scene_path = _download_with_cache('sidewalk_people.jpg', args.cache_dir, args.verbose)
        ref_a = _read_bgr(ref_a_path)
        ref_b = _read_bgr(ref_b_path)
        scene = _read_bgr(scene_path)

        # Naive split — left & right halves as two candidate crops.
        h, w = scene.shape[:2]
        cand_left = scene[:, : w // 2]
        cand_right = scene[:, w // 2 :]

        logger.write_image(name, 'ref_0.jpg', ref_a)
        logger.write_image(name, 'ref_1.jpg', ref_b)
        logger.write_image(name, 'scene_orig.jpg', scene)
        logger.write_image(name, 'cand_0.jpg', cand_left)
        logger.write_image(name, 'cand_1.jpg', cand_right)

        ref_urls = [encode_to_data_url(ref_a), encode_to_data_url(ref_b)]
        cand_urls = [encode_to_data_url(cand_left), encode_to_data_url(cand_right)]
        n_refs, n_cand = 2, 2

        sys_prompt = (
            f'You will be shown {n_refs} REFERENCE images of specific people, then '
            f'{n_cand} CANDIDATE crops taken from a wider scene. For each reference '
            f'(0..{n_refs - 1}), output the candidate index whose person is the SAME '
            'individual as the reference. Use clothing, hair color/length, body shape, '
            'and posture as evidence. '
            f'Output ONLY a JSON list of length {n_refs}, e.g. "[0, 1]". '
            'Use -1 for a reference with no plausible match in the candidates. '
            'Do not include explanations.'
        )
        user_content: List[Dict[str, Any]] = []
        for i, u in enumerate(ref_urls):
            user_content.append({'type': 'text', 'text': f'Reference {i}:'})
            user_content.append({'type': 'image_url', 'image_url': {'url': u}})
        for j, u in enumerate(cand_urls):
            user_content.append({'type': 'text', 'text': f'Candidate {j}:'})
            user_content.append({'type': 'image_url', 'image_url': {'url': u}})
        user_content.append({'type': 'text', 'text': f'Output JSON list of length {n_refs}.'})

        logger.write_request(name, {
            'model': args.model,
            'system_prompt': sys_prompt,
            'n_refs': n_refs,
            'n_cand': n_cand,
            'ref_paths': [str(ref_a_path), str(ref_b_path)],
            'scene_path': str(scene_path),
        })

        client = _new_client(args.timeout_s)
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': user_content},
            ],
        )
        raw = completion.choices[0].message.content or ''
        if args.verbose:
            print(f'    raw: {raw[:200]}')

        try:
            result = ast.literal_eval(raw.strip())
        except Exception as exc:
            logger.write_response(name, {'raw': raw, 'parse_error': str(exc)})
            return False, f'parse failed: {raw[:80]!r}'

        logger.write_response(name, {'raw': raw, 'parsed': result})

        # Overlay: draw a vertical split on the scene + label the candidate that
        # each reference matched to.
        overlay = scene.copy()
        cv2.line(overlay, (w // 2, 0), (w // 2, h), (255, 255, 0), 2)
        _draw_label(overlay, 'cand 0', (10, 30), color=(255, 255, 0))
        _draw_label(overlay, 'cand 1', (w // 2 + 10, 30), color=(255, 255, 0))
        if isinstance(result, list):
            for ref_idx, cand_idx in enumerate(result):
                if isinstance(cand_idx, int) and 0 <= cand_idx < n_cand:
                    cx = (w // 4) if cand_idx == 0 else (3 * w // 4)
                    cv2.circle(overlay, (cx, h // 2), 25, (0, 0, 255), 3)
                    _draw_label(overlay, f'ref {ref_idx} -> {cand_idx}',
                                (cx - 60, h // 2 - 40), color=(0, 0, 255))

        # Stack reference thumbnails next to the scene for context
        thumb_h = 200
        ref_a_thumb = cv2.resize(ref_a, (int(ref_a.shape[1] * thumb_h / ref_a.shape[0]), thumb_h))
        ref_b_thumb = cv2.resize(ref_b, (int(ref_b.shape[1] * thumb_h / ref_b.shape[0]), thumb_h))
        ref_a_thumb = _annotate_top(ref_a_thumb, 'ref 0')
        ref_b_thumb = _annotate_top(ref_b_thumb, 'ref 1')
        scene_thumb = cv2.resize(overlay, (int(overlay.shape[1] * thumb_h / overlay.shape[0]), thumb_h))
        scene_thumb = _annotate_top(scene_thumb, f'matched: {result}')
        composite = _stack_horizontal([ref_a_thumb, ref_b_thumb, scene_thumb])
        logger.write_image(name, 'overlay.jpg', composite)

        if not isinstance(result, list) or len(result) != n_refs:
            return False, f'bad shape: {result}'
        if not all(isinstance(x, int) and -1 <= x < n_cand for x in result):
            return False, f'bad values: {result}'
        return True, f'matched {result}'
    return run


def make_test_vlm_bbox(args, logger: TestLogger) -> Callable[[], Tuple[bool, str]]:
    name = 'vlm_bbox'

    def run() -> Tuple[bool, str]:
        img_path = _download_with_cache('kitchen_fruit.jpg', args.cache_dir, args.verbose)
        img = _read_bgr(img_path)
        logger.write_image(name, 'orig.jpg', img)

        # Image has a fruit basket with oranges + bananas in the foreground.
        prompt = 'banana'
        logger.write_request(name, {
            'model': args.model,
            'prompt': prompt,
            'image_path': str(img_path),
            'image_shape': list(img.shape),
            'max_retries': args.max_retries,
            'timeout_s': args.timeout_s,
        })
        boxes, elapsed = request_bboxes(
            img, prompt,
            model=args.model, max_retries=args.max_retries, timeout_s=args.timeout_s,
        )
        used_prompt = prompt
        if not boxes:
            # Fall back to a broader fruit-class prompt if 'banana' returned
            # nothing — proves the call path works on this image.
            used_prompt = 'fruit'
            boxes, elapsed = request_bboxes(
                img, used_prompt,
                model=args.model, max_retries=args.max_retries, timeout_s=args.timeout_s,
            )

        logger.write_response(name, {
            'used_prompt': used_prompt,
            'boxes': [list(b) for b in boxes],
            'elapsed_s': round(elapsed, 3),
        })
        overlay = _draw_bboxes(img, list(boxes),
                               labels=[f'{used_prompt} {i}' for i in range(len(boxes))])
        logger.write_image(name, 'overlay.jpg', overlay)

        if not boxes:
            return False, f'no boxes after {elapsed:.2f}s'
        h, w = img.shape[:2]
        for x1, y1, x2, y2 in boxes:
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                return False, f'bbox out of bounds: {(x1, y1, x2, y2)} for {w}x{h}'
        return True, f'{len(boxes)} boxes, {elapsed:.2f}s'
    return run


def make_test_seat_recommend(args, logger: TestLogger) -> Callable[[], Tuple[bool, str]]:
    name = 'seat_recommend'

    def run() -> Tuple[bool, str]:
        img_path = _download_with_cache('dining_living.jpg', args.cache_dir, args.verbose)
        img = _read_bgr(img_path)
        logger.write_image(name, 'orig.jpg', img)

        names = ['Alice']
        features = ['blonde hair, blue shirt']
        logger.write_request(name, {
            'model': args.model,
            'names': names,
            'features': features,
            'image_path': str(img_path),
            'image_shape': list(img.shape),
            'max_retries': args.max_retries,
            'timeout_s': args.timeout_s,
        })

        label, point, visible_seats, elapsed = request_seat(
            img,
            names=names,
            features=features,
            model=args.model,
            max_retries=args.max_retries,
            timeout_s=args.timeout_s,
        )

        logger.write_response(name, {
            'label': label,
            'point_xy': list(point) if point else None,
            'visible_seats': visible_seats,
            'elapsed_s': round(elapsed, 3),
        })

        overlay = img.copy()
        if point is not None:
            overlay = _draw_point(overlay, point, label=str(label))
        else:
            _draw_label(overlay, f'label={label!r} (no point)', (10, 30),
                         color=(0, 0, 255))
        # List visible seats along the top
        annotation = f'{len(visible_seats)} seats reported; chosen={label!r}'
        overlay = _annotate_top(overlay, annotation)
        logger.write_image(name, 'overlay.jpg', overlay)

        if not visible_seats:
            return False, f'no seats reported in {elapsed:.2f}s'
        return True, f'label={label!r}, point={point}, seats={len(visible_seats)}'
    return run


_GROCERY_SYS_TEMPLATE = (
    'You will be given a picture of a shelf with {n} main visible layers. '
    'Items on each layer of the shelf are already grouped according to three '
    'categories: food, drink, and utilities. You will then be given a picture of '
    'an object. Please determine which layer the object should be placed on the '
    'shelf. Output a JSON object with keys: object_description (str), '
    'shelf_description (str), reason (str), layer (int — bottom layer is 0).'
)


def make_test_grocery_categorize(args, logger: TestLogger) -> Callable[[], Tuple[bool, str]]:
    name = 'grocery_categorize'

    def run() -> Tuple[bool, str]:
        # The kitchen image has upper cabinets (stand-in for shelf layers) and
        # fruit on the table — feeding the same image as both shelf and object
        # exercises the prompt + JSON-parse path without needing a separate
        # object photo. We're not validating placement correctness, just the
        # call/parse cycle.
        img_path = _download_with_cache('kitchen_fruit.jpg', args.cache_dir, args.verbose)
        img = _read_bgr(img_path)
        logger.write_image(name, 'orig.jpg', img)
        url = encode_to_data_url(img)

        sys_prompt = _GROCERY_SYS_TEMPLATE.format(n=3)
        logger.write_request(name, {
            'model': args.model,
            'system_prompt': sys_prompt,
            'image_path': str(img_path),
            'image_shape': list(img.shape),
            'response_format': 'json_object',
        })

        client = _new_client(args.timeout_s)
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'picture of shelf'},
                    {'type': 'image_url', 'image_url': {'url': url}},
                    {'type': 'text', 'text': 'picture of new object.'},
                    {'type': 'image_url', 'image_url': {'url': url}},
                ]},
            ],
            response_format={'type': 'json_object'},
        )
        raw = completion.choices[0].message.content or ''
        if args.verbose:
            print(f'    raw: {raw[:200]}')

        try:
            obj = json.loads(raw)
        except Exception as exc:
            logger.write_response(name, {'raw': raw, 'parse_error': str(exc)})
            return False, f'parse failed: {exc}'

        logger.write_response(name, {'raw': raw, 'parsed': obj})

        layer_str = f'layer={obj.get("layer")}' if 'layer' in obj else 'layer=?'
        logger.write_image(name, 'overlay.jpg', _annotate_top(img, layer_str))

        if 'layer' not in obj or 'shelf_description' not in obj:
            return False, f'missing keys: {list(obj.keys())}'
        if obj['layer'] is not None and not isinstance(obj['layer'], int):
            return False, f'layer not int|None: {obj["layer"]!r}'
        if not str(obj['shelf_description']).strip():
            return False, 'empty shelf_description'
        return True, f'layer={obj["layer"]}, shelf_desc {len(str(obj["shelf_description"]))} chars'
    return run


# ---------------------------------------------------------------------------
# Fixture-mocked tests — validate the post-fix retry/timeout defenses
# ---------------------------------------------------------------------------

def _mock_completion(content: str):
    """Build an object that quacks like an OpenAI completion."""
    msg = mock.MagicMock()
    msg.content = content
    choice = mock.MagicMock()
    choice.message = msg
    completion = mock.MagicMock()
    completion.choices = [choice]
    return completion


def _patch_openai_with_responses(responses: List[Any]) -> mock.MagicMock:
    """Patch openai.OpenAI so the lazy `from openai import OpenAI` inside
    ``request_bboxes`` and friends picks up our fake. ``responses`` is a list of
    either completion objects (returned) or Exception instances (raised).
    Patches the class — caller is responsible for `start()`/`stop()`."""
    fake_client = mock.MagicMock()
    fake_client.with_options.return_value = fake_client  # chain returns self
    fake_client.chat.completions.create.side_effect = responses
    fake_client.close = mock.MagicMock()
    fake_class = mock.MagicMock(return_value=fake_client)
    return mock.patch('openai.OpenAI', fake_class)


def make_test_parse_recovery(_args, logger: TestLogger) -> Callable[[], Tuple[bool, str]]:
    """request_bboxes succeeds on the 3rd attempt after 2 garbage responses."""
    name = 'parse_recovery'

    def run() -> Tuple[bool, str]:
        valid_payload = {'detections': [{'label': 'cup', 'box_2d': [100, 100, 300, 400]}]}
        responses = [_mock_completion('not json {{{'),
                     _mock_completion('still not json'),
                     _mock_completion(json.dumps(valid_payload))]
        logger.write_request(name, {
            'kind': 'mocked',
            'mock_responses': ['not json {{{', 'still not json', json.dumps(valid_payload)],
            'max_retries': 3, 'timeout_s': 5.0,
        })
        with _patch_openai_with_responses(responses):
            os.environ.setdefault('OPENROUTER_API_KEY', 'fake-for-mock')
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            boxes, elapsed = request_bboxes(
                img, 'cup',
                model='mocked', max_retries=3, timeout_s=5.0,
            )
        logger.write_response(name, {
            'boxes': [list(b) for b in boxes],
            'elapsed_s': round(elapsed, 3),
            'expected': '1 box (recovered on attempt 3)',
        })
        if len(boxes) != 1:
            return False, f'expected 1 box, got {len(boxes)} in {elapsed:.2f}s'
        return True, f'succeeded on attempt 3 ({len(boxes)} box, {elapsed:.2f}s)'
    return run


def make_test_parse_exhaustion(_args, logger: TestLogger) -> Callable[[], Tuple[bool, str]]:
    """All 3 attempts fail. Asserts:
      (a) request_bboxes returns empty (vlm_bbox.py).
      (b) The wholesale tk23 cyclic-fallback regression in feature_matching.py
          is dead — by static inspection of the installed module source. The
          new code uses per-cell `i % n_cand` patching when the VLM responds
          but a single cell is bad; a TOTAL-failure cyclic fallback (the old
          `result = [i % n_cand for i in range(...)]` pattern) is what we
          guard against, not per-cell patches.
      (c) Total-failure path returns the `'VLM match failed on every
          provider'` status=1 error (the provider-chain successor of the
          original `'VLM exhausted'` message).
    """
    name = 'parse_exhaustion'

    def run() -> Tuple[bool, str]:
        logger.write_request(name, {
            'kind': 'mocked',
            'mock_responses': ['not json {{{'] * 3,
            'max_retries': 3, 'timeout_s': 5.0,
            'asserts': [
                'request_bboxes returns []',
                "feature_matching source contains no `i % n_cand for i in range`",
                "feature_matching source contains 'VLM match failed on every provider'",
            ],
        })

        responses = [_mock_completion('not json {{{')] * 3
        with _patch_openai_with_responses(responses):
            os.environ.setdefault('OPENROUTER_API_KEY', 'fake-for-mock')
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            boxes, elapsed = request_bboxes(
                img, 'cup',
                model='mocked', max_retries=3, timeout_s=5.0,
            )
        if boxes:
            logger.write_response(name, {'boxes': [list(b) for b in boxes],
                                          'failure': 'expected empty list'})
            return False, f'expected empty list, got {len(boxes)}'

        # Static check on the installed feature_matching source.
        import inspect
        from kimi_api import feature_matching
        src = inspect.getsource(feature_matching)
        wholesale_cyclic_present = 'i % n_cand for i in range' in src
        post_fix_msg_present = 'VLM match failed on every provider' in src

        logger.write_response(name, {
            'request_bboxes_boxes': [],
            'request_bboxes_elapsed_s': round(elapsed, 3),
            'wholesale_cyclic_fallback_in_source': wholesale_cyclic_present,
            'post_fix_error_msg_in_source': post_fix_msg_present,
            'feature_matching_path': inspect.getfile(feature_matching),
        })

        if wholesale_cyclic_present:
            return False, 'wholesale cyclic-fallback regression in feature_matching.py'
        if not post_fix_msg_present:
            return False, ("expected 'VLM match failed on every provider' "
                           'error_msg not found in feature_matching.py')

        return True, (
            f'request_bboxes empty in {elapsed:.2f}s; '
            'feature_matching wholesale cyclic-fallback dead'
        )
    return run


def make_test_timeout_fires(_args, logger: TestLogger) -> Callable[[], Tuple[bool, str]]:
    """Each VLM call raises a fake timeout. The retry loop hits all 3 and
    returns empty fast — proves the no-timeout-defense is dead."""
    name = 'timeout_fires'

    def run() -> Tuple[bool, str]:
        class FakeTimeout(Exception):
            pass

        logger.write_request(name, {
            'kind': 'mocked',
            'mock_behavior': 'every create() raises FakeTimeout',
            'max_retries': 3, 'timeout_s': 0.5,
        })

        responses = [FakeTimeout('simulated timeout')] * 3
        with _patch_openai_with_responses(responses):
            os.environ.setdefault('OPENROUTER_API_KEY', 'fake-for-mock')
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            boxes, elapsed = request_bboxes(
                img, 'cup',
                model='mocked', max_retries=3, timeout_s=0.5,
            )
            wall = time.perf_counter() - t0

        logger.write_response(name, {
            'boxes': [list(b) for b in boxes],
            'elapsed_s': round(elapsed, 3),
            'wall_s': round(wall, 3),
        })

        if boxes:
            return False, f'expected empty after timeout, got {len(boxes)}'
        if wall > 5.0:
            return False, f'wall={wall:.2f}s exceeded budget'
        return True, f'3 timeouts caught in {wall:.2f}s'
    return run


# ---------------------------------------------------------------------------
# Test registry & main
# ---------------------------------------------------------------------------

ALL_TESTS: List[Tuple[str, str]] = [
    # (name, kind)  kind = 'live' or 'mock'
    ('feature_extraction',  'live'),
    ('feature_matching',    'live'),
    ('vlm_bbox',            'live'),
    ('seat_recommend',      'live'),
    ('grocery_categorize',  'live'),
    ('parse_recovery',      'mock'),
    ('parse_exhaustion',    'mock'),
    ('timeout_fires',       'mock'),
]

_FACTORIES: Dict[str, Callable[[Any, TestLogger], Callable[[], Tuple[bool, str]]]] = {
    'feature_extraction':  make_test_feature_extraction,
    'feature_matching':    make_test_feature_matching,
    'vlm_bbox':            make_test_vlm_bbox,
    'seat_recommend':      make_test_seat_recommend,
    'grocery_categorize':  make_test_grocery_categorize,
    'parse_recovery':      make_test_parse_recovery,
    'parse_exhaustion':    make_test_parse_exhaustion,
    'timeout_fires':       make_test_timeout_fires,
}


def _format_summary(results: List[TestResult]) -> str:
    lines: List[str] = []
    lines.append(f'{"TEST":<22} {"STATUS":<8} {"ELAPSED":<10} NOTES')
    lines.append('-' * 80)
    n_pass = n_fail = 0
    for r in results:
        lines.append(f'{r.name:<22} {r.status:<8} {r.elapsed_s:>7.2f}s   {r.notes}')
        if r.status == 'PASS':
            n_pass += 1
        elif r.status == 'FAIL':
            n_fail += 1
    lines.append('')
    lines.append(f'{n_pass} passed / {n_fail} failed')
    if n_fail:
        lines.append('')
        lines.append('FAILURES:')
        for r in results:
            if r.status == 'FAIL':
                lines.append(f'  {r.name}: {r.notes}')
                if r.error:
                    lines.append('    ' + r.error.replace('\n', '\n    '))
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', type=str, default='', help='Comma-separated test names')
    parser.add_argument('--model', type=str, default='', help='LLM model override')
    parser.add_argument('--timeout-s', type=float, default=30.0)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--log-dir', type=Path, default=Path('vision_log'),
                        help='Log root (a timestamped subdir is created inside)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    load_env()

    selected = [n.strip() for n in args.only.split(',') if n.strip()] if args.only else \
               [name for name, _ in ALL_TESTS]
    unknown = [n for n in selected if n not in _FACTORIES]
    if unknown:
        print(f'[FATAL] unknown test name(s): {unknown}', file=sys.stderr)
        print(f'available: {list(_FACTORIES.keys())}', file=sys.stderr)
        return 2

    needs_live = any(kind == 'live' for name, kind in ALL_TESTS if name in selected)
    if needs_live:
        try:
            require_api_key()
        except RuntimeError as exc:
            print(f'[FATAL] {exc}', file=sys.stderr)
            return 2

    if not args.model:
        args.model = os.environ.get('LLM_MODEL') or 'google/gemini-2.5-flash'

    # Set up logging — create the run directory and tee stdout/stderr to run.log.
    logger = TestLogger(args.log_dir)
    log_file = (logger.run_dir / 'run.log').open('w', buffering=1)
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_stdout, log_file)
    sys.stderr = _Tee(real_stderr, log_file)

    try:
        print(f'tk26_vision web-image VLM smoke test')
        print(f'  model:       {args.model}')
        print(f'  timeout_s:   {args.timeout_s}')
        print(f'  max_retries: {args.max_retries}')
        print(f'  cache_dir:   {args.cache_dir}')
        print(f'  log_dir:     {logger.run_dir}')
        print(f'  tests:       {", ".join(selected)}')
        print()

        results: List[TestResult] = []
        for name, _ in ALL_TESTS:
            if name not in selected:
                continue
            print(f'>> {name}')
            factory = _FACTORIES[name]
            _run(name, factory(args, logger), results=results, logger=logger)

        summary_text = _format_summary(results)
        print()
        print(summary_text)
        logger.write_summary(results)
        print(f'logs written to: {logger.run_dir}')
        return 0 if all(r.status == 'PASS' for r in results) else 1
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        log_file.close()


if __name__ == '__main__':
    sys.exit(main())
