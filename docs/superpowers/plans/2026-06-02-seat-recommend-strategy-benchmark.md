# Seat-Recommendation Strategy Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained offline harness inside the seat-recommendation package that replays logged camera images through four seat-selection strategies × two VLM providers, scores 2D localization against hand-annotated ground truth, and produces a scoreboard so we can decide which approach to port into production.

**Architecture:** A new Python package `src/kimi_api/seat_bench/` (sibling to the `kimi_api` module, so it imports `kimi_api._seat_vlm`/`_image_utils`/`_env` without install). Shared leaf modules (`providers`, `geometry`, `collect`, `score`) are built and unit-tested first. The four strategy runners (`s0`–`s3`) are independent modules that depend only on the shared leaves, so they can be implemented and run **in parallel**. `run.py` executes one `(strategy, provider)` grid cell; the benchmark run dispatches one subagent per cell concurrently; `aggregate.py` builds `report.md` + contact sheets with no API calls.

**Tech Stack:** Python 3.10 (`.venv-vision-main`), OpenAI SDK (OpenRouter + DashScope OpenAI-compatible endpoints), OpenCV, NumPy, Ultralytics YOLO-World, pytest.

**Out of scope (deferred to its own spec):** porting the winning strategy into `_seat_vlm.py` / `seat_recommend_bbox.py`, and anything depth/3D — the logs have no raw depth frames, so this harness scores **2D localization only** (does the recommendation land on the correct empty cushion). The depth/snap/robust-resolve pipeline is downstream of the pixel and is not what's broken.

---

## Background (read before starting)

The production node `src/kimi_api/kimi_api/seat_recommend_bbox.py` asks Gemini (`_seat_vlm.request_seat`) for a single **pointing pixel** `[y,x]` (0–1000 normalized) + a short label, then snaps that pixel to a horizontal surface and samples depth. Logged overlays in `../../vision_log/**/*seat*overlay*.jpg` show the **label is usually right** ("leftmost chair") but the **point lands on the wrong object** (the floor between a seated person's feet, or the person's lap), 200–300 px from the real empty cushion. Snap-to-horizontal can't rescue a point that far onto the wrong object. This harness tests whether **bounding-box** prompting (single-call or two-call zoom) or **set-of-mark** selection localizes better.

**Reusable code to study first:**
- `src/kimi_api/kimi_api/_seat_vlm.py` — the current point prompt, strict→loose JSON fallback, Gemini `reasoning` extra_body. S0 wraps this.
- `src/kimi_api/kimi_api/_image_utils.py::encode_to_data_url` — BGR→JPEG data URL.
- `src/kimi_api/kimi_api/_env.py` — `base_url()` (OpenRouter), `dashscope_base_url()`, `require_dashscope_api_key()` (handles the `DASHCOPE_API_KEY` typo), `require_api_key()`.
- `src/tk_vision_specialized/tk_vision_specialized/qwen_match_vlm.py::_decode_bbox` — 0–1000 box decode (vendor this logic into `geometry.py`).
- `src/object_detection_generalist/object_detection_generalist/world_bbox.py::WorldDetector` — YOLO-World open-vocab detector with the critical CUDA device-pinning logic. **Vendor a slim copy** into `seat_bench/world_candidates.py` (do not cross-import; keep the harness self-contained inside the seat folder).

**API keys** (already present in workspace-root `.env`): `OPENROUTER_API_KEY` (Gemini) and `DASHCOPE_API_KEY` (Qwen via DashScope).

**How to run anything in this harness:**
```bash
source src/tk26_vision/.venv-vision-main/bin/activate
cd src/kimi_api
python -m seat_bench.<module> ...      # kimi_api/ and seat_bench/ are both importable from here
pytest seat_bench/test/ -v
```

---

## File structure

All paths under `src/kimi_api/`:

```
seat_bench/
├── __init__.py
├── README.md                 # how to run, dir layout, scoring definition
├── paths.py                  # repo-relative dir constants + sys.path bootstrap for kimi_api
├── geometry.py               # decode 0-1000 point/box, point_in_box, iou, box_center, draw_overlay
├── providers.py              # call_vlm(provider, messages, schema, ...) -> parsed dict (gemini|qwen)
├── collect.py                # build dataset/ : dedup by md5, copy distinct images, pull req JSON, manifest.json
├── world_candidates.py       # vendored slim YOLO-World multi-term seat detector (for S3)
├── score.py                  # classify each result vs GT, aggregate per cell
├── run.py                    # CLI: run ONE (strategy, provider) cell over all images -> results/<cell>/
├── aggregate.py              # CLI: results/ -> report.md + sheets/<cell>.jpg (no API calls)
├── strategies/
│   ├── __init__.py           # STRATEGIES registry {name: run_fn}
│   ├── base.py               # Result dataclass + build_request_text helper
│   ├── s0_point.py           # control: current pointing prompt
│   ├── s1_bbox_select.py     # single call: per-seat boxes + occupancy + chosen empty
│   ├── s2_zoom.py            # call1 coarse boxes+select -> crop -> call2 refine -> map back
│   └── s3_som.py             # YOLO-World candidates -> numbered marks -> VLM picks a number
├── dataset/                  # GENERATED: <id>.jpg, <id>.req.json, <id>.gt.json, manifest.json
├── results/                  # GENERATED: <strategy>_<provider>/<id>.json + <id>.jpg
├── sheets/                   # GENERATED: <strategy>_<provider>.jpg contact sheets
├── report.md                 # GENERATED: scoreboard
└── test/
    ├── __init__.py
    ├── test_geometry.py
    ├── test_collect.py
    └── test_score.py
```

**Dependency order (informs parallelism):**
1. Tasks 1–2 (scaffold, paths) — foundation.
2. Tasks 3–5 (geometry, providers, collect) — shared leaves, independent of each other.
3. Task 6 (dataset build + **hand annotation**) — produces ground truth; depends on collect.
4. Task 7 (score) — depends on geometry.
5. Task 8 (strategy base + registry).
6. **Tasks 9–12 (S0, S1, S2, S3) — independent; implement in parallel.** S3 also depends on Task 13 (world_candidates).
7. Task 13 (world_candidates) — needed only by S3; can be built in parallel with S0–S2.
8. Task 14 (run.py), Task 15 (aggregate.py).
9. Task 16 — **run the 8-cell grid via concurrent subagents**, then aggregate.

---

## Task 1: Package scaffold

**Files:**
- Create: `src/kimi_api/seat_bench/__init__.py`
- Create: `src/kimi_api/seat_bench/test/__init__.py`
- Create: `src/kimi_api/seat_bench/.gitignore`

- [ ] **Step 1: Create the package init files**

`src/kimi_api/seat_bench/__init__.py`:
```python
"""Offline benchmark for seat-recommendation VLM strategies.

Self-contained eval harness. Not part of the installed kimi_api package;
run modules with `python -m seat_bench.<mod>` from `src/kimi_api/`.
See README.md.
"""
```

`src/kimi_api/seat_bench/test/__init__.py`:
```python
```

- [ ] **Step 2: Ignore generated artifacts**

`src/kimi_api/seat_bench/.gitignore`:
```gitignore
dataset/*.jpg
results/
sheets/
report.md
__pycache__/
```
(Note: `dataset/*.gt.json`, `dataset/*.req.json`, and `dataset/manifest.json` are NOT ignored — ground truth is committed; only the copied images are regenerable.)

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/__init__.py src/kimi_api/seat_bench/test/__init__.py src/kimi_api/seat_bench/.gitignore
git commit -m "feat(seat_bench): package scaffold"
```

---

## Task 2: Paths bootstrap

**Files:**
- Create: `src/kimi_api/seat_bench/paths.py`

- [ ] **Step 1: Write the path constants + import bootstrap**

`src/kimi_api/seat_bench/paths.py`:
```python
"""Filesystem locations + sys.path bootstrap so `import kimi_api.*` works
when running `python -m seat_bench.<mod>` from `src/kimi_api/`."""

from __future__ import annotations

import sys
from pathlib import Path

# seat_bench/ -> kimi_api/ (parent of the seat_bench package dir)
PKG_DIR = Path(__file__).resolve().parent          # .../src/kimi_api/seat_bench
KIMI_API_SRC = PKG_DIR.parent                       # .../src/kimi_api
WORKSPACE_ROOT = KIMI_API_SRC.parents[2]            # .../tk25_ws  (src/kimi_api is under src/tk26_vision/src? see note)

DATASET_DIR = PKG_DIR / "dataset"
RESULTS_DIR = PKG_DIR / "results"
SHEETS_DIR = PKG_DIR / "sheets"
REPORT_PATH = PKG_DIR / "report.md"

# vision_log lives at the OUTER workspace root (tk25_ws/vision_log), two
# levels above tk26_vision. Resolve robustly by walking up for a dir that
# contains a vision_log/ with seat images.
def find_vision_log() -> Path:
    here = PKG_DIR
    for parent in [here, *here.parents]:
        candidate = parent / "vision_log"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate a vision_log/ directory above seat_bench")


def ensure_kimi_api_importable() -> None:
    """Put src/kimi_api on sys.path so `import kimi_api._seat_vlm` resolves."""
    p = str(KIMI_API_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
```

> **Note for implementer:** verify `WORKSPACE_ROOT`/`find_vision_log()` actually resolves to `tk25_ws/vision_log` by running Step 2. The seat package path is `tk25_ws/src/tk26_vision/src/kimi_api/`, and the target logs are at `tk25_ws/vision_log/`. `find_vision_log()` walks up until it finds the dir, so it is robust to the exact depth.

- [ ] **Step 2: Verify path resolution**

Run:
```bash
cd src/kimi_api && python -c "from seat_bench.paths import find_vision_log, DATASET_DIR; print(find_vision_log()); print(DATASET_DIR)"
```
Expected: prints a path ending in `/tk25_ws/vision_log` and `.../seat_bench/dataset`.

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/paths.py
git commit -m "feat(seat_bench): path constants + kimi_api import bootstrap"
```

---

## Task 3: Geometry helpers (TDD)

**Files:**
- Create: `src/kimi_api/seat_bench/geometry.py`
- Test: `src/kimi_api/seat_bench/test/test_geometry.py`

- [ ] **Step 1: Write failing tests**

`src/kimi_api/seat_bench/test/test_geometry.py`:
```python
from seat_bench import geometry as g


def test_decode_point_yx_scales_to_pixels():
    # [y, x] = [500, 250] over a 1000x2000 (h x w) image -> (x=500, y=500)
    assert g.decode_point_yx([500, 250], w=2000, h=1000) == (500, 500)


def test_decode_point_yx_zero_sentinel_is_none():
    assert g.decode_point_yx([0, 0], w=640, h=480) is None


def test_decode_point_yx_malformed_is_none():
    assert g.decode_point_yx("nope", w=640, h=480) is None
    assert g.decode_point_yx([5], w=640, h=480) is None


def test_decode_box_xyxy_scales_and_orders():
    # swapped corners get normalized; 0-1000 -> pixels
    box = g.decode_box_xyxy([500, 500, 250, 250], w=1000, h=1000)
    assert box == (250, 250, 500, 500)


def test_decode_box_xyxy_degenerate_is_none():
    assert g.decode_box_xyxy([100, 100, 100, 100], w=1000, h=1000) is None


def test_point_in_box():
    assert g.point_in_box((50, 50), (0, 0, 100, 100)) is True
    assert g.point_in_box((150, 50), (0, 0, 100, 100)) is False


def test_box_center():
    assert g.box_center((0, 0, 100, 200)) == (50, 100)


def test_iou_identical_is_one():
    assert g.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero():
    assert g.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/kimi_api && pytest seat_bench/test/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'seat_bench.geometry'`.

- [ ] **Step 3: Implement geometry.py**

`src/kimi_api/seat_bench/geometry.py`:
```python
"""Coordinate decode + 2D scoring geometry, shared by all strategies.

Point decode mirrors kimi_api._seat_vlm._decode_point ([y,x] 0-1000).
Box decode mirrors tk_vision_specialized.qwen_match_vlm._decode_bbox
([x1,y1,x2,y2] 0-1000). Both clamp to image bounds.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]            # (x, y) pixels
Box = Tuple[int, int, int, int]    # (x1, y1, x2, y2) pixels


def decode_point_yx(point_yx, w: int, h: int) -> Optional[Point]:
    if not isinstance(point_yx, (list, tuple)) or len(point_yx) < 2:
        return None
    try:
        y0, x0 = float(point_yx[0]), float(point_yx[1])
    except (TypeError, ValueError):
        return None
    if y0 == 0.0 and x0 == 0.0:
        return None
    px = max(0, min(int(round(x0 * w / 1000.0)), w - 1))
    py = max(0, min(int(round(y0 * h / 1000.0)), h - 1))
    return (px, py)


def decode_box_xyxy(box_2d, w: int, h: int) -> Optional[Box]:
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(box_2d[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    px1 = max(0, min(int(round(x1 * w / 1000.0)), w - 1))
    py1 = max(0, min(int(round(y1 * h / 1000.0)), h - 1))
    px2 = max(0, min(int(round(x2 * w / 1000.0)), w - 1))
    py2 = max(0, min(int(round(y2 * h / 1000.0)), h - 1))
    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def point_in_box(pt: Point, box: Box) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_center(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def draw_overlay(
    img_bgr: np.ndarray,
    *,
    point: Optional[Point] = None,
    box: Optional[Box] = None,
    gt_boxes: Optional[list[tuple[Box, bool]]] = None,
    label: str = "",
    hit: Optional[bool] = None,
) -> np.ndarray:
    """Render a result overlay: GT cushions (green=empty, red=occupied),
    predicted box (cyan), predicted point (magenta dot), and a hit/miss tag.
    """
    out = img_bgr.copy()
    if gt_boxes:
        for gbox, occ in gt_boxes:
            color = (0, 0, 200) if occ else (0, 200, 0)
            cv2.rectangle(out, gbox[:2], gbox[2:], color, 2)
    if box is not None:
        cv2.rectangle(out, box[:2], box[2:], (255, 255, 0), 2)
    if point is not None:
        cv2.circle(out, point, 8, (255, 0, 255), -1)
    tag = label
    if hit is not None:
        tag = f"[{'HIT' if hit else 'MISS'}] {label}"
    if tag:
        cv2.putText(out, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/kimi_api && pytest seat_bench/test/test_geometry.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/kimi_api/seat_bench/geometry.py src/kimi_api/seat_bench/test/test_geometry.py
git commit -m "feat(seat_bench): geometry decode/scoring helpers + tests"
```

---

## Task 4: Provider layer

**Files:**
- Create: `src/kimi_api/seat_bench/providers.py`

> No unit test (pure I/O against external APIs). Validated by a live smoke call in Step 2.

- [ ] **Step 1: Implement providers.py**

`src/kimi_api/seat_bench/providers.py`:
```python
"""Unified OpenAI-compatible VLM caller for the benchmark.

Two providers:
  - "gemini": OpenRouter, google/gemini-2.5-pro, reasoning enabled.
  - "qwen":   DashScope OpenAI-compatible, qwen3-vl-plus.

call_vlm() returns the parsed JSON dict (or raises). Strict json_schema is
tried first, falling back to json_object if the route rejects the schema
(same pattern as kimi_api._seat_vlm / qwen_match_vlm).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from .paths import ensure_kimi_api_importable

ensure_kimi_api_importable()
from kimi_api import _env  # noqa: E402


GEMINI_MODEL = "google/gemini-2.5-pro"
QWEN_MODEL = "qwen3-vl-plus"


@dataclass
class ProviderCfg:
    name: str
    model: str
    api_key: str
    base_url: str
    reasoning: bool


def _provider_cfg(provider: str) -> ProviderCfg:
    _env.load_env()
    if provider == "gemini":
        return ProviderCfg("gemini", GEMINI_MODEL, _env.require_api_key(),
                           _env.base_url(), reasoning=True)
    if provider == "qwen":
        return ProviderCfg("qwen", QWEN_MODEL, _env.require_dashscope_api_key(),
                           _env.dashscope_base_url(), reasoning=False)
    raise ValueError(f"unknown provider {provider!r} (expected gemini|qwen)")


def call_vlm(
    provider: str,
    messages: list,
    *,
    schema: Optional[dict] = None,
    schema_name: str = "seat_bench",
    timeout_s: float = 30.0,
    max_retries: int = 3,
    temperature: float = 0.2,
    logger=None,
) -> tuple[dict, float]:
    """Return (parsed_json, elapsed_s). Raises RuntimeError on exhaustion."""
    cfg = _provider_cfg(provider)
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    rf_strict = (
        {"type": "json_schema",
         "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        if schema else {"type": "json_object"}
    )
    rf_loose = {"type": "json_object"}
    use_strict = schema is not None
    extra_body = {"reasoning": {"enabled": True, "max_tokens": 2048}} if cfg.reasoning else None

    t0 = time.perf_counter()
    last_error: Optional[Exception] = None
    try:
        for attempt in range(1, max_retries + 1):
            rf = rf_strict if use_strict else rf_loose
            try:
                kwargs = dict(model=cfg.model, messages=messages,
                              response_format=rf, temperature=temperature)
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                completion = client.with_options(timeout=timeout_s).chat.completions.create(**kwargs)
                raw = completion.choices[0].message.content or ""
                return json.loads(_strip_fences(raw)), time.perf_counter() - t0
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                if logger:
                    logger(f"[{provider}] JSON parse failed attempt {attempt}/{max_retries}: {exc}")
            except Exception as exc:  # noqa: BLE001
                txt = str(exc).lower()
                if use_strict and any(k in txt for k in ("json_schema", "response_format", "schema")):
                    use_strict = False
                    if logger:
                        logger(f"[{provider}] schema rejected, falling back to json_object: {exc}")
                last_error = exc
                if logger:
                    logger(f"[{provider}] call failed attempt {attempt}/{max_retries}: {exc}")
        raise RuntimeError(f"[{provider}] exhausted {max_retries} retries; last={last_error}")
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


_FENCE = __import__("re").compile(r"^\s*```(?:json)?\s*|\s*```\s*$", __import__("re").MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text).strip() if "```" in text else text
```

- [ ] **Step 2: Smoke-test both providers live**

Run (sends one tiny text-only JSON request to each — costs ~nothing):
```bash
cd src/kimi_api && python -c "
from seat_bench.providers import call_vlm
msg=[{'role':'user','content':'Return JSON {\"ok\": true}'}]
for p in ('gemini','qwen'):
    out,el=call_vlm(p,msg,timeout_s=30,max_retries=2)
    print(p, out, round(el,2),'s')
"
```
Expected: each prints a dict containing `ok` and an elapsed time. If `qwen` raises a key error, confirm `DASHCOPE_API_KEY` is in the root `.env`.

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/providers.py
git commit -m "feat(seat_bench): unified gemini/qwen provider caller"
```

---

## Task 5: Dataset collector (TDD on dedup)

**Files:**
- Create: `src/kimi_api/seat_bench/collect.py`
- Test: `src/kimi_api/seat_bench/test/test_collect.py`

- [ ] **Step 1: Write failing test for the pure dedup + req-pairing logic**

`src/kimi_api/seat_bench/test/test_collect.py`:
```python
from seat_bench import collect


def test_dedupe_keeps_first_of_identical_bytes(tmp_path):
    a = tmp_path / "a.jpg"; a.write_bytes(b"IMG1")
    b = tmp_path / "b.jpg"; b.write_bytes(b"IMG1")   # duplicate of a
    c = tmp_path / "c.jpg"; c.write_bytes(b"IMG2")
    distinct = collect.dedupe_by_content([a, b, c])
    assert len(distinct) == 2
    assert a in distinct and c in distinct
    assert b not in distinct


def test_req_path_for_orig_swaps_tokens(tmp_path):
    orig = tmp_path / "node_seat_recommend_bbox_orig_20260503_120414_420.jpg"
    expected = tmp_path / "node_seat_recommend_bbox_req_20260503_120414_420.json"
    assert collect.req_path_for_orig(orig) == expected
```

- [ ] **Step 2: Run to verify failure**

Run: `cd src/kimi_api && pytest seat_bench/test/test_collect.py -v`
Expected: FAIL — `ModuleNotFoundError` / missing functions.

- [ ] **Step 3: Implement collect.py**

`src/kimi_api/seat_bench/collect.py`:
```python
"""Build seat_bench/dataset/ from logged vision_log seat images.

Scans for *seat*orig*.jpg, dedupes byte-identical files (the 'copy'
sessions are literal cp -r duplicates), copies each distinct scene to
dataset/<id>.jpg, and pairs the matching *_req_*.json so strategies can
replay realistic names/features/known_seats. Writes dataset/manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .paths import DATASET_DIR, find_vision_log


def find_seat_origs() -> list[Path]:
    root = find_vision_log()
    return sorted(root.rglob("*seat*orig*.jpg"))


def dedupe_by_content(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    distinct: list[Path] = []
    for p in paths:
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            distinct.append(p)
    return distinct


def req_path_for_orig(orig: Path) -> Path:
    name = orig.name.replace("_orig_", "_req_")
    name = name.rsplit(".", 1)[0] + ".json"
    return orig.with_name(name)


def _load_req(orig: Path) -> dict:
    req = req_path_for_orig(orig)
    if not req.is_file():
        return {"names": [], "features": [], "known_seats": []}
    data = json.loads(req.read_text()).get("request", {})
    return {
        "names": data.get("names", []),
        "features": data.get("features", []),
        "known_seats": data.get("known_seats", []),
    }


def build() -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    origs = find_seat_origs()
    distinct = dedupe_by_content(origs)
    manifest = []
    for i, src in enumerate(distinct):
        sid = f"scene_{i:03d}"
        dst_img = DATASET_DIR / f"{sid}.jpg"
        shutil.copyfile(src, dst_img)
        req = _load_req(src)
        (DATASET_DIR / f"{sid}.req.json").write_text(json.dumps(req, indent=2))
        manifest.append({"id": sid, "src": str(src), **req})
    (DATASET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"collected {len(distinct)} distinct scenes (from {len(origs)} origs)"
          f" -> {DATASET_DIR}")
    return DATASET_DIR


if __name__ == "__main__":
    build()
```

- [ ] **Step 4: Run unit tests to verify pass**

Run: `cd src/kimi_api && pytest seat_bench/test/test_collect.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/kimi_api/seat_bench/collect.py src/kimi_api/seat_bench/test/test_collect.py
git commit -m "feat(seat_bench): dataset collector with byte-dedup + req pairing"
```

---

## Task 6: Build the dataset and hand-annotate ground truth

**Files:**
- Generate: `src/kimi_api/seat_bench/dataset/scene_*.jpg`, `scene_*.req.json`, `manifest.json`
- Create (by hand): `src/kimi_api/seat_bench/dataset/scene_*.gt.json`

> This task is interactive — the implementing agent (or operator) must **look at each image** and write ground truth. Cushion bboxes are generous (scoring is point-in-box), so eyeballed coordinates are acceptable.

- [ ] **Step 1: Build the dataset**

Run: `cd src/kimi_api && python -m seat_bench.collect`
Expected: prints "collected N distinct scenes (from 36 origs)" with N ≈ 20–25, and populates `seat_bench/dataset/`.

- [ ] **Step 2: Define the GT schema**

Each `dataset/<id>.gt.json`:
```json
{
  "id": "scene_000",
  "image_wh": [1280, 720],
  "seats": [
    {"label": "leftmost chair",  "occupied": false, "cushion_bbox": [360, 400, 520, 470]},
    {"label": "middlemost chair","occupied": true,  "cushion_bbox": [540, 410, 700, 560]},
    {"label": "rightmost chair", "occupied": false, "cushion_bbox": [710, 400, 880, 470]}
  ]
}
```
Rules:
- One entry per sittable cushion. `occupied=true` if a person/large object sits on the cushion.
- `cushion_bbox` is `[x1,y1,x2,y2]` in pixels over the **full image**, tight-ish around the visible seat surface (where a person's seat would touch). Generous is fine; do NOT include the backrest.
- Use the `known_seats` from the scene's `.req.json` as the label vocabulary when present; otherwise use descriptive labels.

- [ ] **Step 3: Annotate every scene**

For each `scene_NNN.jpg`: open it (Read tool renders the image; check `image_wh` with `python -c "import cv2;print(cv2.imread('dataset/scene_000.jpg').shape)"`), determine each seat's occupancy and cushion bbox, and write `dataset/scene_NNN.gt.json`. Cross-check occupancy against the scene's `.req.json` `visible_seats` if it carried one, but trust your own eyes for the bbox.

- [ ] **Step 4: Validate all GT files parse and reference in-bounds boxes**

Run:
```bash
cd src/kimi_api && python -c "
import json, glob, cv2
for f in sorted(glob.glob('seat_bench/dataset/*.gt.json')):
    gt=json.load(open(f)); w,h=gt['image_wh']
    for s in gt['seats']:
        x1,y1,x2,y2=s['cushion_bbox']
        assert 0<=x1<x2<=w and 0<=y1<y2<=h, (f,s)
    print(f, len(gt['seats']),'seats, empty=',sum(1 for s in gt['seats'] if not s['occupied']))
print('all GT valid')
"
```
Expected: one line per scene, ending in "all GT valid".

- [ ] **Step 5: Commit ground truth**

```bash
git add src/kimi_api/seat_bench/dataset/*.gt.json src/kimi_api/seat_bench/dataset/*.req.json src/kimi_api/seat_bench/dataset/manifest.json
git commit -m "data(seat_bench): hand-annotated ground truth for N seat scenes"
```

---

## Task 7: Scorer (TDD)

**Files:**
- Create: `src/kimi_api/seat_bench/score.py`
- Test: `src/kimi_api/seat_bench/test/test_score.py`

- [ ] **Step 1: Write failing tests**

`src/kimi_api/seat_bench/test/test_score.py`:
```python
from seat_bench import score

GT = {
    "id": "s0", "image_wh": [1000, 1000],
    "seats": [
        {"label": "left",  "occupied": False, "cushion_bbox": [0, 0, 100, 100]},
        {"label": "right", "occupied": True,  "cushion_bbox": [200, 200, 300, 300]},
    ],
}


def test_point_in_empty_cushion_is_hit():
    r = score.classify({"point_xy": [50, 50], "chosen_label": "left"}, GT)
    assert r["outcome"] == "hit"


def test_point_in_occupied_cushion_is_wrong_seat():
    r = score.classify({"point_xy": [250, 250], "chosen_label": "right"}, GT)
    assert r["outcome"] == "wrong_seat"


def test_point_outside_all_is_miss():
    r = score.classify({"point_xy": [900, 900], "chosen_label": "left"}, GT)
    assert r["outcome"] == "miss"


def test_none_when_seat_available_is_false_none():
    r = score.classify({"point_xy": None, "chosen_label": "none"}, GT)
    assert r["outcome"] == "false_none"


def test_none_when_no_empty_seats_is_correct_reject():
    gt = {"image_wh": [1000, 1000],
          "seats": [{"label": "x", "occupied": True, "cushion_bbox": [0, 0, 50, 50]}]}
    r = score.classify({"point_xy": None, "chosen_label": "none"}, gt)
    assert r["outcome"] == "correct_reject"


def test_aggregate_counts_hit_rate():
    rows = [{"outcome": "hit"}, {"outcome": "miss"}, {"outcome": "hit"},
            {"outcome": "wrong_seat"}]
    agg = score.aggregate(rows)
    assert agg["n"] == 4
    assert agg["hits"] == 2
    assert abs(agg["hit_rate"] - 0.5) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `cd src/kimi_api && pytest seat_bench/test/test_score.py -v`
Expected: FAIL — missing module/functions.

- [ ] **Step 3: Implement score.py**

`src/kimi_api/seat_bench/score.py`:
```python
"""Score a strategy result against hand-annotated ground truth (2D only).

Outcome taxonomy (per scene):
  hit           : recommendation point lands inside an EMPTY cushion bbox.
  wrong_seat    : point lands inside an OCCUPIED cushion bbox.
  miss          : point lands outside every cushion bbox (but a seat existed).
  false_none    : strategy said "none" though >=1 empty seat exists.
  correct_reject: strategy said "none" and no empty seat exists.
The headline metric is hit_rate = hits / scenes_with_empty_seat.
"""

from __future__ import annotations

from typing import Optional

from .geometry import point_in_box


def _has_empty(gt: dict) -> bool:
    return any(not s["occupied"] for s in gt["seats"])


def classify(result: dict, gt: dict) -> dict:
    point = result.get("point_xy")
    chose_none = (
        point is None
        or str(result.get("chosen_label", "")).strip().lower() == "none"
    )
    empty_exists = _has_empty(gt)

    if chose_none:
        outcome = "false_none" if empty_exists else "correct_reject"
        return {"outcome": outcome, "in_box": None}

    pt = (int(point[0]), int(point[1]))
    for s in gt["seats"]:
        if point_in_box(pt, tuple(s["cushion_bbox"])):
            return {
                "outcome": "hit" if not s["occupied"] else "wrong_seat",
                "in_box": s["label"],
            }
    return {"outcome": "miss", "in_box": None}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    by = {}
    for r in rows:
        by[r["outcome"]] = by.get(r["outcome"], 0) + 1
    hits = by.get("hit", 0)
    # denominator excludes correct_reject (no empty seat to find)
    scored = n - by.get("correct_reject", 0)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": hits / scored if scored else 0.0,
        "wrong_seat": by.get("wrong_seat", 0),
        "miss": by.get("miss", 0),
        "false_none": by.get("false_none", 0),
        "correct_reject": by.get("correct_reject", 0),
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd src/kimi_api && pytest seat_bench/test/test_score.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/kimi_api/seat_bench/score.py src/kimi_api/seat_bench/test/test_score.py
git commit -m "feat(seat_bench): result scorer + aggregation with tests"
```

---

## Task 8: Strategy base + registry

**Files:**
- Create: `src/kimi_api/seat_bench/strategies/__init__.py`
- Create: `src/kimi_api/seat_bench/strategies/base.py`

- [ ] **Step 1: Implement base.py**

`src/kimi_api/seat_bench/strategies/base.py`:
```python
"""Shared types + helpers for strategy runners.

Every strategy exposes:  run(img_bgr, req, provider, logger=None) -> Result
where `req` is the dict {names, features, known_seats} from the scene's
.req.json. Result is JSON-serializable via asdict().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence


@dataclass
class Result:
    strategy: str
    provider: str
    chosen_label: str = "none"
    point_xy: Optional[list] = None          # [x, y] pixels or None
    box_xyxy: Optional[list] = None          # [x1,y1,x2,y2] pixels or None
    visible_seats: list = field(default_factory=list)
    n_calls: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_request_text(req: dict) -> str:
    """Replay the production request phrasing (mirrors _seat_vlm._build_text_prompt)."""
    names = req.get("names", []) or []
    features = req.get("features", []) or []
    known = req.get("known_seats", []) or []
    text = "Recommend a seat for a new guest."
    for name, feature in zip(names, features):
        text += f" The person matching description: {feature} is called {name}."
    if known:
        lines = "\n".join(f'  - "{s}"' for s in known)
        text += (
            "\n\nThe seats in this room are pre-catalogued. The recommendation "
            "label MUST be exactly one of these strings, or \"none\" if every "
            "catalogued seat is occupied or not visible:\n" + lines
        )
    return text
```

- [ ] **Step 2: Implement the registry (filled in as strategies land)**

`src/kimi_api/seat_bench/strategies/__init__.py`:
```python
"""Strategy registry. Import lazily so a broken strategy doesn't sink the
whole harness; run.py looks strategies up by name."""

from importlib import import_module

_MODULES = {
    "s0": "seat_bench.strategies.s0_point",
    "s1": "seat_bench.strategies.s1_bbox_select",
    "s2": "seat_bench.strategies.s2_zoom",
    "s3": "seat_bench.strategies.s3_som",
}


def get_strategy(name: str):
    if name not in _MODULES:
        raise ValueError(f"unknown strategy {name!r}; choices={list(_MODULES)}")
    return import_module(_MODULES[name]).run


def all_strategy_names() -> list[str]:
    return list(_MODULES)
```

- [ ] **Step 3: Verify import**

Run: `cd src/kimi_api && python -c "from seat_bench.strategies.base import Result, build_request_text; print(Result('s0','gemini').to_dict()['strategy'])"`
Expected: prints `s0`.

- [ ] **Step 4: Commit**

```bash
git add src/kimi_api/seat_bench/strategies/__init__.py src/kimi_api/seat_bench/strategies/base.py
git commit -m "feat(seat_bench): strategy Result type + registry"
```

---

> **Tasks 9–13 are independent and should be implemented in parallel** (one subagent each). Each produces one strategy module (plus Task 13's shared detector for S3). All depend only on Tasks 3–5, 8.

## Task 9: S0 — pointing baseline (control)

**Files:**
- Create: `src/kimi_api/seat_bench/strategies/s0_point.py`

- [ ] **Step 1: Implement s0_point.py**

`src/kimi_api/seat_bench/strategies/s0_point.py`:
```python
"""S0 control: replicate the production pointing prompt across providers.

Uses the same system prompt as kimi_api._seat_vlm so S0 numbers are
directly comparable to what currently ships. Calls go through the
benchmark provider layer (not _seat_vlm.request_seat) so gemini and qwen
share one code path.
"""

from __future__ import annotations

import numpy as np

from ..geometry import decode_point_yx
from ..paths import ensure_kimi_api_importable
from ..providers import call_vlm
from .base import Result, build_request_text

ensure_kimi_api_importable()
from kimi_api._seat_vlm import _SYSTEM_PROMPT, _RESPONSE_SCHEMA  # noqa: E402
from kimi_api._image_utils import encode_to_data_url  # noqa: E402


def run(img_bgr: np.ndarray, req: dict, provider: str, logger=None) -> Result:
    h, w = img_bgr.shape[:2]
    res = Result(strategy="s0", provider=provider)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_to_data_url(img_bgr)}},
            {"type": "text", "text": build_request_text(req)},
        ]},
    ]
    try:
        parsed, elapsed = call_vlm(provider, messages, schema=_RESPONSE_SCHEMA,
                                   schema_name="seat_pointing", logger=logger)
    except Exception as exc:  # noqa: BLE001
        res.error = str(exc)
        return res
    res.elapsed_s = elapsed
    res.n_calls = 1
    res.raw = parsed
    res.chosen_label = str(parsed.get("label", "none") or "none")
    res.visible_seats = parsed.get("visible_seats", []) or []
    pt = decode_point_yx(parsed.get("point"), w, h)
    if res.chosen_label.strip().lower() == "none":
        pt = None
    res.point_xy = list(pt) if pt else None
    return res
```

- [ ] **Step 2: Smoke-test on one scene**

Run:
```bash
cd src/kimi_api && python -c "
import cv2, json
from seat_bench.strategies.s0_point import run
img=cv2.imread('seat_bench/dataset/scene_000.jpg')
req=json.load(open('seat_bench/dataset/scene_000.req.json'))
print(run(img, req, 'gemini', logger=print).to_dict())
"
```
Expected: a dict with `point_xy` (or None) and `chosen_label`; no exception.

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/strategies/s0_point.py
git commit -m "feat(seat_bench): S0 pointing baseline strategy"
```

---

## Task 10: S1 — single-call bbox + select

**Files:**
- Create: `src/kimi_api/seat_bench/strategies/s1_bbox_select.py`

- [ ] **Step 1: Implement s1_bbox_select.py**

`src/kimi_api/seat_bench/strategies/s1_bbox_select.py`:
```python
"""S1: one call returns a box + occupancy for every visible seat AND the
chosen empty seat's label. Recommendation point = center of the chosen
seat's box. Tests whether box regression localizes truer than pointing.
"""

from __future__ import annotations

import numpy as np

from ..geometry import box_center, decode_box_xyxy
from ..paths import ensure_kimi_api_importable
from ..providers import call_vlm
from .base import Result, build_request_text

ensure_kimi_api_importable()
from kimi_api._image_utils import encode_to_data_url  # noqa: E402

_SYSTEM = (
    "You help a robot seat a new guest. Look at the image and return JSON "
    "with fields: seats, choice.\n"
    "seats — array, one entry per sittable cushion (a 2-cushion sofa = 2 "
    "entries, a single armchair/stool = 1). Each entry: "
    '{"label": "<short identifier with a visual anchor>", '
    '"box_2d": [x1,y1,x2,y2], "occupied": true|false}. '
    "box_2d is the tight bounding box of the SEAT CUSHION (the flat surface "
    "a person sits on, NOT the backrest), normalized 0-1000 over the image "
    "where (0,0) is top-left and (1000,1000) is bottom-right.\n"
    "A cushion is OCCUPIED if a person sits on it or a large object rests on "
    "the cushion fabric; objects on a table/floor/armrest do not occupy it.\n"
    "choice — the label of one entry whose occupied is false (your "
    'recommendation), or "none" if every seat is occupied or none are visible.'
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "seats": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "box_2d": {"type": "array", "items": {"type": "integer"},
                           "minItems": 4, "maxItems": 4},
                "occupied": {"type": "boolean"},
            },
            "required": ["label", "box_2d", "occupied"],
            "additionalProperties": False,
        }},
        "choice": {"type": "string"},
    },
    "required": ["seats", "choice"],
    "additionalProperties": False,
}


def run(img_bgr: np.ndarray, req: dict, provider: str, logger=None) -> Result:
    h, w = img_bgr.shape[:2]
    res = Result(strategy="s1", provider=provider)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_to_data_url(img_bgr)}},
            {"type": "text", "text": build_request_text(req)},
        ]},
    ]
    try:
        parsed, elapsed = call_vlm(provider, messages, schema=_SCHEMA,
                                   schema_name="seat_bbox_select", logger=logger)
    except Exception as exc:  # noqa: BLE001
        res.error = str(exc)
        return res
    res.elapsed_s, res.n_calls, res.raw = elapsed, 1, parsed
    seats = parsed.get("seats", []) or []
    res.visible_seats = seats
    choice = str(parsed.get("choice", "none") or "none")
    res.chosen_label = choice
    if choice.strip().lower() == "none":
        return res
    chosen = next((s for s in seats
                   if str(s.get("label", "")).strip().lower() == choice.strip().lower()), None)
    if chosen is None:
        res.error = f"choice {choice!r} not in seats list"
        return res
    box = decode_box_xyxy(chosen.get("box_2d"), w, h)
    if box is None:
        res.error = "chosen box failed to decode"
        return res
    res.box_xyxy = list(box)
    res.point_xy = list(box_center(box))
    return res
```

- [ ] **Step 2: Smoke-test on one scene**

Run:
```bash
cd src/kimi_api && python -c "
import cv2, json
from seat_bench.strategies.s1_bbox_select import run
img=cv2.imread('seat_bench/dataset/scene_000.jpg')
req=json.load(open('seat_bench/dataset/scene_000.req.json'))
print(run(img, req, 'qwen', logger=print).to_dict())
"
```
Expected: a dict with `box_xyxy` and `point_xy` set (when an empty seat exists); no exception.

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/strategies/s1_bbox_select.py
git commit -m "feat(seat_bench): S1 single-call bbox+select strategy"
```

---

## Task 11: S2 — two-call zoom

**Files:**
- Create: `src/kimi_api/seat_bench/strategies/s2_zoom.py`

- [ ] **Step 1: Implement s2_zoom.py**

`src/kimi_api/seat_bench/strategies/s2_zoom.py`:
```python
"""S2: coarse boxes + select (call 1, reusing S1's contract), then crop the
chosen seat's box with margin and ask a SECOND call to place a precise
point on the cushion within the high-res crop. Crop-space point is mapped
back to full-image coordinates. Targets the 'point on wrong object' error
by giving call 2 far more pixels on the actual seat.
"""

from __future__ import annotations

import numpy as np

from ..geometry import decode_point_yx
from ..paths import ensure_kimi_api_importable
from ..providers import call_vlm
from .base import Result
from .s1_bbox_select import run as s1_run

ensure_kimi_api_importable()
from kimi_api._image_utils import encode_to_data_url  # noqa: E402

_CROP_SYSTEM = (
    "This image is a close-up crop of ONE seat. Return JSON {\"point\": [y, x]} "
    "where the point lands on the cushion fabric (the flat surface a person "
    "sits on, not the backrest, armrest, floor, or any person/object on it). "
    "y and x are integers 0-1000 normalized to THIS crop's dimensions "
    "(y=0 top, x=0 left)."
)
_CROP_SCHEMA = {
    "type": "object",
    "properties": {"point": {"type": "array", "items": {"type": "integer"},
                             "minItems": 2, "maxItems": 2}},
    "required": ["point"], "additionalProperties": False,
}

_MARGIN_FRAC = 0.25  # expand the coarse box by 25% per side before cropping


def run(img_bgr: np.ndarray, req: dict, provider: str, logger=None) -> Result:
    h, w = img_bgr.shape[:2]
    coarse = s1_run(img_bgr, req, provider, logger=logger)
    res = Result(strategy="s2", provider=provider)
    res.visible_seats = coarse.visible_seats
    res.chosen_label = coarse.chosen_label
    res.n_calls = coarse.n_calls
    res.elapsed_s = coarse.elapsed_s
    res.raw = {"coarse": coarse.raw}
    if coarse.error:
        res.error = f"coarse: {coarse.error}"
        return res
    if coarse.box_xyxy is None:        # chose "none" or no decodable box
        return res

    x1, y1, x2, y2 = coarse.box_xyxy
    mx = int((x2 - x1) * _MARGIN_FRAC)
    my = int((y2 - y1) * _MARGIN_FRAC)
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        res.error = "empty crop"
        res.box_xyxy = list(coarse.box_xyxy)
        return res
    ch, cw = crop.shape[:2]

    messages = [
        {"role": "system", "content": _CROP_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_to_data_url(crop)}},
            {"type": "text", "text": "Place the point on the seat cushion."},
        ]},
    ]
    try:
        parsed, elapsed2 = call_vlm(provider, messages, schema=_CROP_SCHEMA,
                                    schema_name="seat_crop_point", logger=logger)
    except Exception as exc:  # noqa: BLE001
        res.error = f"refine: {exc}"
        res.box_xyxy = list(coarse.box_xyxy)
        res.point_xy = list(((x1 + x2) // 2, (y1 + y2) // 2))  # fall back to coarse center
        return res

    res.n_calls += 1
    res.elapsed_s += elapsed2
    res.raw["refine"] = parsed
    pt_crop = decode_point_yx(parsed.get("point"), cw, ch)
    res.box_xyxy = list(coarse.box_xyxy)
    if pt_crop is None:
        res.point_xy = list(((x1 + x2) // 2, (y1 + y2) // 2))
        return res
    res.point_xy = [cx1 + pt_crop[0], cy1 + pt_crop[1]]  # map crop -> full image
    return res
```

- [ ] **Step 2: Smoke-test on one scene**

Run:
```bash
cd src/kimi_api && python -c "
import cv2, json
from seat_bench.strategies.s2_zoom import run
img=cv2.imread('seat_bench/dataset/scene_000.jpg')
req=json.load(open('seat_bench/dataset/scene_000.req.json'))
r=run(img, req, 'gemini', logger=print).to_dict()
print('calls',r['n_calls'],'point',r['point_xy'])
"
```
Expected: `n_calls` = 2 when an empty seat is found, with a `point_xy` mapped back to full-image coords.

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/strategies/s2_zoom.py
git commit -m "feat(seat_bench): S2 two-call zoom strategy"
```

---

## Task 12: S3 — set-of-mark

**Files:**
- Create: `src/kimi_api/seat_bench/strategies/s3_som.py`

> Depends on Task 13 (`world_candidates.py`).

- [ ] **Step 1: Implement s3_som.py**

`src/kimi_api/seat_bench/strategies/s3_som.py`:
```python
"""S3 set-of-mark: detect seat candidates with YOLO-World (open vocab:
chair/sofa/stool/bench/couch/armchair), draw numbered boxes on the image,
and ask the VLM only to PICK a number + occupancy. Removes coordinate
regression from the final decision. Recommendation point = center of the
picked candidate box.

Degraded fallback: if YOLO-World yields < 2 candidates, fall back to S1's
VLM boxes as the marks (logged via res.raw['som_source']='s1_fallback').
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import box_center
from ..paths import ensure_kimi_api_importable
from ..providers import call_vlm
from ..world_candidates import detect_seat_candidates
from .base import Result, build_request_text
from .s1_bbox_select import run as s1_run

ensure_kimi_api_importable()
from kimi_api._image_utils import encode_to_data_url  # noqa: E402

_SYSTEM = (
    "The image has numbered boxes drawn over candidate seats. Return JSON "
    '{"choice": <int or -1>} where choice is the number of the best EMPTY '
    "seat for a new guest, or -1 if every numbered seat is occupied. A seat "
    "is occupied if a person sits on it or a large object rests on the "
    "cushion."
)
_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "integer"}},
    "required": ["choice"], "additionalProperties": False,
}


def _draw_marks(img_bgr: np.ndarray, boxes: list) -> np.ndarray:
    out = img_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(out, str(i), (x1 + 4, max(20, y1 + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    return out


def run(img_bgr: np.ndarray, req: dict, provider: str, logger=None) -> Result:
    res = Result(strategy="s3", provider=provider)
    boxes, det_elapsed = detect_seat_candidates(img_bgr, logger=logger)
    som_source = "yolo_world"

    if len(boxes) < 2:
        # Degrade to S1's VLM boxes as marks.
        s1 = s1_run(img_bgr, req, provider, logger=logger)
        res.elapsed_s += s1.elapsed_s
        res.n_calls += s1.n_calls
        boxes = [tuple(s["box_2d_px"]) for s in _s1_boxes_px(s1, img_bgr)]
        som_source = "s1_fallback"
        if len(boxes) < 1:
            res.error = s1.error or "no candidates from yolo or s1"
            res.raw = {"som_source": som_source}
            return res

    res.elapsed_s += det_elapsed
    marked = _draw_marks(img_bgr, boxes)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_to_data_url(marked)}},
            {"type": "text", "text": build_request_text(req)
                + " Pick the numbered box that is the best empty seat."},
        ]},
    ]
    try:
        parsed, elapsed = call_vlm(provider, messages, schema=_SCHEMA,
                                   schema_name="seat_som", logger=logger)
    except Exception as exc:  # noqa: BLE001
        res.error = str(exc)
        res.raw = {"som_source": som_source, "n_candidates": len(boxes)}
        return res

    res.elapsed_s += elapsed
    res.n_calls += 1
    res.raw = {"som_source": som_source, "n_candidates": len(boxes), "vlm": parsed}
    choice = int(parsed.get("choice", -1))
    if choice < 0 or choice >= len(boxes):
        res.chosen_label = "none"
        return res
    box = boxes[choice]
    res.box_xyxy = list(box)
    res.point_xy = list(box_center(box))
    res.chosen_label = f"candidate_{choice}"
    return res


def _s1_boxes_px(s1: Result, img_bgr: np.ndarray) -> list:
    """Decode S1's normalized seat boxes to pixel boxes for SoM fallback."""
    from ..geometry import decode_box_xyxy
    h, w = img_bgr.shape[:2]
    out = []
    for s in (s1.visible_seats or []):
        box = decode_box_xyxy(s.get("box_2d"), w, h)
        if box is not None:
            out.append({"box_2d_px": list(box)})
    return out
```

- [ ] **Step 2: Smoke-test (after Task 13 is done)**

Run:
```bash
cd src/kimi_api && python -c "
import cv2, json
from seat_bench.strategies.s3_som import run
img=cv2.imread('seat_bench/dataset/scene_000.jpg')
req=json.load(open('seat_bench/dataset/scene_000.req.json'))
r=run(img, req, 'gemini', logger=print).to_dict()
print('source',r['raw'].get('som_source'),'point',r['point_xy'])
"
```
Expected: prints a `som_source` and a `point_xy` (when a seat is picked); no exception.

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/strategies/s3_som.py
git commit -m "feat(seat_bench): S3 set-of-mark strategy"
```

---

## Task 13: YOLO-World seat-candidate detector (vendored)

**Files:**
- Create: `src/kimi_api/seat_bench/world_candidates.py`

> Vendors the device-pinning logic from `object_detection_generalist/world_bbox.py` so the harness stays self-contained inside the seat folder. Multi-term: queries several seat words and merges with NMS.

- [ ] **Step 1: Implement world_candidates.py**

`src/kimi_api/seat_bench/world_candidates.py`:
```python
"""Vendored slim YOLO-World seat detector for S3 candidate generation.

Logic (esp. the CUDA device-pinning of txt_feats/clip_model) is copied from
object_detection_generalist/world_bbox.py — see that file for the rationale
behind re-pinning after every set_classes(). Kept here so seat_bench has no
cross-package import.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

Bbox = Tuple[int, int, int, int]

SEAT_TERMS = ["chair", "sofa", "couch", "stool", "bench", "armchair"]
_WEIGHTS = "yolov8s-worldv2.pt"   # same default as the generalist node
_CONF = 0.05
_IOU = 0.5

_model = None
_device = None


def _get_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    import torch
    from ultralytics import YOLOWorld
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _model = YOLOWorld(_WEIGHTS)
    _model.to(_device)
    return _model, _device


def _set_classes(model, device, classes):
    import torch
    model.set_classes(classes)
    target = torch.device(device)
    for module in (model, getattr(model, "model", None)):
        if module is None:
            continue
        txt = getattr(module, "txt_feats", None)
        if txt is not None and hasattr(txt, "to"):
            module.txt_feats = txt.to(device)
        clip = getattr(module, "clip_model", None)
        if clip is not None and hasattr(clip, "to"):
            module.clip_model = clip.to(device)
            if hasattr(clip, "device"):
                clip.device = target
    model.to(device)


def _nms(boxes: List[Bbox], scores: List[float], iou_thr: float = 0.6) -> List[Bbox]:
    from .geometry import iou
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[Bbox] = []
    used = [False] * len(boxes)
    for idx in order:
        if used[idx]:
            continue
        keep.append(boxes[idx])
        for jdx in order:
            if not used[jdx] and jdx != idx and iou(boxes[idx], boxes[jdx]) > iou_thr:
                used[jdx] = True
        used[idx] = True
    return keep


def detect_seat_candidates(img_bgr: np.ndarray, logger=None) -> tuple[List[Bbox], float]:
    """Return (boxes, elapsed_s): seat-like boxes across SEAT_TERMS, NMS-merged."""
    h, w = img_bgr.shape[:2]
    t0 = time.perf_counter()
    try:
        model, device = _get_model()
        _set_classes(model, device, SEAT_TERMS)
        results = model.predict(img_bgr, device=device, conf=_CONF, iou=_IOU,
                                verbose=False)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger(f"[yolo-world] detect failed: {exc}")
        return [], time.perf_counter() - t0

    boxes: List[Bbox] = []
    scores: List[float] = []
    for r in results or []:
        b = getattr(r, "boxes", None)
        if b is None or b.xyxy is None:
            continue
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy() if b.conf is not None else None
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i].tolist()
            px1 = max(0, min(int(round(x1)), w - 1))
            py1 = max(0, min(int(round(y1)), h - 1))
            px2 = max(0, min(int(round(x2)), w - 1))
            py2 = max(0, min(int(round(y2)), h - 1))
            if px2 <= px1 or py2 <= py1:
                continue
            boxes.append((px1, py1, px2, py2))
            scores.append(float(confs[i]) if confs is not None else 1.0)

    merged = _nms(boxes, scores)
    # Stable left-to-right ordering so the drawn numbers read naturally.
    merged.sort(key=lambda bx: bx[0])
    if logger:
        logger(f"[yolo-world] {len(merged)} seat candidate(s) in "
               f"{(time.perf_counter()-t0)*1000:.0f} ms")
    return merged, time.perf_counter() - t0
```

- [ ] **Step 2: Smoke-test the detector**

Run:
```bash
cd src/kimi_api && python -c "
import cv2
from seat_bench.world_candidates import detect_seat_candidates
img=cv2.imread('seat_bench/dataset/scene_000.jpg')
boxes,el=detect_seat_candidates(img, logger=print)
print(len(boxes),'candidates', round(el,2),'s')
"
```
Expected: prints a candidate count (≥1 on a chair scene) and elapsed; first call pays a one-time model-load/warmup cost. If weights download is blocked, note it — S3 will use its S1 fallback.

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/world_candidates.py
git commit -m "feat(seat_bench): vendored YOLO-World seat-candidate detector for S3"
```

---

## Task 14: Cell runner CLI

**Files:**
- Create: `src/kimi_api/seat_bench/run.py`

- [ ] **Step 1: Implement run.py**

`src/kimi_api/seat_bench/run.py`:
```python
"""Run ONE (strategy, provider) cell over every dataset scene.

Writes results/<strategy>_<provider>/<id>.json (Result + scoring outcome)
and results/<strategy>_<provider>/<id>.jpg (overlay). One subagent runs
one cell; cells are independent so they fan out concurrently.

Usage:
  python -m seat_bench.run --strategy s1 --provider qwen [--ids scene_000 ...] [--limit N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2

from .geometry import draw_overlay
from .paths import DATASET_DIR, RESULTS_DIR
from .score import classify
from .strategies import get_strategy


def _load_scene(sid: str):
    img = cv2.imread(str(DATASET_DIR / f"{sid}.jpg"))
    req = json.load(open(DATASET_DIR / f"{sid}.req.json"))
    gt_path = DATASET_DIR / f"{sid}.gt.json"
    gt = json.load(open(gt_path)) if gt_path.is_file() else None
    return img, req, gt


def _scene_ids() -> list[str]:
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(str(DATASET_DIR / "scene_*.jpg")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--provider", required=True, choices=["gemini", "qwen"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_fn = get_strategy(args.strategy)
    cell = f"{args.strategy}_{args.provider}"
    out_dir = RESULTS_DIR / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.ids or _scene_ids()
    if args.limit:
        ids = ids[: args.limit]

    n_hit = 0
    for sid in ids:
        img, req, gt = _load_scene(sid)
        if img is None:
            print(f"  {sid}: SKIP (no image)")
            continue
        res = run_fn(img, req, args.provider, logger=lambda m: None)
        rec = res.to_dict()
        if gt is not None:
            outcome = classify(rec, gt)
            rec["scoring"] = outcome
            n_hit += 1 if outcome["outcome"] == "hit" else 0
            gt_boxes = [(tuple(s["cushion_bbox"]), s["occupied"]) for s in gt["seats"]]
            hit = outcome["outcome"] == "hit"
        else:
            gt_boxes, hit = None, None
        (out_dir / f"{sid}.json").write_text(json.dumps(rec, indent=2))
        overlay = draw_overlay(
            img,
            point=tuple(rec["point_xy"]) if rec["point_xy"] else None,
            box=tuple(rec["box_xyxy"]) if rec["box_xyxy"] else None,
            gt_boxes=gt_boxes,
            label=f"{cell}:{rec['chosen_label']}",
            hit=hit,
        )
        cv2.imwrite(str(out_dir / f"{sid}.jpg"), overlay)
        tag = rec.get("scoring", {}).get("outcome", "n/a")
        print(f"  {sid}: {tag} ({rec['n_calls']} calls, {rec['elapsed_s']:.1f}s)"
              + (f" ERR {rec['error']}" if rec.get("error") else ""))

    print(f"[{cell}] done: {len(ids)} scenes, {n_hit} hits -> {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify a 2-scene cell end-to-end**

Run: `cd src/kimi_api && python -m seat_bench.run --strategy s0 --provider gemini --limit 2`
Expected: two per-scene lines with an outcome each, a summary line, and two `.json`+`.jpg` files in `seat_bench/results/s0_gemini/`.

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/run.py
git commit -m "feat(seat_bench): per-cell runner CLI with overlay + scoring"
```

---

## Task 15: Aggregator CLI

**Files:**
- Create: `src/kimi_api/seat_bench/aggregate.py`

- [ ] **Step 1: Implement aggregate.py**

`src/kimi_api/seat_bench/aggregate.py`:
```python
"""Aggregate results/<cell>/*.json into report.md + sheets/<cell>.jpg.

No API calls. Reads every cell dir, recomputes per-cell aggregates, writes
a scoreboard sorted by hit_rate, and tiles each cell's overlays into a
contact sheet for eyeballing misses.
"""

from __future__ import annotations

import glob
import json
import math
import os

import cv2
import numpy as np

from .paths import REPORT_PATH, RESULTS_DIR, SHEETS_DIR
from .score import aggregate


def _cells() -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(str(RESULTS_DIR / "*"))
                  if os.path.isdir(p))


def _cell_rows(cell: str) -> list[dict]:
    rows = []
    for jf in sorted(glob.glob(str(RESULTS_DIR / cell / "*.json"))):
        rec = json.load(open(jf))
        sc = rec.get("scoring")
        if sc:
            rows.append({**sc, "elapsed_s": rec.get("elapsed_s", 0.0),
                         "n_calls": rec.get("n_calls", 0)})
    return rows


def _contact_sheet(cell: str, cols: int = 5) -> None:
    imgs = [cv2.imread(p) for p in sorted(glob.glob(str(RESULTS_DIR / cell / "*.jpg")))]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return
    th, tw = 240, 360
    tiles = [cv2.resize(im, (tw, th)) for im in imgs]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEETS_DIR / f"{cell}.jpg"), sheet)


def main():
    lines = ["# Seat-Recommendation Strategy Benchmark — Results", ""]
    lines += ["| cell | n | hit_rate | hits | wrong_seat | miss | false_none | "
              "correct_reject | mean_s | mean_calls |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    table = []
    for cell in _cells():
        rows = _cell_rows(cell)
        if not rows:
            continue
        agg = aggregate(rows)
        mean_s = sum(r["elapsed_s"] for r in rows) / len(rows)
        mean_calls = sum(r["n_calls"] for r in rows) / len(rows)
        table.append((agg["hit_rate"], cell, agg, mean_s, mean_calls))
        _contact_sheet(cell)
    for hit_rate, cell, agg, mean_s, mean_calls in sorted(table, reverse=True):
        lines.append(
            f"| {cell} | {agg['n']} | {hit_rate:.0%} | {agg['hits']} | "
            f"{agg['wrong_seat']} | {agg['miss']} | {agg['false_none']} | "
            f"{agg['correct_reject']} | {mean_s:.1f} | {mean_calls:.1f} |")
    lines += ["", "Contact sheets per cell under `sheets/`. Green box = empty "
              "GT cushion, red = occupied, cyan = predicted box, magenta dot = "
              "predicted point.", ""]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH} and {SHEETS_DIR}/*.jpg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify aggregation on whatever cells exist**

Run: `cd src/kimi_api && python -m seat_bench.aggregate`
Expected: writes `seat_bench/report.md` with a scoreboard row for `s0_gemini` (from Task 14's smoke run) and a contact sheet.

- [ ] **Step 3: Commit**

```bash
git add src/kimi_api/seat_bench/aggregate.py
git commit -m "feat(seat_bench): scoreboard + contact-sheet aggregator"
```

---

## Task 16: Run the full grid via concurrent subagents, then aggregate

> This is the benchmark execution. **Dispatch one subagent per grid cell concurrently** (the user asked for concurrent fan-out). 4 strategies × 2 providers = **8 cells**. Each subagent is independent (writes to its own `results/<cell>/` dir), so there is no shared-state contention.

- [ ] **Step 1: Dispatch 8 concurrent cell-runner subagents**

Issue 8 `Agent` calls **in a single message** (so they run concurrently), one per cell. Each subagent's prompt:

> "Run the seat_bench benchmark cell `<STRATEGY>`/`<PROVIDER>`. Steps: (1) `cd /home/tinker/tk25_ws && source src/tk26_vision/.venv-vision-main/bin/activate`; (2) `cd src/tk26_vision/src/kimi_api`; (3) `python -m seat_bench.run --strategy <STRATEGY> --provider <PROVIDER>`. Report the final summary line (`[cell] done: N scenes, H hits`) and any per-scene `ERR` lines. Do not edit any code; if the run errors, capture the traceback and report it verbatim."

Cells: `s0/gemini, s0/qwen, s1/gemini, s1/qwen, s2/gemini, s2/qwen, s3/gemini, s3/qwen`.

> **Rate-limit note:** if many cells fail with HTTP 429, re-dispatch the failed cells in two waves (gemini cells, then qwen cells) instead of all 8 at once. S2 makes 2 calls/scene and S3 makes 1 call/scene + local YOLO, so they are the slowest cells.

- [ ] **Step 2: Aggregate**

Run: `cd src/kimi_api && python -m seat_bench.aggregate`
Expected: `report.md` now has 8 rows sorted by hit_rate, and `sheets/` has 8 contact sheets.

- [ ] **Step 3: Review the scoreboard and contact sheets**

Read `seat_bench/report.md` and each `sheets/<cell>.jpg`. Sanity checks:
- S0 hit-rate should reproduce the poor localization seen in the logged overlays (low).
- Compare S1/S2/S3 against S0 per provider. Note which strategy×provider wins on hit_rate, and the latency/call-count cost.
- Spot-check the contact sheets for systematic failure patterns (e.g., S1 boxes spanning the gap between cushions; S3 YOLO-World missing stools).

- [ ] **Step 4: Write the findings summary + commit results**

Append a short "Findings" section to `report.md` (top 1–2 strategies, per-provider notes, recommendation for the production-rewrite spec). Commit:
```bash
git add src/kimi_api/seat_bench/report.md
git commit -m "data(seat_bench): full 8-cell benchmark results + findings"
```
(Per-scene `results/` JPEGs/JSON and `sheets/` are gitignored — the committed artifacts are `report.md` + the GT under `dataset/`.)

- [ ] **Step 5: Hand off to the production-rewrite spec**

The winning strategy + numbers feed a **separate brainstorming/spec** for porting into `_seat_vlm.py` + `seat_recommend_bbox.py` with Gemini-first/Qwen-fallback. That is explicitly out of scope here.

---

## Self-review notes (for the plan author)

- **Spec coverage:** harness location inside the seat folder (`src/kimi_api/seat_bench/`) ✓ (Tasks 1–2); four strategies S0–S3 ✓ (Tasks 9–12); Gemini-via-OpenRouter + Qwen3-via-DashScope ✓ (Task 4); S3 YOLO-World open-vocab incl. sofas/stools ✓ (Task 13); hand-annotated ground truth ✓ (Task 6); 2D-only scoring (no depth in logs) ✓ (Task 7 + Background); strategies built/tested in parallel ✓ (Tasks 9–13 marked independent); benchmark run via concurrent subagents ✓ (Task 16); production rewrite deferred ✓ (out-of-scope + Task 16 Step 5).
- **Type consistency:** `Result` fields (`point_xy`, `box_xyxy`, `chosen_label`, `visible_seats`, `n_calls`, `elapsed_s`, `raw`, `error`) are used consistently across s0–s3, run.py, aggregate.py. `decode_box_xyxy`/`decode_point_yx`/`box_center`/`point_in_box`/`iou`/`draw_overlay` signatures match call sites. `classify()`/`aggregate()` output keys match aggregate.py's reads.
- **Known soft spots to watch during execution:** (a) `paths.find_vision_log()` must resolve to `tk25_ws/vision_log` — verified in Task 2 Step 2; (b) DashScope key is under the typo'd `DASHCOPE_API_KEY` — handled by `_env.require_dashscope_api_key()`; (c) YOLO-World weight download may be blocked offline — S3 degrades to the S1-box fallback, logged.
```
