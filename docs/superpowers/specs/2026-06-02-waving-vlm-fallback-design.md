# VLM fallback for waving-person detection — design

**Date:** 2026-06-02
**Status:** Approved (brainstorming) — pending implementation plan
**Package:** `tk_vision_specialized` (+ interface change in `tinker_vision_msgs_26`; self-contained — no new inter-package dependency)
**Node:** `detect_waving_persons` (`waving_person_server.py`)

## Problem

`detect_waving_persons` finds waving people with YOLO11m-seg (person boxes) →
MediaPipe Pose per crop → `is_waving()` heuristic (wrist-above-nose /
wrist-above-elbow). The heuristic misses real wavers under partial occlusion,
truncation, atypical poses, or when MediaPipe fails to localize keypoints in a
small/own-angled crop. There is no recovery path: a missed waver is simply
absent from `waving_persons[]`.

We want a **VLM fallback**: when the heuristic finds fewer wavers than the
caller expects (a threshold carried in the request), call an open-vocabulary
VLM (DashScope Qwen3-VL primary, OpenRouter Gemini fallback) to find the
wavers MediaPipe missed, and merge them into the response.

## Goals / non-goals

**Goals**
- Add a caller-supplied count threshold to the request; trigger the VLM only
  when MediaPipe under-delivers against it.
- Augment (not replace) MediaPipe results — keep the depth-accurate MediaPipe
  centroids, add the missed wavers.
- Every VLM-found waver still returns a depth-derived 3D `PointStamped`,
  consistent with the existing response contract.
- Reuse the proven `_seat_bbox_vlm.py` provider-chain *pattern*, but implement
  it with the package's **existing decoupled VLM-client convention**
  (`qwen_match_vlm.py` / `vlm_match_client.py` / `_vlm_common.py`): `os.environ`
  key resolution + base-URL constants + `_vlm_common.encode_data_url`. No
  `kimi_api` import — the waving feature stays self-contained.

**Non-goals**
- Replacing the MediaPipe heuristic (it stays the primary, fast path).
- Re-judging each YOLO crop individually with the VLM (rejected: misses
  YOLO-undetected people, N calls per frame).
- Changing the camera/sync/TF machinery.
- Real-time / streaming waving detection (the VLM path is a per-request,
  multi-second augmentation).

## Contract change

`tinker_vision_msgs_26/srv/DetectWaving.srv` gains one request field:

```
float32 threshold_meters
string  target_frame
int32   min_waving_persons    # NEW: expected number of wavers. <=0 => VLM fallback disabled for this call.
---
int32 status
string error_msg
geometry_msgs/PointStamped[] waving_persons

sensor_msgs/Image rgb_image
sensor_msgs/Image depth_image
sensor_msgs/Image[] segments
```

- Default `0` (unset field in ROS) ⇒ no VLM ever ⇒ **fully backward
  compatible** with existing callers.
- Trigger condition: `enable_vlm_fallback AND chain_non_empty AND
  len(mediapipe_wavers_after_depth_filter) < min_waving_persons`.
- The count compared is the **post-depth-filter** MediaPipe waver count (the
  usable wavers), computed before the closest-first sort.

Rebuilding `tinker_vision_msgs_26` rebuilds dependents; no other interface
field changes.

## Architecture

```
detect_waving_callback(request, response)
  ├─ existing: copy frame/depth/intrinsics, TF snapshot, depth→XYZ grid, YOLO
  ├─ existing: per YOLO person → MediaPipe → is_waving → centroid (mask∩depth)
  │            (NOW ALSO: retain every person's (bbox, seg_mask) for VLM reuse)
  ├─ NEW: if trigger → _vlm_augment(...)
  │        ├─ request_waving_persons_chain(rgb)          # qwen → gemini
  │        ├─ per VLM box: dedup vs existing wavers (IoU / center-in-box)
  │        ├─ per fresh box: centroid_from_box(...)     # mask reuse | box-center
  │        ├─ depth filter + target_frame transform (same as MediaPipe)
  │        └─ append all surviving VLM wavers
  ├─ existing: closest-first sort over combined list
  ├─ existing: debug overlay (VLM wavers drawn distinctly) + vision log
  └─ existing: transform + status/error_msg
```

### Decoupling: reuse existing in-package helpers (no new env module)

The package already ships a self-contained, kimi_api-free VLM convention used by
the match/judge clients — the waving client follows it verbatim, so **no
`_vlm_env.py` is created** and **nothing imports `kimi_api`**:
- Image encoding: `from ._vlm_common import encode_data_url, strip_fences`
  (`_vlm_common.py`, already present, pure cv2/numpy).
- `.env` discovery: `from dotenv import load_dotenv; load_dotenv(override=False)`
  at module import (mirrors `vlm_match_client.py:32`).
- Key resolution from `os.environ` only (so pytest `monkeypatch.delenv` works):
  Qwen via `_QWEN_KEY_NAMES = ('DASHCOPE_API_KEY', 'DASHSCOPE_API_KEY')`, Gemini
  via `OPENROUTER_API_KEY` — identical names/order to `vlm_match_client.py`.
- Base URLs as module constants:
  `_QWEN_DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'`,
  `_GEMINI_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'`.
- `decode_box_xyxy(box_2d, w, h)` — 0–1000 normalized → **clamped** xyxy pixels
  (own copy; clamps to image bounds since the box drives depth sampling).

**Note on `package.xml`:** the package *already* carries
`<exec_depend>kimi_api</exec_depend>` for the unrelated `placing_vlm.py`. We do
**not** add to or rely on it for waving, and we do not remove it (it still serves
`placing_location_server`). The waving path introduces zero new coupling.

### New module: `tk_vision_specialized/tk_vision_specialized/_waving_vlm.py`

Mirrors the *control flow* of `_seat_bbox_vlm.py` (single-call → chain,
strict-schema → json_object fallback, errors-only fallthrough) while using the
in-package decoupled helpers above — function-based like `qwen_match_vlm.py`.

```python
@dataclass
class WavingVlmResult:
    boxes: list = field(default_factory=list)   # [[x1,y1,x2,y2] px, ...] whole-person boxes
    provider: str = ""
    elapsed_s: float = 0.0
    error: Optional[str] = None                  # soft error -> chain fallthrough

def select_boxes(parsed: dict, w: int, h: int) -> WavingVlmResult: ...   # pure parser
def request_waving_persons(rgb_bgr, *, provider, model, timeout_s,
                           max_retries, logger) -> WavingVlmResult: ...
def request_waving_persons_chain(rgb_bgr, *, provider_models, timeout_s,
                                 max_retries, logger) -> WavingVlmResult: ...
```

- OpenAI-compatible client (`from openai import OpenAI`), strict
  `json_schema` → `json_object` fallback on schema rejection, per-attempt
  retry — same control flow as `request_seat_bbox`.
- Chain tries `(provider, model)` pairs in order; **errors-only
  fallthrough**. A clean empty result (`boxes=[]`, no error — "nobody else is
  waving") is a legitimate terminal answer and does **not** trigger fallback.
- Keys/base-URLs from `os.environ` + module constants; image encoding from
  `._vlm_common`; `decode_box_xyxy` local to this module — no `kimi_api` import.

**Prompt (system):** identify *every* person actively waving — a raised or
waving hand/arm at or above shoulder/head height to get attention; exclude
arms-down/crossed. Return whole-**person** boxes (so they overlap YOLO person
masks), `box_2d` normalized 0–1000, `(0,0)` top-left.

**Schema (strict):**
```json
{"type":"object","properties":{
  "persons":{"type":"array","items":{"type":"object","properties":{
    "box_2d":{"type":"array","items":{"type":"integer"},"minItems":4,"maxItems":4},
    "waving":{"type":"boolean"}},
    "required":["box_2d","waving"],"additionalProperties":false}}},
  "required":["persons"],"additionalProperties":false}
```
`select_boxes` keeps entries with `waving=true` and a decodable box.

### Server changes (`waving_person_server.py`)

**Retain all person masks.** The MediaPipe loop currently stores `seg_mask`
only for wavers. Add a parallel list `person_records = [(x1,y1,x2,y2,
seg_mask_or_None), ...]` for **every** YOLO person, so a VLM box can reuse the
mask of a person MediaPipe judged "still."

**`_resolve_provider_chain()` — non-fatal variant.** Unlike
`seat_recommend_bbox` (where the VLM is mandatory and a missing primary key
raises at init), here the VLM is an optional augmentation:
- Build `[(provider, model)]` from `vlm_provider` then `vlm_fallback_provider`,
  including only providers whose key is present.
- If **no** provider has a key → log one warning, set chain `= []`. The node
  starts normally and serves MediaPipe-only; the trigger condition's
  `chain_non_empty` is false so the VLM is never attempted.
- `enable_vlm_fallback=false` ⇒ chain forced empty (hard kill-switch).

### New module: `tk_vision_specialized/tk_vision_specialized/_waving_geometry.py` (pure, ROS-free)

The box→3D and dedup logic lives in a **pure numpy module** — no ROS, no
network — so it is unit-testable without spinning up a `Node`. The server imports
`box_iou`, `is_duplicate_box`, and `centroid_from_box` from it.

**`centroid_from_box(points, validmask, box_xyxy, person_records)`** returns
`(centroid_xyz, used_mask)` or `None`:
1. **Mask reuse** — if `box_xyxy` overlaps a `person_records` seg-mask
   (box-vs-mask-bbox IoU over a small threshold), use `mask ∩ validmask` →
   `mean(XY)`, `median(Z)`. Identical math to the MediaPipe waver path. The
   `used_mask` is returned so the caller can log it.
2. **Box-center fallback** — else gather valid `points` inside `box_xyxy`; if
   ≥10 px → `mean(XY)`, `median(Z)`; else expand the box once (×1.5) and retry;
   else return `None` (skip this VLM box).

**Dedup.** `box_iou(a, b)` + `is_duplicate_box(box, existing, *, iou_thresh)`.
A VLM box is a duplicate if IoU ≥ `vlm_dedup_iou` (default 0.3) with, or its
center lies inside, any already-accepted waver box (MediaPipe or earlier VLM).
Dedup runs against the growing accepted-box list so VLM boxes don't duplicate
each other.

**Accept policy: add all.** Every deduped VLM waver that yields a centroid and
passes the depth filter is appended. The threshold is the *trigger* only, not
a cap (e.g. `min=1`, MediaPipe `0`, VLM finds `3` → return `3`). Best recall,
simplest logic.

**Depth filter + transform.** VLM wavers pass through the same
`threshold_meters` test and (via the start-of-callback TF snapshot, valid for
the whole call since the head is held still) the same `target_frame` transform
as MediaPipe wavers.

**Debug overlay + logging.** VLM wavers are drawn in a distinct color labeled
`waving (vlm)` and added to `all_person_annotations` so the published debug
frame shows them. Vision-log detections tag `cls_name='waving_person_vlm'`;
`request_ctx` gains `min_waving_persons`; `extras` gains `n_vlm_added` and
`vlm_provider`.

**Pre-existing bug fix (in-scope, the callback we edit).** Line 592
`if self.show_window and self._frame_queue is not None:` references
`self._frame_queue`, which is never initialized → latent `AttributeError` on
the success path when `show_window=true`. Guard defensively with
`getattr(self, '_frame_queue', None)` (one line; no viewer rearchitecture).

### Parameters (new)

| Param | Default | Meaning |
|---|---|---|
| `enable_vlm_fallback` | `true` | Global kill-switch. `false` ⇒ never call the VLM. |
| `vlm_provider` | `'qwen'` | Primary provider (`qwen`\|`gemini`). |
| `vlm_fallback_provider` | `'gemini'` | Secondary; `''` disables fallback. |
| `vlm_model_qwen` | `'qwen3-vl-plus'` | Qwen model id (DashScope). |
| `vlm_model_gemini` | `'google/gemini-2.5-pro'` | Gemini model id (OpenRouter). |
| `vlm_timeout_s` | `20.0` | Per-call timeout. |
| `vlm_max_retries` | `3` | Per-provider retries. |
| `vlm_dedup_iou` | `0.3` | IoU threshold for VLM-vs-existing dedup. |

Keys: `qwen` ⇒ `DASHSCOPE_API_KEY` (or legacy `DASHCOPE_API_KEY`); `gemini` ⇒
`OPENROUTER_API_KEY` — resolved directly from `os.environ` in `_waving_vlm.py`,
matching `vlm_match_client.py`'s names/order (no `kimi_api` import).

## Error handling

- VLM unreachable / all providers fail / parse exhausted → throttled warning,
  return the MediaPipe-only result. **VLM failure never errors the service.**
- Missing keys → handled at init (chain dropped/empty); never a runtime crash.
- `status`: `0` if any waver from either source, `1` if none, `-1` only on the
  existing fatal paths (no image, depth decode failure, TF failure).
- Latency: when triggered, adds ~5–20 s (synchronous, same profile as
  `seat_recommend_bbox`). Caller holds the head still during the call.

## Data flow (VLM-found waver)

```
VLM box_2d (0-1000) ─decode→ xyxy px ─dedup→ fresh? ─centroid_from_box→ xyz (cam frame)
   ─threshold_meters filter→ keep? ─do_transform_point(snapshot)→ PointStamped(target_frame)
   → append to waving_persons → closest-first sort → response
```

## Testing

- **Pure unit (no network):** `select_boxes` (well-formed / malformed / mixed
  `waving` flags / undecodable boxes); `box_iou` + dedup; `centroid_from_box`
  over a synthetic `points`/`validmask` grid (mask-reuse tier, box-center tier,
  sparse-skip).
- **Provider fallthrough:** mirror
  `object_detection_generalist/test/test_vlm_bbox_fallback.py` — monkeypatch the
  OpenAI client so provider 1 raises and provider 2 returns boxes; assert the
  chain returns provider 2's result and that a clean empty (no error) does
  **not** fall through.
- **Node startup:** node constructs with **no** API keys and
  `enable_vlm_fallback=true` without raising (chain empty, MediaPipe-only).
- **Integration (T-suite):** existing `detect_waving` smoke unchanged when
  `min_waving_persons=0`.
- **Manual T4:** occlusion/truncation scene where MediaPipe misses a real
  waver and `min_waving_persons=1` → confirm the VLM augments and the centroid
  is plausible.

## Files touched

| File | Change |
|---|---|
| `tinker_vision_msgs_26/srv/DetectWaving.srv` | + `int32 min_waving_persons` |
| `tk_vision_specialized/.../_waving_vlm.py` | **new** VLM client + chain (uses `_vlm_common`, `os.environ`; no `kimi_api`) |
| `tk_vision_specialized/.../_waving_geometry.py` | **new** pure ROS-free box/depth helpers (`box_iou`, `is_duplicate_box`, `centroid_from_box`) |
| `tk_vision_specialized/.../waving_person_server.py` | retain person masks; trigger; `_vlm_augment`; `_resolve_provider_chain`; params; overlay/log tags; `_frame_queue` guard (geometry/dedup imported from `_waving_geometry`) |
| `tk_vision_specialized/test/test_waving_vlm.py` | **new** VLM decoder/key/chain/fallthrough tests |
| `tk_vision_specialized/test/test_waving_geometry.py` | **new** pure box/depth unit tests |
| `CLAUDE.md` (tk26_vision) | document the new params + fallback behavior |

Unchanged: `package.xml` (no new dependency — `kimi_api` already present for
`placing_vlm`, untouched), `requirements.txt` (`openai`/`python-dotenv` already
listed), `setup.py` (no new entry point), `_vlm_common.py` (reused as-is).

## Open decisions (resolved during brainstorming)

- Fallback role: **augment** (keep MediaPipe, add missed) — not replace, not
  per-crop re-judge.
- VLM→3D: **mask reuse if overlapping, else box-center robust depth.**
- Providers: **Qwen3-VL primary → Gemini fallback**, errors-only chain.
- Threshold carrier: **new `int32 min_waving_persons`, default 0 = off**, plus
  `enable_vlm_fallback` ROS kill-switch.
- Accept count: **add all deduped VLM wavers** (threshold is trigger, not cap).
- `_frame_queue` latent bug: **fix defensively** (in the callback we edit).
- Package coupling: **decoupled** — follow the package's existing kimi_api-free
  VLM convention (`_vlm_common` + `os.environ` keys), no new env module, no
  `kimi_api` import on the waving path.
