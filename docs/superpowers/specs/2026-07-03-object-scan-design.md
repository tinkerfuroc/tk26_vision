# object_scan — batched labels-only VLM scanner

**Date:** 2026-07-03
**Status:** Design — approved for planning
**Owner:** Cindy
**Packages touched:** `tk26_vision/kimi_api`, `tk26_vision/tinker_vision_msgs_26`, `tk25_decision/behavior_tree`

## Problem

`pick_and_place_rulebook.py` scans the dining table with `BtNode_ScanForGeneralist`
(`/object_detection_generalist`, `force_vlm_sam=True`). The scan prompt is the full
RoboCup@Home Incheon 2026 Known Objects list — a **32-class** `" . "`-joined string
(`constants.json:table_scan_prompt`). Asking Gemini for bounding boxes over all 32
classes in a **single** VLM call reliably misses objects: the model returns a partial
set, so the robot's inventory/announcement is incomplete.

We want a scanner that recognizes **every vocabulary class present in the scene** by
**splitting the vocabulary into batches** and issuing one VLM call per batch, then
unioning the results. The task is pure recognition — "which of these classes are on the
table" — so the scanner returns **labels only**: no bounding boxes, masks, depth, or 3D
centroids, and no post-processing. (Decision recorded 2026-07-03: labels only, direct
VLM, no post-process.)

## Non-goals

- **No geometry.** No SAM, no depth, no centroids, no TF. Grasp-oriented single-item
  re-detects that need 3D (`_breakfastItem` re-detect, `_reDetectActive`) **stay on the
  generalist** and are out of scope.
- **No instance counting.** Output is the *set of classes* present, deduped. Two shirts
  → one `"shirt"`. ("recognize every single object belonging to the vocabulary" is read
  as every vocabulary *class* present, not every physical instance.)
- **Not a replacement for** `object_match_all_server` (reference-image dataset matcher in
  `tk_vision_specialized`). That node is prior art we mirror in *shape* (batched
  Gemini/Qwen chain) but do not reuse — it matches image datasets, not a text vocabulary,
  and carries a heavier match+judge+SAM pipeline than this task needs.

## Architecture

```
pick_and_place_rulebook.py
  phaseTableScan / phaseTableCleanup
        │  _scanForObjectScan(vocab=TABLE_SCAN_PROMPT.split(" . "))
        ▼
  BtNode_ObjectScan (ServiceHandler, tk25_decision)
        │  ObjectScan.Request{camera, vocabulary[]}
        ▼
  /object_scan   (kimi_api/object_scan.py, tk26_vision)
        │  latest color frame → data URL
        │  split vocabulary into batches of `batch_size`
        │  ThreadPoolExecutor(max_workers): one call per batch
        ▼
  _scan_vlm.request_scan_labels_chain   (Gemini → Qwen, errors-only fallthrough)
        │  JSON list of visible candidates, hallucinations dropped
        ▼
  union(found_labels) in vocab order  →  ObjectScan.Response{status, found_labels[]}
        ▼
  BtNode_ObjectScan writes SimpleNamespace(status, objects=[{cls, segment=None}, …])
        ▼
  BtNode_WriteFoundItems (obj.cls) / BtNode_BuildInventory (o.cls, o.segment)  — unchanged
```

## Components

### 1. `tinker_vision_msgs_26/srv/ObjectScan.srv` (new)

```
# Direct-VLM, labels-only scene scan over a candidate vocabulary.

# camera in ['orbbec', 'realsense'] (substring match; defaults to orbbec).
string camera

# Candidate class names to look for. The service splits this into batches
# and returns the subset actually visible in the scene.
string[] vocabulary
---
std_msgs/Header header
# 0 = ok (found_labels may be empty on a genuinely empty scene).
# 1 = failure (no camera frame, or every VLM batch failed) — see error_msg.
int32 status
string error_msg
# Subset of `vocabulary` present in the scene. Deduped, preserving vocabulary
# order. No geometry — labels only.
string[] found_labels
```

Register in `tinker_vision_msgs_26/CMakeLists.txt` alongside the other `srv/` entries.
Append `build/tinker_vision_msgs_26/rosidl_generator_py` is already on the IDE path — no
`.vscode` change needed (the package is already listed).

### 2. `kimi_api/_scan_vlm.py` (new)

Mirrors `_match_vlm.py` line-for-line in structure (provider chain, retry, fence-strip,
`ast.literal_eval`). Public API:

```python
class ScanVlmError(RuntimeError): ...

@dataclass
class ScanVlmResult:
    labels: list          # validated subset of candidates (original casing)
    provider: str
    elapsed_s: float

def request_scan_labels(image_url, candidates, *, provider, model,
                        qwen_api_backend='dashscope', timeout_s=20.0,
                        max_retries=3, logger=None) -> ScanVlmResult
def request_scan_labels_chain(image_url, candidates, *, provider_models,
                              qwen_api_backend='dashscope', timeout_s=20.0,
                              max_retries=3, logger=None) -> ScanVlmResult
```

Provider resolution reuses `_env.resolve_qwen_target` / `require_api_key` / `base_url`
exactly as `_match_vlm` does. Message shape: system prompt + user content
`[{image_url}, {text: candidate list}]`.

**Prompt (system):** "You are a visual object detector. You are given ONE image of a scene
and a list of candidate object names. Return a JSON list containing exactly the candidate
names — copied verbatim from the provided list — that are actually visible in the image.
Include a name only if you are confident the object is present. Never include a name that
is not in the provided list. If none are present, return `[]`. Output ONLY the JSON list."

**Validation (`_validate_labels`):** parse → must be a `list`; keep each element that
matches a candidate case-insensitively (return the candidate's original casing); drop
anything else (hallucinated / not-in-vocab). The **empty list is a valid terminal answer**
("none of these here") and is returned as success — it does **not** trigger fallback.
Only an API exception or a structurally-unparseable response (not a list) consumes a
retry / falls through to the next provider. This matches the kimi_api chain contract
("a legitimate answer, including empty, is terminal; only errors fall through").

### 3. `kimi_api/object_scan.py` (new node) → service `/object_scan`

Node class `ObjectScanServer(Node)`, `main()` calls `load_env()` then spins under a
`MultiThreadedExecutor` (concurrent per-batch OpenAI calls block executor threads — same
requirement as `object_match_all_server` / `generalist_node`).

**Camera intake** — direct subscription, no detection-service dependency (matches
`feature_recognition`):
- orbbec color: `/camera/color/image_raw` (param `orbbec_image_topic`)
- realsense color: `/camera/xarm_camera/color/image_raw` (param `realsense_image_topic`)
- keep the latest `Image` per camera under a lock; `BEST_EFFORT`, `KEEP_LAST` depth 1.

**Callback (`MutuallyExclusiveCallbackGroup` on the service):**
1. select camera (`orbbec` default); if no frame within `img_sync_thres_s`, return
   `status=1, error_msg="No <camera> frame available"`.
2. `cv_bridge` → BGR → `_image_utils.encode_to_data_url`.
3. if `vocabulary` empty → `status=1, error_msg="empty vocabulary"`.
4. split `vocabulary` into batches of `batch_size` (preserve order).
5. `ThreadPoolExecutor(max_workers)` — one `request_scan_labels_chain(image_url, batch, …)`
   per batch. A batch that raises `ScanVlmError` is logged and contributes no labels
   (does not fail the whole scan) — tracked in a `batches_fail` counter.
6. union all batch labels, dedup, order by first appearance in `vocabulary`.
7. `status = 0` if **at least one batch succeeded** (even with zero labels found);
   `status = 1, error_msg="all N VLM batches failed: …"` if **every** batch raised.
8. write a vision-log debug artifact (frame + per-batch prompt/response JSON) via the
   established `VisionLogger` convention — do not strip; humans read these.

**ROS params** (model selection copied from `object_detection_generalist`):

| Param | Default | Notes |
|---|---|---|
| `service_name` | `object_scan` | |
| `llm_model` | `default_flash_model()` = `google/gemini-2.5-flash` | primary Gemini; flash matches the generalist's `vlm_model` and is fast enough for N batches |
| `vlm_fallback_provider` | `qwen` | `''` disables; dropped with a warning if `DASHSCOPE_API_KEY` absent (no crash) |
| `scan_model_qwen` | `''` | `''` → DashScope default `qwen3-vl-plus` via `resolve_qwen_target` |
| `qwen_api_backend` | `dashscope` | |
| `batch_size` | `8` | **tune later** per operator note; controls classes-per-VLM-call |
| `max_workers` | `4` | concurrent batches |
| `vlm_timeout_s` | `20.0` | per provider call |
| `vlm_max_retries` | `3` | per provider |
| `orbbec_image_topic` | `/camera/color/image_raw` | |
| `realsense_image_topic` | `/camera/xarm_camera/color/image_raw` | |
| `img_sync_thres_s` | `1.0` | max frame age |
| `vision_logging_enabled` | `true` | shared session dir convention |
| `vision_log_folder` | `vision_log` | |

Provider chain built at init like `grocery_categorize._resolve_categorize_provider_chain`:
`[('gemini', llm_model)]`, then append `('qwen', resolved)` if `vlm_fallback_provider ==
'qwen'` and the DashScope key resolves; `require_api_key()` at init fails fast on a missing
`OPENROUTER_API_KEY`.

**Entry point** in `kimi_api/setup.py`: `'object_scan = kimi_api.object_scan:main'`.

### 4. `TemplateNodes/Vision.py` → `BtNode_ObjectScan` (new)

`ServiceHandler` subclass modeled on `BtNode_ScanForGeneralist`:

```python
BtNode_ObjectScan(name, bb_key, vocabulary: list[str],
                  service_name="object_scan", use_orbbec=True)
```

- `initialise()`: build `ObjectScan.Request{camera, vocabulary}`, `call_service_async`.
- `update()`: on `response.done()` and `status==0`, wrap the labels into a duck-typed
  result the existing consumers accept and write it to `bb_key`:

  ```python
  from types import SimpleNamespace
  result = SimpleNamespace(
      status=0,
      objects=[SimpleNamespace(cls=lbl, segment=None) for lbl in resp.found_labels],
  )
  self.bb_write_client.set(self.bb_key, result, overwrite=True)
  ```

  This is the **only adaptation layer** — it keeps `BtNode_WriteFoundItems` (`obj.cls`)
  and `BtNode_BuildInventory` (`o.cls`, `o.segment`) working **unchanged**. The service
  stays labels-only; the decision-layer glue packs them.
- `status != 0` → `FAILURE` (retried by the `_scanForObjectScan` wrapper).
- **Mock:** write `SimpleNamespace(status=0, objects=[])`. `BtNode_BuildInventory` already
  seeds a canned queue when the upstream scan is empty under mock, so offline trees run.

Import `ObjectScan` from `tinker_vision_msgs_26.srv` in `messages.py`'s conditional block
(and add a `mock_messages.py` stub) so the BT still imports when the msg pkg is absent.

### 5. `mock_config.json`

Add under `mock_mode.subsystems.vision.nodes`:
```json
"BtNode_ObjectScan": "KEYPRESS"
```
(`"IMMEDIATE"` is the alternative for unattended full-mock offline runs.)

### 6. `pick_and_place_rulebook.py` wiring

Add a helper next to `_reDetectActive`:

```python
from .config import TABLE_SCAN_PROMPT  # already imported

def _scanForObjectScan(name, bb_key, vocabulary=None, use_orbbec=True, retries=3):
    vocab = vocabulary or [c.strip() for c in TABLE_SCAN_PROMPT.split(" . ") if c.strip()]
    return py_trees.decorators.Retry(
        name="retry object scan",
        child=BtNode_ObjectScan(name=name, bb_key=bb_key,
                                vocabulary=vocab, use_orbbec=use_orbbec),
        num_failures=retries,
    )
```

Swap **only the whole-table multi-object scans**:
- `phaseTableCleanup` (line ~695): replace `_scanForGeneralistRetry(... bb_key=KEY_SCAN_RESULTS_TABLE, object=TABLE_SCAN_PROMPT, use_orbbec=True)` with
  `_scanForObjectScan(name="scan table for cleanup", bb_key=KEY_SCAN_RESULTS_TABLE)`.
- `phaseTableScan` (live mission): its scan is currently commented out (lines ~834–845),
  and the commented original wrapped the scan in `FailureIsSuccess("scan may fail", …)`.
  Restore that structure: wire
  `py_trees.decorators.FailureIsSuccess("scan may fail", _scanForObjectScan(name="scan table", bb_key=KEY_SCAN_RESULTS_TABLE))`
  in before `BtNode_WriteFoundItems`, so a total scan failure lets the phase continue
  (WriteFoundItems then announces "could not find any objects"). `phaseTableCleanup` keeps
  the original un-wrapped placement (parity with its existing `_scanForGeneralistRetry`).

**Untouched** (need 3D centroids for grasp, stay on the generalist): `_reDetectActive`,
`_breakfastItem`'s `re-detect {item}`, and the `scanForGeneralistRetry` import.

## Data flow / interface contracts

- **Vocabulary in:** `string[]` (BT splits `TABLE_SCAN_PROMPT` on `" . "`). Per-request,
  not node-side — keeps the BT in control and the service reusable for other vocabularies.
- **Labels out:** `string[] found_labels`, deduped, vocab order. Each label is a verbatim
  member of the request vocabulary (validation guarantees this).
- **Downstream compat:** BT node repacks to `objects[].cls` — no change to
  `WriteFoundItems` / `BuildInventory`.

## Error handling

| Condition | Service | BT node |
|---|---|---|
| No camera frame | `status=1, error_msg` | `FAILURE` → Retry |
| Empty vocabulary | `status=1` | `FAILURE` → Retry |
| One batch fails, others ok | `status=0`, partial labels, log `batches_fail` | writes labels |
| Every batch fails | `status=1, error_msg="all N VLM batches failed"` | `FAILURE` → Retry exhausts; in `phaseTableScan` the `FailureIsSuccess` wrap lets the phase continue (WriteFoundItems announces "could not find any objects"); in `phaseTableCleanup` the phase fails (parity with the original generalist wiring) |
| Gemini errors, Qwen ok | transparent (chain fallthrough) | writes labels |
| Both providers fail on a batch | that batch empty | see "one/every batch fails" |
| Missing `OPENROUTER_API_KEY` | node fails fast at init | tree.setup blocks (documented op requirement) |

## Testing

- **kimi_api unit** (`kimi_api/test/`): `_scan_vlm` validation — hallucination drop,
  case-insensitive match, empty-list-is-success (no fallback), parse-failure retry,
  chain fallthrough on `ScanVlmError`. Mock the OpenAI client (no network), same style as
  `test_vlm_match_client`.
- **Batch splitting**: pure function test — 32 items / `batch_size=8` → 4 batches; union
  ordering preserved; dedup across batches.
- **BT node**: labels → `objects[].cls` repack shape; mock path writes empty objects;
  `WriteFoundItems` / `BuildInventory` consume the repacked result (existing tests or a
  new focused one).
- **T1 startup smoke**: node starts, advertises `/object_scan`, SIGTERM-clean; key
  present/absent behavior — fold into `scripts/tests/` per the tiered suite.
- **Manual live (T2)**: one `/object_scan` call against the table scene; compare recall vs
  the single-call generalist to validate the batching win and pick a `batch_size`.

## Build / run

```bash
# interfaces first (new srv), then kimi_api
./src/tk26_vision/scripts/build.sh --packages-select tinker_vision_msgs_26 kimi_api
ros2 run kimi_api object_scan            # /object_scan

# BT side
colcon build --packages-select behavior_tree && source install/setup.bash
```

Requires `OPENROUTER_API_KEY` (Gemini) and, for the Qwen fallback, `DASHSCOPE_API_KEY` —
same `.env` the other kimi_api nodes use.

## Open questions / follow-ups

- `batch_size` default (8) is a starting bet; finalize from the T2 recall-vs-latency
  sweep (mirror `scripts/benchmark_match_batch_size.py`).
- If a future task needs both "scan all classes" **and** geometry, revisit whether
  `object_scan` should optionally delegate geometry to the generalist per found label —
  explicitly out of scope now.
