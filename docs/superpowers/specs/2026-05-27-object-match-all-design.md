# Design: `object_match_all` — concurrent VLM matching across all dataset items

**Status:** Draft for review
**Author:** Cindy (via Claude brainstorm)
**Date:** 2026-05-27
**Affected package:** `src/tk26_vision/src/tk_vision_specialized/`
**Affected interfaces:** new `tinker_vision_msgs_26/srv/ObjectMatchAll.srv`

## 1. Problem and goals

The existing `/object_match` service answers a single question: "where is item X in the
camera frame?" The caller passes one `category` name, the server runs a single
Qwen3-VL grounding call against the scene plus the reference image for that
category, picks the top bbox, and segments it with MobileSAM.

We want a sibling service that answers the dual question: "given the entire
items dataset, where is each item in the camera frame?" Conceptually this is
what `yolo_seg_node` does — one inference returns every detected object as a
list — but driven by the same VLM-plus-reference-image pipeline that the
single-category match uses, with concurrent VLM calls fanning out across the
dataset and a batching strategy to keep token cost bounded.

Concrete goals:

- Drop-in compatible at the response shape with `ObjectDetection.srv`, so any BT
  node that currently calls `/object_detection_yolo` can be retargeted to the
  new service by changing one parameter.
- Batched VLM calls (multiple categories per call) with the batch size tuned by
  an offline benchmark, kept as a ROS parameter so it can be re-tuned in the
  field without rebuilding.
- Provider-agnostic at the interface: Qwen3-VL (DashScope) and Gemini 2.5 Pro
  (OpenRouter), selectable per node via a parameter.
- Sparse-scene optimized (typical RoboCup table or shelf has 2–6 items).
  Conservative confidence handling, low-false-positive bias.
- Failure model is per-batch / per-detection: one failing VLM call or one
  no-depth detection should not poison the whole response.

Non-goals:

- Not a refactor of `YOLOSegmentationNode` or `object_match_server`. The new
  node is standalone; it does not subclass the existing detector.
- Not an open-vocabulary detector. The dataset is the items registered in
  `items_map.yaml`; out-of-vocabulary categories are out of scope.
- Not an action server. The single-frame response shape is sufficient for the
  initial consumers; streaming/cancellation can be added later if needed.

## 2. User-facing surface

### 2.1 New service: `tinker_vision_msgs_26/srv/ObjectMatchAll.srv`

```
# Camera identifier, same convention as ObjectDetection.srv:
#   "realsense", "orbbec", or substrings thereof.
string camera

# Empty list = scan every entry in items_map.yaml.
# Non-empty list = scan only these dataset keys; unknown keys are warned about
# and dropped from the scan. If every key in the filter is unknown, the
# response is status=1.
string[] category_filter

# TF frame to express centroids in. Empty string = raw camera frame.
string target_frame

# Sort modes (mirrors ObjectDetectionGeneralist.srv conventions):
#   sort_closest  - by sqrt(x^2+y^2+z^2) ascending, camera frame
#   sort_highest  - by camera-frame Z ascending
# Both false (default) = confidence descending.
bool sort_closest
bool sort_highest

# Payload toggles (mirrors ObjectDetectionGeneralist.srv).
bool return_rgb_image
bool return_depth_image
bool return_segments

---

# Response shape is identical to ObjectDetection.srv so callers expecting
# /object_detection_yolo's contract are drop-in compatible.
std_msgs/Header header
int32 status                     # 0=ok with >=1 object, 1=empty/failure
string error_msg
Object[] objects                 # cls / conf / centroid populated;
                                 # id, object_id, similarity, being_pointed
                                 # follow the conventions in ObjectMatch.srv
string detection_source          # always "vlm_match_all"

sensor_msgs/Image rgb_image      # populated iff return_rgb_image
sensor_msgs/Image depth_image    # populated iff return_depth_image; 32FC1 metres
sensor_msgs/Image[] segments     # one 8UC1 mask per object iff return_segments
```

### 2.2 Service name and node

- Default service name: `/object_match_all`.
- Node name: `object_match_all_server`.
- Lives in package `tk_vision_specialized` (sibling of `object_match_server.py`).
- Entry point in `setup.py`: `object_match_all = tk_vision_specialized.object_match_all_server:main`.

### 2.3 Empty-scene invariant

Following the workspace's existing convention for `ObjectDetection`-shaped
services: an empty scene returns `status=1, objects=[]`, never an exception.
Callers check `status` first.

## 3. Architecture

### 3.1 Module layout

```
src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/
├── object_match_all_server.py        # NEW: ROS Node + service callback
├── camera_data_source.py             # NEW: ROS camera-sync + intrinsics
│                                     #      + depth->3D + TF + centroid
│                                     #      + VisionLogger composition.
│                                     #      Logic copied from
│                                     #      YOLOSegmentationNode without
│                                     #      subclassing.
├── vlm_match_client.py               # NEW: MatchClient protocol +
│                                     #      QwenMatchClient +
│                                     #      GeminiMatchClient
├── vlm_judge_client.py               # NEW: JudgeClient protocol +
│                                     #      Qwen + Gemini backends
├── _vlm_common.py                    # NEW: shared decode utilities
│                                     #      (fence-strip, base64-encode,
│                                     #      retry-loop scaffold)
├── match_pipeline.py                 # NEW: pure-Python orchestrator,
│                                     #      no ROS deps, unit-testable
├── nms.py                            # NEW: IoU, within-cat NMS,
│                                     #      cross-cat clustering, judge
│                                     #      payload builder
└── items_map_loader.py               # existing, reused as-is

src/tk26_vision/src/tk_vision_specialized/scripts/
├── produce_match_ground_truth.py     # NEW: single-cat VLM -> GT JSON
├── benchmark_match_batch_size.py     # NEW: GT JSON + scenes -> CSV + summary
└── README.md                          # NEW: regenerate-GT + run-benchmark guide

src/tk26_vision/src/tk_vision_specialized/test/
├── test_nms.py                       # NEW
├── test_match_pipeline.py            # NEW
├── test_vlm_match_client.py          # NEW
└── test_vlm_judge_client.py          # NEW

src/tk26_vision/src/tinker_vision_msgs_26/srv/
└── ObjectMatchAll.srv                # NEW
```

### 3.2 Composition (no subclassing)

```python
class ObjectMatchAllServer(rclpy.node.Node):
    def __init__(self):
        super().__init__('object_match_all_server')
        self.params       = _declare_params(self)
        self.camera       = CameraDataSource(self, self.params)   # owns subs, sync,
                                                                  # intrinsics, TF,
                                                                  # VisionLogger
        self.items        = ItemsMapLoader(...)
        self.match_client = build_match_client(self.params.vlm_provider, ...)
        self.judge_client = build_judge_client(self.params.judge_provider, ...)
        self.sam          = SamPredictor(...)   # from object_detection_generalist
        self.pipeline     = MatchPipeline(self.match_client, self.judge_client,
                                          self.sam, self.items, self.params,
                                          logger=self.get_logger())
        self.srv          = self.create_service(
            ObjectMatchAll, '/object_match_all', self._callback,
            callback_group=MutuallyExclusiveCallbackGroup())
```

`CameraDataSource` is the only piece that holds ROS state and the only piece
whose logic is lifted from `YOLOSegmentationNode`. It exposes:

- `snapshot(camera) -> (rgb_bgr, points_xyz, valid_mask, header, intrinsic) | None`
- `centroid_for(points, mask, valid_mask, bbox, camera) -> Point | None`
- `transform_point(point, target, source, stamp) -> Point | None`
- `frame_supports_tf_transform(camera) -> bool`
- `write(...)` — wraps a `VisionLogger` instance and produces the same
  per-request artifacts as the other vision nodes.

`MatchPipeline` is pure-Python and accepts `(rgb_bgr, points_xyz, valid_mask,
intrinsic, header, request)` and returns `list[FinalRow]` plus a counters
dict. No ROS imports; unit-testable with fake clients.

### 3.3 SAM wrapper reuse

`SamPredictor` is imported from `object_detection_generalist.sam_mask` — the
same cross-package import `object_match_server` already does. The wrapper
already accepts a list of bboxes and returns masks aligned 1:1, so batched
segmentation is a single `self.sam.segment(rgb_bgr, bboxes)` call.

Warmup happens at node init on a synthetic 64×64 image, same pattern as
`object_match_server._init_sam`.

## 4. Pipeline / data flow

```
[1]  CameraDataSource.snapshot(camera)
        -> (rgb_bgr, points_xyz, valid_mask, header, intrinsic)
        Same sync semantics as today's object_match_server.

[2]  Resolve category filter
        keys = req.category_filter or items_map.keys()
        Unknown keys warned and dropped; all-unknown -> status=1.
        refs = [(k, items.get_data_url(k)) for k in keys]

[3]  Partition into batches
        B = self.params.batch_size                # ROS param, default 3
        batches = chunks(refs, B)

[4]  Stage-1: concurrent VLM match
        pool = ThreadPoolExecutor(max_workers=params.max_workers)
        futures = [pool.submit(match_client.match_batch,
                               rgb_bgr, batch,
                               timeout_s=params.vlm_per_call_timeout_s,
                               max_retries=params.vlm_max_retries)
                   for batch in batches]
        for f in as_completed(futures, timeout=stage1_timeout_s):
            try: rows.extend(f.result())
            except Exception: log and continue
        If all batches failed -> status=1; if stage1 budget elapsed with 0
        batches landed -> status=1.

[5]  Within-category NMS                          (nms.suppress_within_category)
        For each label, greedy NMS at iou >= nms_within_category_iou (0.5).
        Allows multiple instances of one category when their bboxes don't
        overlap.

[6]  Cross-category clustering                    (nms.cluster_for_judge)
        Union-find on the pairwise IoU graph at iou >= cluster_iou (0.5).
        Singletons (or clusters with only one distinct label) pass through
        unchanged. Clusters with >=2 distinct labels enter the judge stage.

[7]  Stage-2: concurrent VLM judge
        For each conflict cluster:
            union_bbox = union(rows.bbox) + judge_crop_margin_px
            crop       = rgb_bgr[union_bbox]
            competing  = [(label, items.get_data_url(label))
                          for label in distinct_labels(cluster)]
        Submit all judge calls concurrently with max_workers=params.max_workers
        and stage2_timeout_s wall-clock budget.
        Resolution per cluster:
            JudgeChoice(label=L) with L in cluster labels -> pick highest-conf
                row of label L; replace its conf with the judge's conf.
            JudgeChoice(label="") -> drop the whole cluster (judge abstained).
            None / timeout / exception -> fall back to highest-conf row.

[8]  Assemble surviving rows
        singletons + judge winners -> list[MatchRow]

[9]  MobileSAM batched
        masks, sam_s = self.sam.segment(rgb_bgr, [r.bbox for r in rows])
        len(masks) == len(rows) by SamPredictor contract.

[10] Centroids
        For each (row, mask): camera.centroid_for(points, mask, valid_mask,
                                                  row.bbox, camera).
        If None, retry with rect_mask(bbox); if still None, drop that
        detection with a warning.
        If every detection dropped -> status=1.

[11] Optional TF transform to target_frame
        All-or-nothing: any per-detection TF failure when target_frame is set
        -> status=1, header.frame_id stays at camera frame. Avoids mixing
        frames inside one Object[] (the schema has no per-Object frame_id).

[12] Pack response
        header.frame_id = target_frame or camera_frame_id
        Sort per request flags (sort_closest, sort_highest) or by conf desc.
        Build Object[] with cls/conf/centroid populated.
        Optionally fill rgb_image / depth_image / segments per request bools.
        Log one INFO summary line with counters and timings.
        If response.objects empty -> status=1 with descriptive error_msg.
```

### 4.1 Invariants

- **One scene frame per request.** All batches and the judge see the same
  `(rgb_bgr, points_xyz, header, intrinsic)` snapshot captured at step [1].
- **Single `MutuallyExclusiveCallbackGroup` on the service** so concurrent
  requests serialize at the node boundary. Internal parallelism is unaffected.
- **Two distinct executors** (stage-1 and stage-2). Same `max_workers` cap
  applied to each. Prevents judge calls from queuing behind stale stage-1
  futures.
- **Empty scene -> `status=1, objects=[]`, never an exception.**

### 4.2 Worst-case wall-clock per request

```
T <= T_camera_sync + stage1_timeout_s + stage2_timeout_s + T_SAM + T_pack
   ~= 0.1 + 15 + 10 + 0.2 + 0.05 = ~25 s ceiling
   ~= 6-10 s typical on a sparse scene with no conflicts
```

Action wrappers and BT timeouts that depend on this service should give it
30 s headroom.

## 5. VLM client adapters

### 5.1 `vlm_match_client.py`

```python
@dataclass(frozen=True)
class MatchRow:
    label: str                                # constrained to refs' labels
    bbox: tuple[int, int, int, int]           # xyxy in scene pixels
    conf: float                                # [0.0, 1.0]

class MatchClient(Protocol):
    def match_batch(
        self,
        scene_bgr: np.ndarray,
        refs: list[tuple[str, str]],          # [(label, ref_data_url), ...]
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> list[MatchRow]: ...

def build_match_client(provider: str, **opts) -> MatchClient:
    if provider == 'qwen':   return QwenMatchClient(**opts)
    if provider == 'gemini': return GeminiMatchClient(**opts)
    raise ValueError(...)
```

Adapter responsibilities (both providers):

- Encode scene + N refs into the provider's multi-image request shape.
- Provide a system prompt that enforces the response schema and constrains
  `label` to the set of input labels.
- Decode the response body, normalize bbox coordinates to scene pixel xyxy,
  clamp to image bounds, reject degenerate boxes.
- Drop any row whose `label` is not in the input set (defensive against
  hallucinated labels).
- Retry on parse failure up to `max_retries` (default 1). On a Qwen
  schema-rejection error, fall back to `json_object` mode without consuming
  the retry budget (matches today's `request_match_bboxes` behavior).

#### Qwen3-VL coordinate convention

Provider returns `box_2d` as `[x1, y1, x2, y2]` normalized to `0..1000` over
the **scene** dimensions. Adapter scales to scene pixel xyxy.

#### Gemini coordinate convention

Provider has no stable normalized convention; system prompt asks explicitly
for absolute pixel xyxy in the scene's dimensions. Adapter clamps to bounds.

#### Credentials

- Qwen backend: `DASHCOPE_API_KEY` then `DASHSCOPE_API_KEY` (matches today's
  `qwen_match_vlm._resolve_api_key`).
- Gemini backend: `OPENROUTER_API_KEY` (matches `kimi_api/_env.py`).
- Missing credential at node init for the selected provider -> `RuntimeError`.
  Fails loud rather than silently per-request, matching `kimi_api` behavior.

### 5.2 `vlm_judge_client.py`

```python
@dataclass(frozen=True)
class JudgeChoice:
    label: str                              # one of competing labels, or "" if abstain
    conf: float

class JudgeClient(Protocol):
    def choose(
        self,
        crop_bgr: np.ndarray,
        competing: list[tuple[str, str]],   # (label, ref_data_url)
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> JudgeChoice | None: ...
```

Provider split mirrors `MatchClient`. Same retry / fallback semantics. Prompt
instructs the model to return `{label, confidence}` with `label` constrained
to competing labels or `null` to abstain.

### 5.3 Why match and judge are separate clients

- Their prompts, schemas, and input shapes are unrelated; one class with two
  methods would obscure both.
- Tests want to mock them independently (a "match works but judge times out"
  scenario should be a one-line setup).
- Independent providers per role are useful for benchmarking. The node
  exposes `vlm_provider` and `judge_provider` as separate ROS params, with
  `judge_provider` defaulting to the value of `vlm_provider`.

### 5.4 Shared utilities in `_vlm_common.py`

- `strip_fences(text)` — drop ` ```json ... ``` ` fences some Qwen revisions emit.
- `encode_data_url(bgr_ndarray)` — base64 jpeg data URL.
- `retry_loop(...)` — common backoff and parse-retry scaffold.

## 6. Cross-batch dedup + judge details

All in `nms.py` (pure functions, no ROS).

### 6.1 IoU and within-category NMS

Standard `iou(box_a, box_b)`. `suppress_within_category(rows, iou_thresh)`
groups by label, sorts by confidence descending, greedily keeps boxes that
don't IoU-overlap any already-kept box.

### 6.2 Cross-category clustering

`cluster_for_judge(rows, iou_thresh)` builds an IoU graph (edge iff
`iou(a, b) >= iou_thresh`) and returns connected components via union-find.
Pairwise loop is O(n^2), acceptable under the sparse-scene assumption.

### 6.3 Judge payload

`build_judge_payload(cluster, items, scene_bgr, margin_px=20)`:

1. Compute union bbox of cluster member boxes.
2. Expand by `judge_crop_margin_px` (default 20), clamped to image bounds.
3. Crop scene.
4. Distinct labels in cluster -> `competing = [(label, ref_data_url), ...]`.

The judge picks **the label**, not a specific row. After the judge returns
`L`, the pipeline keeps the highest-conf row in the cluster whose label is
`L`, replacing its `conf` with the judge's confidence.

### 6.4 Judge fallbacks

| Judge outcome | Pipeline action |
|---|---|
| `JudgeChoice(label=L)` with L in cluster labels | Keep highest-conf row of label L; use judge's conf. |
| `JudgeChoice(label="")` (abstain) | Drop the whole cluster. Log at INFO. |
| `None` (timeout / exception / parse fail) | Keep highest-conf row in the cluster as-is. |
| L not in cluster labels (defensive) | Treat as `None`. |

Rationale: judge abstain is a real signal we honor; judge transport failure
is not a meaningful signal about the detections themselves, so we degrade to
best-effort.

## 7. Errors, timeouts, observability

### 7.1 Failure matrix

| Stage | Failure | Response | Pipeline action |
|---|---|---|---|
| [1] Camera sync | No recent frame within `img_sync_thres` | `status=1`, `error_msg='No {camera} camera data within sync threshold'` | Return. |
| [1] Camera sync | No intrinsics yet | `status=1`, `error_msg='No {camera} camera intrinsics available'` | Return. |
| [1] Camera sync | RGBD processing raises | `status=1`, `error_msg='camera data processing error: {exc}'` | Return. ERROR log. |
| [2] Categories | All filter keys unknown | `status=1`, `error_msg='Unknown items: ...'` | Return. |
| [4] Match batch | Per-batch timeout / exception | (warn) | Drop that batch; continue. |
| [4] Match batch | All batches failed | `status=1`, `error_msg='all VLM match batches failed: {summary}'` | Return. |
| [4] Stage-1 budget | Budget elapsed with stragglers | (warn) | Cancel stragglers, proceed. |
| [4] Stage-1 budget | Budget elapsed, zero batches landed | `status=1`, `error_msg='stage1 budget exhausted ...'` | Return. |
| [7] Judge | Per-call timeout / exception / parse fail | (warn) | Fall back to highest-conf row. |
| [7] Judge | Abstain | (info) | Drop cluster. |
| [7] Stage-2 budget | Budget elapsed | (warn) | Cancel in-flight, fall back. |
| [9] SAM | `segment()` raises | `status=1`, `error_msg='SAM error: {exc}'` | Return. ERROR log. |
| [10] Centroid | SAM mask + rect fallback both invalid | (warn) | Drop that detection. |
| [10] Centroid | All detections dropped | `status=1`, `error_msg='no valid-depth pixels for any matched object'` | Return. |
| [11] TF | Any per-detection TF fails when `target_frame` set | `status=1`, `error_msg='TF {src}->{tgt} unavailable for {k}/{n} detections'` | Return. |
| [12] Pack | Empty after all stages | `status=1`, `error_msg='no items matched'` | Return. |
| Init | Items map missing / empty | `RuntimeError` at construction | Node refuses to start. |
| Init | API key missing for selected provider | `RuntimeError` at construction | Node refuses to start. |
| Init | SAM weights missing | `RuntimeError` at construction | Node refuses to start. |

### 7.2 ROS parameters

| Param | Default | Purpose |
|---|---|---|
| `service_name` | `'object_match_all'` | Service advertised. |
| `vlm_provider` | `'qwen'` | `qwen` or `gemini`. |
| `judge_provider` | `''` | Empty string = inherit `vlm_provider`. Otherwise `qwen` or `gemini`. |
| `vlm_model` | `''` | Override; empty = use the selected adapter's built-in default (Qwen adapter: `qwen3-vl-plus`; Gemini adapter: `google/gemini-2.5-pro`). |
| `judge_model` | `''` | Override for the judge role; empty = inherit `vlm_model` resolution for the judge's provider. |
| `vlm_base_url` | `''` | Override; empty = use the selected adapter's built-in default (Qwen: DashScope; Gemini: OpenRouter). |
| `vlm_per_call_timeout_s` | 12.0 | Per-batch / per-judge HTTP timeout. |
| `vlm_max_retries` | 1 | Per-call retry budget. |
| `stage1_timeout_s` | 15.0 | Stage-1 wall-clock budget. |
| `stage2_timeout_s` | 10.0 | Stage-2 wall-clock budget. |
| `max_workers` | 8 | ThreadPoolExecutor cap per stage. |
| `batch_size` | 3 | Categories per match call (set from benchmark). |
| `nms_within_category_iou` | 0.5 | Within-category NMS threshold. |
| `cluster_iou` | 0.5 | Cross-category cluster threshold. |
| `judge_crop_margin_px` | 20 | Context margin around cluster union bbox. |
| `min_valid_centroid_pixels` | 8 | Minimum valid-depth pixels per centroid. |
| `items_map_path` | `''` | Override items_map directory (absolute path); empty = auto-resolve via package share + dev-tree fallback, same as `object_match_server`. |
| `sam_weights` | `'mobile_sam.pt'` | SAM checkpoint. |
| `sam_device` | `''` | SAM device override; empty -> auto. |
| `vision_logging_enabled` | `true` | Same convention as the rest of the vision tree. |
| `vision_log_folder` | `'vision_log'` | Same. |
| `log_raw_vlm` | `false` | DEBUG dump of raw provider responses (large). |

Camera-topic params (`realsense_color_topic`, etc.) are inherited via
`CameraDataSource` from the same defaults the existing detectors use.

### 7.3 Observability

- **One INFO summary line per request**:
  ```
  match_all: batches={B} ok={ok} fail={fail}
             rows_in={n_raw} after_nms={n_nms} after_judge={n_final}
             clusters_conflict={k} judge_ok={jok} judge_abstain={jab} judge_fail={jfa}
             total_s={t} stage1_s={s1} stage2_s={s2} sam_s={ts}
  ```
- **WARN per dropped batch / dropped detection / judge failure** with reason.
  Bounded by `N_batches + N_detections`.
- **ERROR on node-init failures and unrecoverable cases** (SAM crash, RGBD
  processing exception).
- **DEBUG dumps** of per-batch parsed detection lists, with raw provider
  bodies suppressed unless `log_raw_vlm:=true`.
- **Vision-logger artifacts** per request (orig + overlay + req-JSON) under
  the workspace's existing session-resolution rules (`$TINKER_VISION_SESSION_TS`
  -> newest existing -> fresh strftime).

In-memory counters dict updated per call and surfaced in the summary line:

```
match_all_requests_total, match_all_requests_failed_total,
batches_ok_total, batches_timeout_total, batches_error_total,
judge_ok_total, judge_abstain_total, judge_fail_total,
detections_total, detections_dropped_no_depth_total,
tf_failed_total
```

### 7.4 Shutdown behavior

`with ThreadPoolExecutor()` context managers tear down on `destroy_node`.
In-flight VLM calls finish or hit the per-call timeout — HTTP requests are
not cancelled mid-flight. SAM has no cancellable async surface; if a request
is mid-SAM when SIGTERM arrives, the request completes and then the node
shuts down. Matches `object_match_server`.

## 8. Testing

### 8.1 Unit tests (`src/tk_vision_specialized/test/`)

| File | Coverage | Network required |
|---|---|---|
| `test_nms.py` | IoU, within-cat NMS, cross-cat clustering. Property-style: random box sets; assert idempotence and invariants (no two kept boxes within IoU thresh; cluster union covers input). | No |
| `test_match_pipeline.py` | Full pipeline driven by `FakeMatchClient`, `FakeJudgeClient`, `FakeSam`, `FakeCameraDataSource`. Covers: empty scene; one batch fails; all batches fail; abstain judge; judge timeout fallback; detection dropped for no depth; TF all-fail; sort modes. | No |
| `test_vlm_match_client.py` | Adapter decode against canned provider JSON, both Qwen (0..1000 normalize) and Gemini (pixel xyxy) paths. Label-whitelist enforcement. Fence-strip and json_object fallback. | No |
| `test_vlm_judge_client.py` | Adapter decode for both providers, abstain decoding, error path. | No |

`Fake*` clients are dataclasses configured per-test (return rows, raise on
call, sleep-then-return for timeout). Pipeline is fully exercisable without
network or GPU.

### 8.2 Integration tiers (extend existing T0-T4)

| Tier | New cases |
|---|---|
| **T0** static | Shebang + entry-point import. `ObjectMatchAll.srv` compile. Items map sanity. |
| **T1** startup | Node starts, advertises `/object_match_all`, terminates on SIGTERM. With Qwen creds and with Gemini creds (env-driven). Negative: missing API key for selected provider -> `RuntimeError` at init. |
| **T2** live | Empty scene -> `status=1, objects=[]`. Camera-sync timeout reproducibility. |
| **T3** interaction | Staged sparse table (2-6 items): expect >=1 hit per visible item, no hallucinations. Drop-in cross-check: retarget an existing consumer of `/object_detection_yolo` at `/object_match_all` via param. |
| **T4** hardware | Reuse `shelf_scene` and `person` staged setups; operator scores precision/recall. |

Suite invariant: **empty scene returns `status=1, objects=[]`, not an
exception.** Matches the existing `ObjectDetection` invariant.

### 8.3 Offline batch-size benchmark

Two sibling scripts in `scripts/`. Not part of T0-T4 because they cost real
VLM tokens and only need to run when the dataset or the provider changes.

#### 8.3.1 `produce_match_ground_truth.py`

Generates ground truth by running the **existing single-category** VLM call
(`qwen_match_vlm.request_match_bboxes`) over each `(scene, category)` pair
and treating high-confidence single-category predictions as ground truth.

**Inputs:**

```
--scenes-dir <dir>        directory of scene_*.jpg
--items-dir  <dir>        items_map.yaml + reference images
--provider   qwen         qwen (default) | gemini
--top-k      3            forwarded to request_match_bboxes
--min-conf   0.6          drop GT entries below this threshold
--out        ground_truth_<ts>.json
```

**Algorithm:** for each scene, for each category, call
`request_match_bboxes(scene, ref, item_name=category, top_k=...)`. Keep
returned candidates with `conf >= min_conf`. Apply within-category NMS at
iou=0.5 to collapse `top_k` overlaps.

**Output JSON:**

```json
{
  "_meta": {
    "provider": "qwen", "vlm_model": "qwen3-vl-plus",
    "top_k": 3, "min_conf": 0.6,
    "items": ["biscuit", "bread", "..."],
    "generated_at": "2026-05-27T14:00:00Z"
  },
  "scene_001.jpg": [
    {"category": "milk", "bbox": [120, 80, 240, 320], "conf": 0.91},
    {"category": "bread", "bbox": [310, 90, 470, 280], "conf": 0.84}
  ]
}
```

The `_meta.items` set is checked by the scorer against the items_map seen at
benchmark time; mismatch -> hard error. Catches "I added a category and
forgot to regenerate GT."

**Cost:** `N_scenes * N_categories` single-cat calls. ~$1-3 per regeneration
on the default 10-scene x 10-item dataset.

**Caveat documented at the top of the output and in the README:** this is
"VLM ground truth," not human ground truth. It measures **agreement with the
single-category service we trust in production**, not absolute correctness.
The benchmark question this enables is "does batching degrade what we
already accept?", not "is the VLM right." If batching reaches F1=1.0
against this GT, batching is free.

**Manual override:** the JSON is a plain dict; the operator can hand-edit
before benchmarking to correct known-bad single-cat predictions.

#### 8.3.2 `benchmark_match_batch_size.py`

Sweeps batch sizes against a labeled scene set and reports
precision/recall/latency/token-cost per (provider, batch_size).

**Inputs:**

```
--scenes-dir <dir>
--items-dir  <dir>
--ground-truth ground_truth_<ts>.json
--batch-sizes 1 2 3 5 8
--provider   qwen|gemini|both
--repeats    3
--out-prefix benchmark_<ts>
```

**Scoring:** TP if predicted bbox has IoU >= 0.3 with a GT bbox of the same
label. Precision, recall, F1 averaged across scenes.

**Outputs:**

- `benchmark_<ts>.csv` — one row per (scene, provider, batch_size, repeat).
- `benchmark_<ts>.md` — summary table of medians and p95s per
  (provider, batch_size), plus a one-line recommended default. Selection
  rule: maximize F1, break ties by lower token cost, then lower p95 latency.
- Recommendation is **advisory**. The operator updates `batch_size` defaults
  in launch params after reviewing the summary. Premature auto-pin creates a
  config-drift problem when the dataset changes.

**Re-run triggers** (documented in `DEV_NOTES.md`):

- items_map.yaml additions or removals,
- VLM provider switch,
- accuracy regression observed in T3/T4.

**Provider matrix:** GT is generated once with `--provider qwen`. The
benchmark sweeps both batched providers against the same Qwen GT so the
agreement numbers are directly comparable. Cross-provider GT sanity sweeps
(regenerate GT with `--provider gemini`, eyeball diff) are out of scope for
the initial implementation; noted as a follow-up.

## 9. Cost and latency analysis

Token-cost per request (rough order of magnitude, sparse 10-item dataset):

- **Stage-1:** `ceil(N/B) * (1 scene image + B ref images + system_prompt)`.
  At B=3 with N=10 that is 4 calls. Scene image is the dominant token cost;
  the refs are small (~100 KB each).
- **Stage-2:** `~N_conflicts * (1 crop image + k ref images)`. Sparse scenes
  produce 0-1 conflict clusters in practice, so this is usually one call or
  zero. Each call is cheaper than stage-1 calls because the crop is smaller.

Latency:

- Stage-1: dominated by the slowest of `ceil(N/B)` parallel calls. Bound by
  `vlm_per_call_timeout_s` per call and `stage1_timeout_s` for the whole
  stage.
- Stage-2: usually negligible (0-1 calls) or capped by `stage2_timeout_s`.
- SAM: ~0.2 s for typical N on the warm-cached MobileSAM weights.
- Total typical: 6-10 s on a sparse scene with no conflicts. Worst case
  ~25 s.

The cost ceiling is the headline operator-facing number; the offline
benchmark provides the data to lower B without exceeding it.

## 10. Migration / drop-in semantics

The response schema of `ObjectMatchAll.srv` is intentionally identical to
`ObjectDetection.srv`. To swap a consumer from `/object_detection_yolo` to
`/object_match_all`:

1. Update the consumer's service-name ROS param (most callers already expose
   one).
2. If the consumer relies on YOLO class names that aren't in `items_map.yaml`,
   either extend the items_map or keep that caller on the YOLO service.
3. Latency expectations change: YOLO is sub-second; the VLM matcher is
   seconds-tens-of-seconds. Action wrappers and BT timeouts need adjustment.

`/object_match_all` does not replace `/object_detection_yolo`. Both coexist;
callers pick the right one per use case. The VLM matcher is the recommended
target for **dataset-restricted, latency-tolerant** scans where the
reference-image advantage matters (visually similar SKUs the YOLO model
can't tell apart).

## 11. Out of scope / follow-ups

- Refactor of `YOLOSegmentationNode` to extract camera infra into a shared
  module (the new node implements the shared module independently for now).
- Action-server variant with streaming partial results.
- Cross-provider GT sanity sweeps in the offline benchmark.
- Auto-tuning `batch_size` at runtime based on response-quality signals.
- Open-vocabulary (out-of-items_map) detection — out of scope; use
  `object_detection_generalist` for that.
- Lifting `VisionLogger` out of `object_detection_new` into a shared package
  (current spec composes via `CameraDataSource`; the lift is a separate
  cleanup).

## 12. Open questions

None at this stage. The original brainstorm covered:

- Consumer / output shape: BT replacement for `yolo_seg_node` — settled.
- Dataset source: `items_map.yaml`, plan to scale to N — settled.
- VLM provider: configurable, both Qwen and Gemini — settled.
- Batch-size tuning: offline benchmark + ROS-param override — settled.
- Reference handling: 1 scene + N refs in one call, text-labeled — settled.
- Scene composition: sparse-scene optimized, low-FP — settled.
- Cross-batch dedup: within-cat NMS + per-conflict VLM judge (concurrent) — settled.
- SAM: batched in one call — settled.
- Concurrency: bounded 8 workers, 12 s per-call timeout — settled.
- Subclassing vs standalone: standalone (no subclassing) — settled.
- GT for benchmark: sibling script that runs the existing single-category
  VLM to produce GT JSON — settled.
