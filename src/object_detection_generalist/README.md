# object_detection_generalist

Open-vocabulary object detection for tk26 vision. Runs a clean pretrained YOLO for classes YOLO already knows; for anything else (or when the caller explicitly opts in), falls back to an open-vocab detector. The fallback model is selected at node init by the `enable_vlm` parameter:

- **`enable_vlm=False` (default)** — local **YOLO-World** for bounding boxes + **FastSAM** for masks. Hundreds of ms per call, no network.
- **`enable_vlm=True`** — **Gemini 2.5 Flash** for bounding boxes + **FastSAM** for masks. Multi-second per call, requires `OPENROUTER_API_KEY`.

3D centroid + camera-sync logic is inherited unchanged from `object_detection_new.object_seg_yolo.YOLOSegmentationNode`.

## Interface

Service: `/object_detection_generalist`, type [`tinker_vision_msgs_26/srv/ObjectDetectionGeneralist`](../tinker_vision_msgs_26/srv/ObjectDetectionGeneralist.srv).

This is a **new srv** with typed boolean flags — not the legacy `tinker_vision_msgs_26/srv/ObjectDetection` (same package, different type) that `/object_detection` and `/object_detection_yolo` still accept. See [`tinker_vision_msgs_26/README.md`](../tinker_vision_msgs_26/README.md#objectdetectionsrv-field-mapping-from-tk23--tk26) for the field mapping.

Request fields that matter for path selection:

| Field | Meaning |
|---|---|
| `prompt` | Natural-language or class name. Must match one of `model.names.values()` for the YOLO branch; anything else forces the open-vocab fallback branch. |
| `force_vlm_sam` | Skip YOLO entirely, always use **VLM + FastSAM only** (operator override; ignores `enable_vlm`). Short-circuits **before** any YOLO-World code path — `force_vlm_sam=True` guarantees YOLO-World does not run. Field name is historical. |
| `use_vlm_sam_fallback` | Per-request opt-in for the open-vocab fallback. Runs **YOLO-World and VLM concurrently**: if YOLO-World returns objects first, the VLM result is abandoned (the network call continues in the background as a daemon thread but its output is discarded); otherwise we wait for VLM. |
| `sort_closest` / `sort_highest` | Typed boolean replacements for the old substring-parsed `flags` string. |
| `return_rgb_image` / `return_depth_image` / `return_segments` | Payload toggles (save bandwidth when false). |

Response adds `string detection_source ∈ {"yolo", "yolo_world", "vlm_sam", "none"}` so the caller can tell which branch answered.

## Branching logic

```
if request.force_vlm_sam:                        # operator override
    run VLM + SAM only                           # (regardless of enable_vlm)

elif prompt ∈ model.names:                       # YOLO knows this class
    run YOLO
    if YOLO returns []:
        if request.use_vlm_sam_fallback:         # race
            run YOLO-World ∥ VLM, prefer YOLO-World if non-empty
        elif node.allow_auto_fallback:           # single fallback per enable_vlm
            run YOLO-World      (enable_vlm=False, default)
            or VLM + SAM        (enable_vlm=True)
        else:
            return YOLO's empty result

elif request.use_vlm_sam_fallback:               # OOV prompt, race
    run YOLO-World ∥ VLM, prefer YOLO-World if non-empty

elif node.allow_auto_fallback:                   # OOV prompt, single fallback
    run YOLO-World or VLM per enable_vlm

else:
    status=1, error_msg="class not in YOLO names and fallback disabled"
```

`allow_auto_fallback` is a node-level ROS parameter (default `True`). Flip it off at launch (`-p allow_auto_fallback:=false`) if the caller must opt in per-request.

`enable_vlm` is a node-level ROS parameter (default `False`). It only affects the **auto-fallback** path; the per-request flags (`force_vlm_sam`, `use_vlm_sam_fallback`) override it. When `enable_vlm=False` and the auto-fallback is triggered, only YOLO-World runs.

### Race details (`use_vlm_sam_fallback=True`)

Both pipelines spawn as daemon `threading.Thread`s from the service callback. The service callback waits on YOLO-World first (bounded by `vlm_timeout_s`, floor 5 s). If YOLO-World produces ≥1 object, the callback returns immediately with that result and **cancels the VLM leg**:

- An `abandon_event` is set. The VLM worker checks it (a) at every retry boundary inside `request_bboxes`, (b) immediately after the HTTP returns in `_vlm_pipeline`, and (c) before acquiring the FastSAM lock. So **FastSAM, centroid math, and TF lookups are guaranteed to be skipped** — the abandoned thread does no GPU work and does not contend with the next race's `_sam_lock`.
- The OpenAI client owned by that VLM call is also `close()`d cross-thread as a best-effort. With sync httpx this does **not** reliably interrupt a blocking socket read, so worst-case the VLM thread stays parked in `recv()` until its `vlm_timeout_s` fires (default 20 s), then exits cleanly. Best-case (close happens before the request enters `recv()`, or the platform honors close cross-thread) the thread terminates within milliseconds.

If YOLO-World comes back empty (or errored), the callback waits up to `vlm_timeout_s × vlm_max_retries + 5 s` for VLM and returns whichever leg has non-empty results. If even that wait expires, we cancel the VLM leg the same way and return an error.

**Concurrent requests.** Because the service callback returns as soon as a winner is found, the next request can start while a previous abandoned VLM thread is still parked in its socket. That thread does no useful work after abandon (it just exits when its HTTP attempt unblocks), so the practical impact is at most `N` simultaneously parked HTTP sockets where `N` is the number of races started within one `vlm_timeout_s` window.

The race threads are NOT executor-managed. They access:
- **YOLO-World** and **VLM** (independent models, no contention).
- **FastSAM** (shared between race legs; serialized by `self._sam_lock` since Ultralytics models are not thread-safe).
- **`tf2_ros.Buffer`** for centroid/sort transforms (read-side is thread-safe).
- **`rclpy` logger** (thread-safe).

### Executor configuration

`main()` pins a `MultiThreadedExecutor(num_threads=max(8, cpu_count()))`. The service callback blocks during the race, so the camera sync callback groups (`cb_realsense`, `cb_orbbec`) and the TF-listener subscription must run on separate executor threads to keep image streams + TF lookups fresh while the race is in flight. Default `cpu_count()` covers it on the dev workstation; the explicit floor of 8 prevents starvation on lower-thread embedded targets.

If YOLO-World fails to load at startup (missing weights / ultralytics import error), the node logs the error and continues; the race is still safe (YOLO-World leg returns an `error` immediately). With `enable_vlm=False` and YOLO-World unavailable, the auto-fallback path returns `status=1` with `error_msg='YOLO-World unavailable at node init'`.

## Running

```bash
# API key for the VLM branch (optional at startup — lazy-checked on first VLM call)
export OPENROUTER_API_KEY=...   # or put it in /home/tinker/tk25_ws/.env (auto-loaded)

# cameras running per CAMERA_BRINGUP.md, then:
ros2 run object_detection_generalist generalist_node
```

Common parameter overrides:

| Param | Default | Notes |
|---|---|---|
| `service_name` | `object_detection_generalist` | Rename the advertised service. |
| `model_path` | `yolo11m-seg.pt` | YOLO weights (auto-downloaded by Ultralytics on first run). |
| `fastsam_weights` | `FastSAM-s.pt` | ~22 MB, auto-downloaded. |
| `enable_vlm` | `False` | When `True`, the fallback path uses Gemini VLM+SAM instead of YOLO-World+SAM. |
| `world_weights` | `yolov8s-worldv2.pt` | YOLO-World v2 weights (~25 MB, auto-downloaded). Bigger variants: `m`, `l`, `x`. |
| `world_conf_threshold` | `0.05` | YOLO-World tends to need a low threshold for novel classes; raise if you see false positives. |
| `world_iou_threshold` | `0.5` | NMS IoU for YOLO-World. |
| `vlm_model` | `google/gemini-2.5-flash` | OpenRouter model tag. Override per deployment via `-p vlm_model:=anthropic/claude-sonnet-4-6` etc. |
| `vlm_timeout_s` | `20.0` | Per-VLM-call timeout. With `vlm_stream=True` (default) this is the per-chunk inactivity bound rather than a total-response deadline. |
| `vlm_max_retries` | `3` | JSON parse / API retries before returning empty. |
| `vlm_stream` | `True` | Stream the OpenRouter VLM response as SSE chunks. Keeps the HTTP connection active during long Gemini generations so intermediate proxies don't reap the silent socket, and gives sub-100 ms cancellation latency when the YOLO-World race partner wins. Flip to `False` to fall back to a single blocking response. |
| `realsense_max_distance_m` | `1.0` | Range gate: drop detections whose centroid is farther than this from the realsense (arm) camera. Applied **only** when `request.camera == 'realsense'` — orbbec is unaffected. Set to `0.0` (or any non-positive value) to disable. Distance is Euclidean from the camera origin in the camera body frame (`sqrt(x²+y²+z²)` on `Object.centroid`, which on realsense is never TF-transformed). |
| `allow_auto_fallback` | `True` | See branching above. |
| `orbbec_depth_topic` | `/camera/depth_registered/points` | Must match what the camera launch publishes. For the canonical Femto Bolt launch this is `/camera/depth/points` — override accordingly. |

## Example calls

```bash
# YOLO branch (chair is a COCO class)
ros2 service call /object_detection_generalist tinker_vision_msgs_26/srv/ObjectDetectionGeneralist \
  "{camera: realsense, prompt: chair}"

# Open-vocabulary via the default fallback (YOLO-World)
ros2 service call /object_detection_generalist tinker_vision_msgs_26/srv/ObjectDetectionGeneralist \
  "{camera: realsense, prompt: 'monitor screen', use_vlm_sam_fallback: true}"

# Force fallback even on a COCO class (uses YOLO-World by default)
ros2 service call /object_detection_generalist tinker_vision_msgs_26/srv/ObjectDetectionGeneralist \
  "{camera: realsense, prompt: bottle, force_vlm_sam: true}"

# Run with Gemini fallback instead of YOLO-World
ros2 run object_detection_generalist generalist_node --ros-args -p enable_vlm:=true
```

## Architecture

- `generalist_node.py` — `GeneralistDetectionNode` subclasses `YOLOSegmentationNode`; overrides `_init_service` / `_detection_service_callback`. Camera sync, TF, depth projection, 3D centroid, and sort-mode logic are inherited, not duplicated.
- `world_bbox.py` — `WorldDetector` wraps `ultralytics.YOLOWorld`, loads once at node init on the parent's `self.device`, exposes `detect(rgb, prompt) -> (bboxes, confs, elapsed)`. Calls `set_classes([prompt])` per request (cached against last prompt) so the CLIP text head re-projects to the new label.
- `vlm_bbox.py` — `request_bboxes(rgb, prompt, model, …)` sends a base64 data-URL image + strict-JSON system prompt to OpenRouter, retries on parse failure, decodes Gemini's `[y0,x0,y1,x1]` 0–1000-normalized output into xyxy pixel coords. Key loading is deferred until first call via `kimi_api._env.require_api_key`, so the node starts cleanly without a key.
- `sam_mask.py` — `FastSAMPredictor` wraps `ultralytics.FastSAM`, loads once at node init on the parent's `self.device` (GPU if available), exposes `segment(rgb, bboxes) -> list[bool HxW mask]` aligned 1:1 with the input bboxes. Used by both fallback paths.

## Latency budget (measured on RTX 5070 Ti, 2026-04-22)

| Branch | Range |
|---|---|
| YOLO only | ~90 ms / call |
| YOLO-World (yolov8s-worldv2) | ~150 – 400 ms / call |
| VLM round-trip (Gemini 2.5 Flash) | 5 – 10 s / call |
| FastSAM bbox-prompted mask | ~100 ms / call |

YOLO-World keeps the open-vocab path within real-time-ish budget (sub-second total with FastSAM). Switch to `enable_vlm:=true` only when YOLO-World can't recognise the target class — Gemini is much slower and network-bound.

## Dependencies

- `object_detection_new` (base class, service callback internals)
- `kimi_api` (shared `_env.py` for API key loading)
- `tinker_vision_msgs_26` (all interfaces: new `ObjectDetectionGeneralist.srv`, legacy `ObjectDetection.srv`, shared `Object.msg`)
- Python (venv): `openai`, `python-dotenv`, `ultralytics>=8.4.33`, `numpy`, `opencv-python`

## Related services

- `/object_detection_yolo` — specialist, custom-trained competition model, `excluded_classes=['person']`. Served by `object_detection_new/yolo_seg_node`.
- `/object_detection` — pretrained COCO YOLO on the **legacy** `tinker_vision_msgs_26/srv/ObjectDetection` (string-flag schema inherited from tk23). Kept for backward compatibility with tk25_decision BTs that still hard-code this name. Served by `object_detection_new/yolo_seg_default_node`.

## Changelog

- **2026-05-02** — VLM call now streams the OpenRouter response by default (`vlm_stream=True`). Concatenates SSE `delta.content` chunks into the same JSON we parsed before, so the wire is never silent and intermediate proxies / NAT can't reap the connection mid-call. `vlm_timeout_s` becomes a per-chunk inactivity bound; abandon (race-cancel) latency drops from "up to one full HTTP attempt" to sub-100 ms because every chunk is a checkpoint. Strict→loose `response_format` fallback and `client_holder` cross-thread close path preserved unchanged. Set `-p vlm_stream:=false` to revert to the blocking call.
- **2026-05-02** — `sort_closest` now uses Euclidean distance `sqrt(x²+y²+z²)` on `Object.centroid` instead of single-axis (was sorting by `centroid.x` on realsense / `centroid.z` on orbbec). Single-axis ignored lateral / vertical offset and broke entirely on orbbec because centroids are TF-transformed to `target_frame` before sort. Fix lives in the parent class `_sort_objects_and_segments` (`object_detection_new/object_seg_yolo.py`); `generalist_node` inherits it.
- **2026-05-02** — Added `realsense_max_distance_m` (default `1.0`). Detections whose centroid sits farther than this from the realsense (arm) camera are dropped before the response is built. Applies to all three branches (yolo, yolo_world, vlm_sam) via a single shared post-build helper `_apply_realsense_range_gate`. Orbbec is untouched. Set `0.0` to disable. Same commit also documents the `force_vlm_sam ⇒ no YOLO-World` invariant in the dispatch chain (no behavior change — the dispatch was already correct, but the comment + README note guard against future regressions).
