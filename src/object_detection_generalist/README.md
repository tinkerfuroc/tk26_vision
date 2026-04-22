# object_detection_generalist

Open-vocabulary object detection for tk26 vision. Runs a clean pretrained YOLO for classes YOLO already knows; for anything else (or when the caller explicitly opts in), falls back to **Gemini 2.5 Pro** for bounding boxes and **FastSAM** for segmentation masks. 3D centroid + camera-sync logic is inherited unchanged from `object_detection_new.object_seg_yolo.YOLOSegmentationNode`.

## Interface

Service: `/object_detection_generalist`, type [`tinker_vision_msgs_26/srv/ObjectDetectionGeneralist`](../tinker_vision_msgs_26/srv/ObjectDetectionGeneralist.srv).

This is a **new srv** with typed boolean flags — not the legacy `tinker_vision_msgs_26/srv/ObjectDetection` (same package, different type) that `/object_detection` and `/object_detection_yolo` still accept. See [`tinker_vision_msgs_26/README.md`](../tinker_vision_msgs_26/README.md#objectdetectionsrv-field-mapping-from-tk23--tk26) for the field mapping.

Request fields that matter for path selection:

| Field | Meaning |
|---|---|
| `prompt` | Natural-language or class name. Must match one of `model.names.values()` for the YOLO branch; anything else forces the VLM+SAM branch. |
| `force_vlm_sam` | Skip YOLO entirely, always use VLM+SAM. |
| `use_vlm_sam_fallback` | Per-request opt-in for the VLM+SAM path when `prompt` is out of vocabulary. |
| `sort_closest` / `sort_highest` | Typed boolean replacements for the old substring-parsed `flags` string. |
| `return_rgb_image` / `return_depth_image` / `return_segments` | Payload toggles (save bandwidth when false). |

Response adds `string detection_source ∈ {"yolo", "vlm_sam", "none"}` so the caller can tell which branch answered.

## Branching logic

```
if request.force_vlm_sam:                        # operator override
    branch = vlm_sam
elif prompt ∈ model.names:                       # YOLO knows this class
    branch = yolo
    if yolo returns [] AND (request.use_vlm_sam_fallback OR node.allow_auto_fallback):
        branch = vlm_sam                          # fall-through
elif request.use_vlm_sam_fallback OR node.allow_auto_fallback:
    branch = vlm_sam
else:
    status=1, detection_source=none, error_msg="class not in YOLO names and fallback disabled"
```

`allow_auto_fallback` is a node-level ROS parameter (default `True`). Flip it off at launch (`-p allow_auto_fallback:=false`) if the caller must opt in per-request.

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
| `vlm_model` | `google/gemini-2.5-pro` | OpenRouter model tag. Override per deployment via `-p vlm_model:=anthropic/claude-sonnet-4-6` etc. |
| `vlm_timeout_s` | `20.0` | Per-VLM-call timeout. |
| `vlm_max_retries` | `3` | JSON parse / API retries before returning empty. |
| `allow_auto_fallback` | `True` | See branching above. |
| `orbbec_depth_topic` | `/camera/depth_registered/points` | Must match what the camera launch publishes. For the canonical Femto Bolt launch this is `/camera/depth/points` — override accordingly. |

## Example calls

```bash
# YOLO branch (chair is a COCO class)
ros2 service call /object_detection_generalist tinker_vision_msgs_26/srv/ObjectDetectionGeneralist \
  "{camera: realsense, prompt: chair}"

# Open-vocabulary via VLM+SAM
ros2 service call /object_detection_generalist tinker_vision_msgs_26/srv/ObjectDetectionGeneralist \
  "{camera: realsense, prompt: 'monitor screen', use_vlm_sam_fallback: true}"

# Force VLM+SAM even on a COCO class
ros2 service call /object_detection_generalist tinker_vision_msgs_26/srv/ObjectDetectionGeneralist \
  "{camera: realsense, prompt: bottle, force_vlm_sam: true}"
```

## Architecture

- `generalist_node.py` — `GeneralistDetectionNode` subclasses `YOLOSegmentationNode`; overrides `_init_service` / `_detection_service_callback`. Camera sync, TF, depth projection, 3D centroid, and sort-mode logic are inherited, not duplicated.
- `vlm_bbox.py` — `request_bboxes(rgb, prompt, model, …)` sends a base64 data-URL image + strict-JSON system prompt to OpenRouter, retries on parse failure, decodes Gemini's `[y0,x0,y1,x1]` 0–1000-normalized output into xyxy pixel coords. Key loading is deferred until first call via `kimi_api._env.require_api_key`, so the node starts cleanly without a key.
- `sam_mask.py` — `FastSAMPredictor` wraps `ultralytics.FastSAM`, loads once at node init on the parent's `self.device` (GPU if available), exposes `segment(rgb, bboxes) -> list[bool HxW mask]` aligned 1:1 with the input bboxes.

## Latency budget (measured on RTX 5070 Ti, 2026-04-22)

| Branch | Range |
|---|---|
| YOLO only | ~90 ms / call |
| VLM round-trip (Gemini 2.5 Pro) | 9 – 14 s / call |
| FastSAM bbox-prompted mask | ~100 ms / call |

VLM dominates and is network-bound. Don't put the generalist on a hot path — it's for open-vocabulary cases that the specialist can't cover.

## Dependencies

- `object_detection_new` (base class, service callback internals)
- `kimi_api` (shared `_env.py` for API key loading)
- `tinker_vision_msgs_26` (all interfaces: new `ObjectDetectionGeneralist.srv`, legacy `ObjectDetection.srv`, shared `Object.msg`)
- Python (venv): `openai`, `python-dotenv`, `ultralytics>=8.4.33`, `numpy`, `opencv-python`

## Related services

- `/object_detection_yolo` — specialist, custom-trained competition model, `excluded_classes=['person']`. Served by `object_detection_new/yolo_seg_node`.
- `/object_detection` — pretrained COCO YOLO on the **legacy** `tinker_vision_msgs_26/srv/ObjectDetection` (string-flag schema inherited from tk23). Kept for backward compatibility with tk25_decision BTs that still hard-code this name. Served by `object_detection_new/yolo_seg_default_node`.
