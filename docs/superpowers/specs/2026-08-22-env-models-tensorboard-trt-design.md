# VLM model IDs from `.env`, tensorboard upgrade, FoundationStereo TensorRT revival — design

**Date:** 2026-08-22
**Status:** approved in brainstorming; implementation plan to follow
**Follows:** the 2026-08-22 dependency scan of `tk26_vision` (results in `DEV_NOTES.md`); the
user selected these three items and ruled torch unchanged and `monocular_depth` removal out of scope.

## Problems

1. **VLM model IDs are hardcoded.** Every vision node declares its model params with literal
   defaults (`google/gemini-2.5-pro`, `google/gemini-2.5-flash`, `qwen3-vl-plus`) while the
   workspace `.env` at `/home/tinker/tk25_ws/.env` already carries newer choices
   (`LLM_MODEL`, `FLASH_MODEL`) that only `kimi_api/_env.py` reads. Model choice should be a
   deployment setting, not a code edit, and vision must be able to diverge from GPSR's picks
   (its bbox/pointing prompts were calibrated on Gemini).
2. **tensorboard 2.11.2 pins protobuf to 3.20.3** in `.venv-vision-main`. tensorboard is only
   a runtime import of `torchreid.__init__`; nothing else in the venv needs protobuf 3.
3. **`.venv-fs` has no `tensorrt`**, and no FoundationStereo weights/engines exist on this box
   (`weights_root` in `foundation_stereo.yaml` points at a directory that does not exist).
   `foundation_stereo_node` only serves the `fast_trt` kind, so the node is dead here.

## Goals

- Every vision VLM model default resolves from `.env` keys with a documented fallback chain;
  explicit ROS params still override.
- `tensorboard 2.20.0` + `protobuf 6.33.6` in `.venv-vision-main`, nothing else moved.
- `tensorrt-cu12 10.16.1.11` in `.venv-fs` and working two-stage FP16 engines for the
  `23-36-37` Fast-FoundationStereo checkpoint (the one `stereo_runner.py` already maps for
  its `fast_fp16`/`fast_fp32` kinds) built on this RTX 2080 Ti, discoverable by the node
  under the existing default variant name.

## Non-goals

`monocular_depth` removal (out of this session's scope per the user); OpenCV consolidation;
openai 3.x; torch/CUDA changes; NumPy 2; `.venv-calib`; a D405 engine variant; TensorRT 11.

## Constraints (binding)

- **Venv freeze rule** (user-mandated): each venv install is `pip install --no-deps` of the
  named packages only; `pip freeze` before/after may differ only in the named lines;
  `pip check` must be clean; any other movement is a stop-and-report.
- `.venv-vision-main` and `.venv-fs` live only in the main checkout (git-ignored); code is
  edited in the worktree.
- The `tk_vision_specialized` VLM clients must not import `kimi_api` (existing decoupling
  convention) — hence the resolver lives in `vision_util`.
- The GPU is shared with a running simulator; heavy GPU work runs under `nice` and is
  sequenced, never parallel.

## Design

### 1. Model-ID resolver

New `src/vision_util/vision_util/vlm_models.py` (imports: `os` only):

```python
def vision_vlm_model() -> str:    # VISION_VLM_MODEL → LLM_MODEL → 'google/gemini-2.5-pro'
def vision_flash_model() -> str:  # VISION_VLM_FLASH_MODEL → FLASH_MODEL → 'google/gemini-2.5-flash'
def vision_qwen_model() -> str:   # VISION_QWEN_MODEL → 'qwen3-vl-plus'
```

Each reads `os.environ`, strips whitespace, and treats empty strings as unset. `.env` is
already loaded (`load_dotenv()` from CWD upward) by every VLM-using entry point before
parameters are declared; the resolver does not load `.env` itself.

`kimi_api/_env.py` re-exports the three functions; `default_model()` /
`default_flash_model()` become thin aliases of `vision_vlm_model()` /
`vision_flash_model()` (they have no non-vision callers).

**Call sites whose defaults move to the resolver** (literal → resolver, behaviour identical
when no env key is set):

| File | Symbol | Resolver |
|---|---|---|
| `kimi_api/kimi_api/seat_recommend_bbox.py` | `llm_model`, `bbox_model_gemini` params | `vision_vlm_model()` |
| `kimi_api/seat_bench/providers.py` | `GEMINI_MODEL`, `QWEN_MODEL` | vlm / qwen |
| `object_detection_generalist/.../generalist_node.py` | `vlm_model`; `vlm_fallback_models`, `dashscope_qwen_model` | `vision_flash_model()`; `f'dashscope/{vision_qwen_model()}'` |
| `object_detection_generalist/.../vlm_bbox.py` | docstring/example defaults | text only |
| `tk_vision_specialized/.../placing_location_server.py` | `vlm_model`, `placing_model_qwen` | vlm / qwen |
| `tk_vision_specialized/.../waving_person_server.py` | `vlm_model_gemini`, `vlm_model_qwen` | vlm / qwen |
| `tk_vision_specialized/.../vlm_match_client.py`, `vlm_judge_client.py`, `vlm_match_client_gemini.py`, `_waving_vlm.py` | `_GEMINI_DEFAULT_MODEL`, `_QWEN_DEFAULT_MODEL` | vlm / qwen (computed at call time, not import time, so a late `load_dotenv` still counts) |
| `tk_vision_specialized/scripts/produce_match_ground_truth.py` | `--vlm-model` default | qwen |
| `scripts/object_scan_webui/scan_core.py` | `GEMINI_MODEL`, `QWEN_MODEL` | flash / qwen |
| `scripts/compare_feature_matching_models.py` | `PRO_MODEL`, `FLASH_MODEL` | vlm / flash |
| `scripts/tests/manual/{gemini_bbox_decode,seat_recommend_vlm_bench,vlm_timeout_isolation,web_image_vlm_smoke,vlm_provider_bench}.py` | `--model` defaults | vlm or flash as today |

Nodes that declare these params log the resolved default once at INFO so a wrong `.env` is
visible in the T1 log.

**`.env` / `.env.example`:** add

```
VISION_VLM_MODEL=google/gemini-3.1-pro-preview
VISION_VLM_FLASH_MODEL=google/gemini-3.7-flash
VISION_QWEN_MODEL=qwen3-vl-plus
```

to `/home/tinker/tk25_ws/.env` (workspace root, outside this repo — the implementer appends
the keys; never prints or commits secrets) and documents all three with the fallback chain
in `src/kimi_api/.env.example`. Qwen stays on `qwen3-vl-plus`: DashScope's newer
`qwen3.7-plus` is a thinking-by-default hybrid model the calibrated bbox decoder has not
been benchmarked against.

**Tests:** `src/vision_util/test/test_vlm_models.py` — precedence and empty-string handling
via `monkeypatch.setenv/delenv`; per-package tests asserting each declared default equals the
resolver's value under a controlled env (import the module with the env set; no ROS spin).

### 2. tensorboard / protobuf

`.venv-vision-main`: `pip install --no-deps tensorboard==2.20.0 tensorboard-data-server==0.7.2
protobuf==6.33.6`. Freeze diff may contain exactly these three changed lines. `pip check`
clean. Orphans from 2.11 (`google-auth`, `google-auth-oauthlib`, `requests-oauthlib`,
`tensorboard-plugin-wit`) are left (same policy as the jax orphans).

Gate: `import torchreid` and `vision_track.reid.reid_backbone` building `osnet_ain_x1_0`
(imagenet init) succeed; `src/vision_track/test` suite green. Pins in
`.venv-vision-main.uv-project/pyproject.toml`, root `requirements.txt`, `freeze.lock.txt`;
`uv lock` re-run only if its diff is confined to those packages.

### 3. FoundationStereo TensorRT

1. **Install** into `.venv-fs`: `pip install --no-deps --extra-index-url https://pypi.nvidia.com
   tensorrt-cu12==10.16.1.11 tensorrt-cu12-bindings==10.16.1.11 tensorrt-cu12-libs==10.16.1.11`.
   Freeze diff: exactly those three additions. `import tensorrt` → `10.16.1.11`;
   `src/foundation_stereo/test/test_stereo_runner_imports.py` green. `unlockable.txt` gains a
   "installed 2026-08-22" note; `freeze.lock.txt` refreshed.
2. **Weights root** moves to `~/.cache/tk26_vision/weights/foundation_stereo`.
   `StereoRunner.__init__` applies `os.path.expanduser` to `weights_root` (today only the
   `$FOUNDATION_STEREO_VENDOR_ROOT` override is expanded). Layout follows the runner's
   existing `_fast_pickle` path:
   `…/foundation_stereo/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
   and `…/Fast-FoundationStereo/output_two_stage/{feature_runner.engine, post_runner.engine,
   onnx.yaml}`. `foundation_stereo.yaml` updated.
3. **`src/foundation_stereo/scripts/fetch_fast_fs_weights.py`** — `gdown --folder` of the
   upstream Drive folder (readme link), keeping only the `23-36-37` checkpoint; idempotent;
   writes `SHA256SUMS` after the first successful download and verifies on later runs. If
   Drive is unreachable or the folder layout differs, it exits non-zero with the exact error —
   no workaround.
4. **`src/foundation_stereo/scripts/build_trt_engines.py`** — runs the vendored
   `Fast-FoundationStereo/scripts/make_onnx.py` on the `23-36-37` pickle (`--height 576
   --width 960 --valid_iters 4 --max_disp 192` — the resolution/iteration configuration the
   previous `…_576x960_iters4` engines used; both dims are multiples of 32 as the exporter
   asserts) into a temp dir, then builds
   each ONNX with the TensorRT Python API (`Builder`, `OnnxParser`, explicit batch, `FP16`
   flag, `set_memory_pool_limit(WORKSPACE, 4 GiB)`), writes the serialized engines plus
   `onnx.yaml` into `output_two_stage/` (refuses to overwrite without `--force`), and prints
   build time. `trtexec` is not used (not shipped in the pip wheels).
5. **Gate:** `ros2 launch foundation_stereo foundation_stereo.launch.py warmup_on_launch:=true`
   logs a successful warmup forward (engine load + zero 480×848 pair through
   resize → engines → resize back); a `get_depth` service call if the RealSense is publishing.
   Warmup forward time recorded in DEV_NOTES.

Risks, stated: Drive availability; `make_onnx.py` under torch 2.8 / triton 3.4 on cc 7.5;
engine build time on the shared 2080 Ti (run under `nice -n 10`, budget 20 min); FP16 accuracy
on Turing vs upstream's Ampere numbers is not validated here beyond the node's own warmup.

### 4. Verification & docs

Final pass: `scripts/tests/t0_static.sh`; unit suites of `vision_util`, `kimi_api`,
`tk_vision_specialized`, `object_detection_generalist`, `vision_track`, `foundation_stereo`
(lint tests excluded as repo-wide pre-existing failures). Docs: `CLAUDE.md` (OpenRouter
section lists the three keys; node bullets say "default from `VISION_*`"; tensorboard/protobuf
and `.venv-fs` sentences corrected), `src/kimi_api/.env.example`,
`src/foundation_stereo/README.md` (provisioning rewritten around the two scripts and the new
`weights_root`), `docs/ENVIRONMENT.md` pins, dated `DEV_NOTES.md` entry with the scan results,
freeze diffs, and the warmup timing.

## Acceptance

- With `VISION_*` unset, every node's declared default equals today's literal (parity test).
- With the new `.env`, `ros2 param get` on each node shows the `.env` value.
- `.venv-vision-main` freeze diff = {tensorboard, tensorboard-data-server, protobuf}; `pip check` clean.
- `.venv-fs` freeze diff = {tensorrt-cu12, -bindings, -libs}; `pip check` clean.
- FoundationStereo warmup succeeds on this box with the locally built engines.
