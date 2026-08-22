# tk26_vision — Python Environment Setup

> Part of the tk25_ws workspace. Full safety doctrine, prerequisites, untracked-sidecar staging, and
> cross-module build order live in the canonical guide: `src/tk25_basic/docs/ENV_SETUP.md` (tk25_basic repo).

This module owns **three** GPU venvs, deliberately split because their numpy/torch/tensorrt ABIs are
mutually incompatible:

| venv | Python | torch | numpy | why isolated |
|---|---|---|---|---|
| `.venv-vision-main` | 3.10.12 | 2.11.0+cu128 | 1.26.4 | the shared vision tree (detection, tracking, pan-tilt, kimi_api, utils) |
| `.venv-da3` | 3.10.12 | 2.11.0+cu128 | **1.23.4** | `depth_anything_3` pins `numpy<2` — installing it into the shared venv would cascade-break every other vision node |
| `.venv-fs` | **3.10.20** | **2.8.0+cu128** | 1.26.4 | `torch 2.8` + `tensorrt 10.16` conflict with both other venvs; the only non-3.10.12 venv in the whole workspace |

**All three are git-UNTRACKED** — their `*.uv-project/` sidecars are absent on a fresh clone and
`restore_venvs.sh` silently `[skip]`s them. **Stage the sidecars first** (see the hub guide,
`src/tk25_basic/docs/ENV_SETUP.md` §3) before any restore. All three are then in `restore.sh`'s
`VENVS` list and are restored by it.

## Safety doctrine (summary)

Never touch `/usr/bin/python3`, `~/.local`, or `pip --user`. Every interpreter is a uv-managed
CPython (`export UV_PYTHON_PREFERENCE=only-managed` forces uv to provision its own under
`~/.local/share/uv/python`, never the system one) and every install is venv-scoped — either
`uv sync --frozen` from a sidecar lock or `uv pip install --python <venv>/bin/python <pkg>`. **Never a
bare `pip install`.** Full rationale + the "did anything leak into system python" audit live in the
hub guide at `src/tk25_basic/docs/ENV_SETUP.md` §1 and §8.

## Prerequisites (this module)

- **uv** (0.10.x); `export UV_PYTHON_PREFERENCE=only-managed` exported in your shell profile.
- **ROS 2 Humble at exactly `/opt/ros/humble`** — the `ros2_packages.pth` shim that `restore_venvs.sh`
  writes into each venv hardcodes that prefix + python3.10.
- **System `/usr/bin/python3 == 3.10`** — `tkbuild` hard-aborts if the venv python-minor differs from
  the system python-minor.
- **NVIDIA driver + CUDA 12.8 / Blackwell-class GPU** — all three venvs are GPU venvs (they install
  cu128 torch wheels; no system CUDA toolkit is needed since the wheels bundle their own runtime libs).
- **Sidecar staging done** — the three `*.uv-project/` dirs copied onto the target (hub guide §3),
  plus the on-disk source trees these venvs consume at sync time:
  `thirdparty/depth-anything-3/` (DA3 directory source), `src/foundation_stereo/`, and
  `thirdparty/foundation_stereo/{FoundationStereo,Fast-FoundationStereo}/`.

## Environment(s)

### `src/tk26_vision/.venv-vision-main` — Python 3.10.12 — GPU

The shared vision venv. Covers `object_detection_new`, `object_detection_generalist`, `vision_track`,
`tk_vision_specialized`, `pan_tilt`, `kimi_api`, `vision_util`. torch **2.11.0+cu128** (+ torchvision
0.26.0+cu128, triton 3.6.0), numpy **1.26.4**.

- **Sidecar tracked on a fresh clone?** No — **UNTRACKED**; stage `.venv-vision-main.uv-project/`
  first (hub guide §3). Handled by `restore_venvs.sh`? **Yes** (once the sidecar is present).
- **Restore:**
  ```bash
  export UV_PYTHON_PREFERENCE=only-managed
  cd /home/tinker/tk25_ws
  ./src/tk25_basic/tools/restore_venvs.sh vision-main      # bare positional substring matches ONLY this venv
  # Explicit equivalent:
  #   UV_PROJECT_ENVIRONMENT=/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main \
  #     uv sync --project /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main.uv-project \
  #             --no-install-project --frozen
  ```
- **Unlockables:** the torchreid OSNet ReID chain — **not in the lock** (no `unlockable.txt` either;
  these are a hand-install) and without it `vision_track`'s ReID regresses to imagenet-init only.
  Pinned to the live `freeze.lock.txt`; the install is zero-churn (numpy stays 1.26.4, torch stays
  2.11.0+cu128). Run **after** `uv sync`, venv-scoped:
  ```bash
  uv pip install --python /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python \
    torchreid==0.2.5 gdown==6.1.0 tensorboard==2.20.0 tensorboard-data-server==0.7.2
  ```
  `tensorboard` is a runtime import dep of torchreid's `__init__` (loads the training engine), so it
  is required even for inference-only use — no `--no-deps` deviation.
  **protobuf 6.33.6 (2026-08-22):** unlike tensorboard, `protobuf` **is** a locked dependency —
  `.venv-vision-main.uv-project/pyproject.toml` and `uv.lock` pin it at `protobuf==6.33.6` (was
  `3.20.3`, the version tensorboard 2.11.2 used to pin transitively), so a plain `uv sync --frozen`
  brings the right version without a hand-install. The `--no-deps` bump that produced this pin moved
  exactly `protobuf`, `tensorboard`, and `tensorboard-data-server`; `pip check` was unchanged before
  vs. after.
- **Editable/git sources needing a local tree:** `clip` is a **git dep**
  (`clip @ git+https://github.com/ultralytics/CLIP.git@81ff68ed7…`) resolved + built by `uv sync` —
  needs network to github.com at sync time; **no local tree**. No editable/directory sources.
- **Verify:**
  ```bash
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c \
    "import ultralytics, cv2, clip, torchreid, openai, dotenv, serial, fastapi, numpy; \
     from torchreid.reid.models import build_model; \
     print('ultralytics', ultralytics.__version__, 'numpy', numpy.__version__, 'torchreid build_model OK')"
  # expect numpy 1.26.4
  ```
- **Gotchas:**
  - torch/cu128 resolves from the `pytorch-cu128` index baked into the lock — **never** hand
    `pip install torch`. **numpy must stay 1.26.4** (the torch 2.11 / scipy / opencv ABI; the torchreid
    add was validated zero-churn).
  - **Do NOT** build `monocular_depth` (→ `.venv-da3`) or `foundation_stereo` (→ `.venv-fs`) against
    this venv.
  - Per-package `requirements.txt` (`kimi_api`, `pan_tilt`, `vision_track`) install into this venv —
    route them venv-scoped, e.g.
    `uv pip install --python …/.venv-vision-main/bin/python -r src/tk26_vision/src/vision_track/requirements.txt`,
    never a bare activated `pip`.
  - **Optional runtime asset (recommended for arena):** MSMT17-trained OSNet ReID weights via
    `./src/tk26_vision/scripts/fetch_reid_weights.sh` → `~/.cache/torch/checkpoints/` (idempotent; the
    tracker auto-loads them over the imagenet init). `drive.google.com` is unreachable from sandboxed
    CI, so pre-warm on a connected host before offline runs.
  - Optional `OPENROUTER_API_KEY` in the workspace-root `.env` for `kimi_api` LLM nodes
    (`cp src/tk26_vision/src/kimi_api/.env.example .env`).
  - `restore_venvs.sh` writes the `ros2_packages.pth` ROS shim into the venv (hardcodes
    `/opt/ros/humble` + python3.10).

### `src/tk26_vision/.venv-da3` — Python 3.10.12 — GPU (monocular_depth / Depth-Anything-3)

Isolated because `depth_anything_3` pins `numpy<2`. torch **2.11.0+cu128** (+ torchvision
0.26.0+cu128), numpy **1.23.4**.

- **Sidecar tracked on a fresh clone?** No — **UNTRACKED**; stage `.venv-da3.uv-project/` first.
  Handled by `restore_venvs.sh`? **Yes** (once staged).
- **Restore:**
  ```bash
  export UV_PYTHON_PREFERENCE=only-managed
  # Ensure the DA3 directory-source tree exists first — the lock pins
  #   depth-anything-3 @ file:///home/tinker/tk25_ws/src/tk26_vision/thirdparty/depth-anything-3
  test -f /home/tinker/tk25_ws/src/tk26_vision/thirdparty/depth-anything-3/pyproject.toml \
    || echo 'STAGE thirdparty/depth-anything-3/ before restore (hub guide §3)'
  cd /home/tinker/tk25_ws
  ./src/tk25_basic/tools/restore_venvs.sh .venv-da3        # bare positional substring
  ```
- **Unlockables:** none — `depth_anything_3` is a directory source covered by `uv sync`; no
  `unlockable.txt`.
- **Editable/git sources needing a local tree:** `depth-anything-3` is a **file-URL directory source**
  at `src/tk26_vision/thirdparty/depth-anything-3` — required on disk **before** sync. The lock has no
  `editable=true`, so it is installed as a **built copy, not editable**. The vendored
  `tk26_vision patch:` in `thirdparty/depth-anything-3/src/depth_anything_3/api.py` (defers the heavy
  export-pipeline imports to call time) must be present before sync; editing the tree later requires a
  re-sync:
  ```bash
  uv pip install --python /home/tinker/tk25_ws/src/tk26_vision/.venv-da3/bin/python \
    /home/tinker/tk25_ws/src/tk26_vision/thirdparty/depth-anything-3
  ```
- **Verify:**
  ```bash
  /home/tinker/tk25_ws/src/tk26_vision/.venv-da3/bin/python -c \
    "import numpy, torch, torchvision, depth_anything_3 as da3; \
     print('numpy', numpy.__version__, 'torch', torch.__version__)"
  # expect numpy 1.23.4 ; torch 2.11.0+cu128
  ```
- **Gotchas:**
  - **numpy pinned 1.23.4** — never bump above 2 here.
  - **Runtime asset (not pip):** DA3 weights (default `depth-anything/DA3-SMALL`) auto-download to
    `~/.cache/huggingface/hub` on first inference.
  - Build via `build_monocular_depth.sh` (or `tkbuild tk26_vision --packages-select monocular_depth`),
    **never** the main `build.sh` — it would resolve `depth_anything_3` against the wrong venv.

### `src/tk26_vision/.venv-fs` — Python 3.10.20 — GPU (foundation_stereo)

The **only non-3.10.12 venv** in the workspace. torch **2.8.0+cu128** (+ torchvision 0.23.0+cu128,
torchaudio 2.8.0+cu128, triton 3.4.0), numpy **1.26.4** (pinned via the sidecar's
`override-dependency`).

- **Sidecar tracked on a fresh clone?** No — **UNTRACKED**; stage `.venv-fs.uv-project/` (4 files incl.
  `unlockable.txt`) first. Handled by `restore_venvs.sh`? **Yes** (once staged) — uv downloads managed
  CPython 3.10.20, so keep `only-managed`.
- **Restore + the full post-restore sequence:**
  ```bash
  export UV_PYTHON_PREFERENCE=only-managed
  # Ensure the package + vendored model trees exist first:
  #   src/tk26_vision/src/foundation_stereo
  #   src/tk26_vision/thirdparty/foundation_stereo/{FoundationStereo,Fast-FoundationStereo}
  cd /home/tinker/tk25_ws

  # STEP 1 — restore (uv fetches managed 3.10.20; writes ros2_packages.pth):
  ./src/tk25_basic/tools/restore_venvs.sh .venv-fs         # bare positional substring

  # STEP 2 — UNLOCKABLE: TensorRT (venv-scoped, AFTER torch; the wheel bundles -bindings + -libs):
  uv pip install --python /home/tinker/tk25_ws/src/tk26_vision/.venv-fs/bin/python \
    --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.16.1.11

  # STEP 3 — build the ROS pkg (writes build/foundation_stereo via colcon develop-install):
  ./src/tk26_vision/scripts/build_foundation_stereo.sh

  # STEP 4 — CRITICAL .pth shim. colcon writes the egg-link but NOT easy-install.pth, and neither
  #          restore_venvs.sh nor build_foundation_stereo.sh recreates this — re-add it AFTER EVERY
  #          'uv sync --force' / venv wipe / FS rebuild, once build/foundation_stereo exists.
  #          Content = exactly one line: the build dir path.
  printf '%s\n' /home/tinker/tk25_ws/build/foundation_stereo \
    > /home/tinker/tk25_ws/src/tk26_vision/.venv-fs/lib/python3.10/site-packages/foundation_stereo.pth
  ```
- **Unlockables:** `tensorrt-cu12==10.16.1.11` (NVIDIA index; the wheel pulls `-bindings` + `-libs`
  automatically). It is listed commented-out in the sidecar's `unlockable.txt`; install it by hand
  per STEP 2. **Installed 2026-08-22** on this box — `unlockable.txt` carries an `# INSTALLED
  2026-08-22 on tinker (RTX 2080 Ti)` note under the entry. The actual `--no-deps` install used to
  land it named all three packages explicitly (`tensorrt-cu12==10.16.1.11
  tensorrt-cu12-bindings==10.16.1.11 tensorrt-cu12-libs==10.16.1.11`); freeze diff was exactly those
  three additions, `pip check` unchanged. **`freeze.lock.txt` now exists for this venv** (427 lines,
  git-ignored, same convention as `.venv-vision-main`/`.venv-da3`) and is the diff-target for any
  further install. The `-libs` wheel is ~4.3 GB — see the README's provisioning note about
  `--no-cache-dir` and the "no pytest of its own" test-invocation workaround.
- **Editable/git sources needing a local tree:** the `foundation_stereo` ROS package itself —
  installed by `build_foundation_stereo.sh` as a colcon develop-install, wired into the venv via the
  STEP-4 `foundation_stereo.pth` pointing at `/home/tinker/tk25_ws/build/foundation_stereo`. Also
  needs the vendored model trees `thirdparty/foundation_stereo/{FoundationStereo,Fast-FoundationStereo}`
  on disk.
- **Verify:**
  ```bash
  /home/tinker/tk25_ws/src/tk26_vision/.venv-fs/bin/python -c \
    "import torch, tensorrt, numpy, cv2; \
     print('torch', torch.__version__, 'trt', tensorrt.__version__, 'np', numpy.__version__); \
     assert torch.__version__.startswith('2.8.0') and tensorrt.__version__=='10.16.1.11' \
            and numpy.__version__=='1.26.4'"
  # confirm the .pth too:
  test "$(cat /home/tinker/tk25_ws/src/tk26_vision/.venv-fs/lib/python3.10/site-packages/foundation_stereo.pth)" \
       = /home/tinker/tk25_ws/build/foundation_stereo && echo 'foundation_stereo.pth OK'
  ```
- **Gotchas:**
  - **numpy pinned 1.26.4** (sidecar `override-dependency`) — numpy 2.x segfaults the system
    `cv_bridge.boost`. Do not bump.
  - The STEP-4 `foundation_stereo.pth` is **fragile across `--force`** — it is the single most common
    "FoundationStereo imports nothing" failure. Re-add it after any venv wipe.
  - **Runtime asset:** FoundationStereo TRT **engines** at `weights_root` (default
    `~/.cache/tk26_vision/weights/foundation_stereo`, user-expanded) are GPU- and TRT-version-locked
    and **must be re-exported on the target GPU** — copying engines across machines won't work (and
    the D435 50 mm-baseline engine is geometrically wrong on D405). Fetch the checkpoint with
    `fetch_fast_fs_weights` and build the two-stage FP16 engines with `build_trt_engines`; see
    `src/foundation_stereo/README.md` § Weights & engines for the exact commands, the resulting
    layout, and the measured build/warmup timings on this box.
  - Build only via `build_foundation_stereo.sh` (or `tkbuild tk26_vision --packages-select
    foundation_stereo`).

## Build

`tkbuild` builds the whole sub-workspace, routing each package to its correct venv automatically
(Phase 1.5 `PER_PKG_VENV_BY_WS`: `monocular_depth`→`.venv-da3`, `foundation_stereo`→`.venv-fs`; the
rest → `.venv-vision-main`):

```bash
cd /home/tinker/tk25_ws
./tkbuild tk26_vision
```

Standalone equivalents (what `tkbuild tk26_vision` runs under the hood):

```bash
# Everything under .venv-vision-main, then re-shebang install-tree entry points to the venv python:
./src/tk26_vision/scripts/build.sh                    # forwards colcon args, e.g. --packages-select pan_tilt kimi_api

# monocular_depth under .venv-da3 (defaults to --packages-select monocular_depth; re-shebangs entry point):
./src/tk26_vision/scripts/build_monocular_depth.sh

# foundation_stereo under .venv-fs (defaults to --packages-select foundation_stereo; re-shebangs entry point).
# Re-add foundation_stereo.pth afterward (Environment §.venv-fs STEP 4):
./src/tk26_vision/scripts/build_foundation_stereo.sh
```

- Plain `colcon build` writes `#!/usr/bin/python3` shebangs that can't see venv-only deps (`openai`,
  `dotenv`, `ultralytics`, `pyserial`, …) — **always** use the wrappers / `tkbuild`. If you must run
  raw colcon, follow up with `./src/tk26_vision/scripts/fix_venv_shebangs.sh` (idempotent; covers all
  tk26 packages).
- **Do NOT** pass `monocular_depth` or `foundation_stereo` to the main `build.sh` — it resolves their
  numpy<2 / torch-2.8 deps against the wrong venv and starts the entry point under the wrong python.
- If a build errors on stale symlinks: `rm -rf build/<pkg> install/<pkg>` and rebuild that package.