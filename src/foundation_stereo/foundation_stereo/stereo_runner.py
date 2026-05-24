"""FoundationStereo + Fast-FoundationStereo runner — ROS2-stripped port.

Lifted from dualrRGB-foundationStereo/webapp/stereo_runner.py with the
overhead cuts from docs/superpowers/specs/2026-05-24-foundation-stereo-design.md §10:
- no nvidia-smi subprocess
- no PyTorch peak-memory counters
- CUDA-event timing is opt-in via measure_forward_ms

Single-slot model cache, evicted on backend or TRT-variant switch. One
internal threading.Lock serializes GPU access across all callers (service,
action, streaming worker).
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


# Resolve the vendored thirdparty tree. This file lives at
# src/tk26_vision/src/foundation_stereo/foundation_stereo/stereo_runner.py;
# the vendor root is ../../../thirdparty/foundation_stereo/.
_THIS = os.path.dirname(os.path.realpath(__file__))
_VENDOR_ROOT = os.path.realpath(
    os.path.join(_THIS, "..", "..", "..", "thirdparty", "foundation_stereo")
)
_FS_DIR = os.path.join(_VENDOR_ROOT, "FoundationStereo")
_FAST_DIR = os.path.join(_VENDOR_ROOT, "Fast-FoundationStereo")


def _swap_namespace(target_dir: str) -> None:
    """Evict cached `core.*` / `Utils` modules and put `target_dir` at sys.path[0].

    Both FoundationStereo and Fast-FoundationStereo ship a top-level `core/`
    package + `Utils.py` with the same module names but different classes.
    Without this swap, importing one after the other leaks the cached
    classes.
    """
    for name in list(sys.modules):
        if name == "Utils" or name == "core" or name.startswith("core."):
            del sys.modules[name]
    for d in (_FS_DIR, _FAST_DIR):
        while d in sys.path:
            sys.path.remove(d)
    sys.path.insert(0, target_dir)


def _discover_trt_variants(weights_root: str) -> dict:
    """Find any directory under `<weights_root>/Fast-FoundationStereo/` that
    contains a complete two-stage TRT engine set."""
    fast_root = os.path.join(weights_root, "Fast-FoundationStereo")
    out = {}
    if not os.path.isdir(fast_root):
        return out
    for entry in sorted(os.listdir(fast_root)):
        d = os.path.join(fast_root, entry)
        if not os.path.isdir(d):
            continue
        needed = ("feature_runner.engine", "post_runner.engine", "onnx.yaml")
        if all(os.path.exists(os.path.join(d, f)) for f in needed):
            out[entry] = d
    return out


_DEFAULT_ITERS = {"vitl": 32, "vits": 32, "fast_fp32": 8, "fast_fp16": 8}

# Filled in lazily by `StereoRunner.__init__` when `weights_root` is known.
TRT_VARIANTS: dict = {}


@dataclass
class InferResult:
    disp: np.ndarray
    depth: np.ndarray
    vis_jpg: bytes                  # JPEG-encoded disparity vis; empty if not requested
    scale_used: float
    load_s: float = 0.0
    forward_ms: float = 0.0         # 0.0 if measure_forward_ms=False
    forward_s: float = 0.0          # always populated (wall clock)
    post_s: float = 0.0


class StereoRunner:
    def __init__(self, weights_root: str):
        self._weights_root = weights_root
        self._fs_pretrained = os.path.join(
            weights_root, "FoundationStereo", "pretrained_models"
        )
        self._fast_pickle = os.path.join(
            weights_root, "Fast-FoundationStereo", "weights",
            "23-36-37", "model_best_bp2_serialize.pth",
        )
        global TRT_VARIANTS
        TRT_VARIANTS = _discover_trt_variants(weights_root)
        self._default_trt_variant = (
            "output_two_stage" if "output_two_stage" in TRT_VARIANTS
            else next(iter(TRT_VARIANTS), None)
        )
        self._ckpt_map = {
            "vitl":      os.path.join(self._fs_pretrained, "23-51-11", "model_best_bp2.pth"),
            "vits":      os.path.join(self._fs_pretrained, "11-33-40", "model_best_bp2.pth"),
            "fast_fp32": self._fast_pickle,
            "fast_fp16": self._fast_pickle,
            "fast_trt":  TRT_VARIANTS.get(self._default_trt_variant)
                         if self._default_trt_variant else None,
        }

        self._model = None
        self._model_kind: Optional[str] = None
        self._trt_variant: Optional[str] = None
        self._trt_input_hw: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    @property
    def current_model(self) -> Optional[str]:
        return self._model_kind

    @property
    def current_trt_variant(self) -> Optional[str]:
        return self._trt_variant

    def _resolve_variant(self, kind: str, variant: Optional[str]) -> Optional[str]:
        if kind != "fast_trt":
            return None
        if variant in TRT_VARIANTS:
            return variant
        return self._default_trt_variant

    def _ensure_model(self, kind: str, variant: Optional[str] = None) -> None:
        assert kind in self._ckpt_map, f"unknown model kind {kind}"
        resolved = self._resolve_variant(kind, variant)
        cache_key = (kind, resolved)
        current_key = (self._model_kind, self._trt_variant)
        if cache_key == current_key and self._model is not None:
            return

        if self._model is not None:
            logging.info(f"[stereo_runner] freeing {self._model_kind} model")
            del self._model
            self._model = None
            self._model_kind = None
            self._trt_variant = None
            self._trt_input_hw = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.autograd.set_grad_enabled(False)

        if kind in ("fast_fp32", "fast_fp16"):
            pickle = self._fast_pickle
            if not os.path.isfile(pickle):
                raise FileNotFoundError(f"weights missing: {pickle}")
            logging.info(f"[stereo_runner] loading {kind} from {pickle}")
            _swap_namespace(_FAST_DIR)
            model = torch.load(pickle, map_location="cpu", weights_only=False)
            model.cuda().eval()

        elif kind == "fast_trt":
            if resolved is None:
                raise RuntimeError(
                    f"no two-stage TRT engines found under "
                    f"{self._weights_root}/Fast-FoundationStereo/"
                )
            trt_dir = TRT_VARIANTS[resolved]
            logging.info(f"[stereo_runner] loading {kind} variant={resolved} from {trt_dir}")
            _swap_namespace(_FAST_DIR)
            cfg = OmegaConf.load(os.path.join(trt_dir, "onnx.yaml"))
            from core.foundation_stereo import TrtRunner  # noqa: WPS433
            feat_eng = os.path.join(trt_dir, "feature_runner.engine")
            post_eng = os.path.join(trt_dir, "post_runner.engine")
            model = TrtRunner(cfg, feat_eng, post_eng).cuda().eval()
            self._trt_input_hw = (int(cfg.image_size[0]), int(cfg.image_size[1]))
            self._trt_variant = resolved

        elif kind in ("vitl", "vits"):
            ckpt_path = self._ckpt_map[kind]
            cfg_yaml = os.path.join(os.path.dirname(ckpt_path), "cfg.yaml")
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"weights missing: {ckpt_path}")
            logging.info(f"[stereo_runner] loading {kind} from {ckpt_path}")
            _swap_namespace(_FS_DIR)
            cfg = OmegaConf.load(cfg_yaml)
            if "vit_size" not in cfg:
                cfg["vit_size"] = "vitl"
            cfg["mixed_precision"] = True
            cfg["valid_iters"] = 32
            cfg["hiera"] = 0
            cfg["low_memory"] = 0
            cfg["corr_implementation"] = cfg.get("corr_implementation", "reg")
            args = OmegaConf.create(cfg)

            from core.foundation_stereo import FoundationStereo  # noqa: WPS433
            model = FoundationStereo(args)
            ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model.cuda().eval()

        else:
            raise ValueError(f"unknown kind: {kind}")

        self._model = model
        self._model_kind = kind

    def infer(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        K: np.ndarray,
        baseline: float,
        kind: str = "fast_trt",
        scale: float = 0.5,
        valid_iters: Optional[int] = None,
        z_far: float = 10.0,
        remove_invisible: bool = True,
        trt_variant: Optional[str] = None,
        live: bool = False,
        measure_forward_ms: bool = True,
        want_debug_jpeg: bool = False,
    ) -> InferResult:
        """Run one inference.

        Args:
          left_rgb, right_rgb: (H, W, 3) uint8 stereo pair (RGB order).
          K: (3, 3) intrinsics of the *left* camera, at original
             resolution. Used to derive depth from disparity after scaling.
          baseline: stereo baseline in metres (positive).
          kind: model kind; see _DEFAULT_ITERS keys + 'fast_trt'.
          scale: image-resize factor before inference. Ignored for fast_trt
             (engine input shape is baked).
          valid_iters: per-backend iteration count override.
          z_far: depth clamp in metres.
          remove_invisible: drop pixels whose match would lie outside the
             right image (the reference flag from the upstream demo).
          trt_variant: directory basename inside Fast-FoundationStereo/.
          live: when True, skip depth math / point-cloud build entirely.
             Returns only disparity + JPEG vis.
          measure_forward_ms: when True, record CUDA events around
             model.forward to populate `forward_ms`. ~100 µs sync cost.
          want_debug_jpeg: when True, JPEG-encode the disparity vis into
             InferResult.vis_jpg.
        """
        assert left_rgb.ndim == 3 and left_rgb.shape[2] == 3
        assert right_rgb.shape == left_rgb.shape

        with self._lock:
            resolved = self._resolve_variant(kind, trt_variant)
            cache_key = (kind, resolved)
            current_key = (self._model_kind, self._trt_variant)
            need_load = (cache_key != current_key) or (self._model is None)
            t_load = time.time()
            self._ensure_model(kind, variant=trt_variant)
            load_s = (time.time() - t_load) if need_load else 0.0

            return self._run(
                left_rgb, right_rgb, K, baseline, scale, valid_iters, z_far,
                remove_invisible, live=live,
                measure_forward_ms=measure_forward_ms,
                want_debug_jpeg=want_debug_jpeg, load_s=load_s,
            )

    def _run(self, left_rgb, right_rgb, K, baseline, scale, valid_iters, z_far,
             remove_invisible, *, live, measure_forward_ms, want_debug_jpeg,
             load_s):
        K = K.astype(np.float32).copy()
        scale = float(min(max(scale, 0.05), 1.0))

        img0 = cv2.resize(left_rgb, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(right_rgb, fx=scale, fy=scale, dsize=None)
        H, W = img0.shape[:2]
        img0_ori = img0.copy()

        forward_ms = 0.0
        forward_s = 0.0
        padder = None

        try:
            with torch.inference_mode():
                t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
                t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)

                use_events = measure_forward_ms and torch.cuda.is_available()
                if use_events:
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                wall_t0 = time.time()

                if self._model_kind == "fast_trt":
                    Heng, Weng = self._trt_input_hw
                    t0e = torch.nn.functional.interpolate(
                        t0, size=(Heng, Weng), mode="bilinear", align_corners=False)
                    t1e = torch.nn.functional.interpolate(
                        t1, size=(Heng, Weng), mode="bilinear", align_corners=False)
                    if use_events: start_evt.record()
                    disp_e = self._model.forward(t0e, t1e)
                    if use_events: end_evt.record()
                    disp_up = torch.nn.functional.interpolate(
                        disp_e.float(), size=(H, W), mode="bilinear", align_corners=False)
                    disp = (disp_up * (float(W) / float(Weng))).clamp_min(0).data.cpu().numpy().reshape(H, W)
                else:
                    from core.utils.utils import InputPadder  # noqa: WPS433
                    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
                    t0, t1 = padder.pad(t0, t1)

                    iters = (valid_iters if valid_iters else
                             _DEFAULT_ITERS.get(self._model_kind, 32))
                    if self._model_kind == "fast_fp32":
                        with torch.amp.autocast("cuda", enabled=False):
                            if use_events: start_evt.record()
                            disp = self._model.forward(t0, t1, iters=iters, test_mode=True,
                                                       optimize_build_volume="pytorch1")
                            if use_events: end_evt.record()
                    elif self._model_kind == "fast_fp16":
                        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                            if use_events: start_evt.record()
                            disp = self._model.forward(t0, t1, iters=iters, test_mode=True,
                                                       optimize_build_volume="pytorch1")
                            if use_events: end_evt.record()
                    else:  # vitl / vits
                        with torch.cuda.amp.autocast(True):
                            if use_events: start_evt.record()
                            disp = self._model.forward(t0, t1, iters=iters, test_mode=True)
                            if use_events: end_evt.record()
                    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W)

                if use_events:
                    torch.cuda.synchronize()
                    forward_ms = float(start_evt.elapsed_time(end_evt))
                forward_s = time.time() - wall_t0
        finally:
            try:
                del t0, t1, padder
            except NameError:
                pass
            try:
                del t0e, t1e, disp_e, disp_up
            except NameError:
                pass

        post_t0 = time.time()

        # Disparity vis only when explicitly asked.
        vis_jpg = b""
        if want_debug_jpeg:
            from Utils import vis_disparity  # noqa: WPS433
            vis = vis_disparity(disp)
            vis_stacked = np.concatenate([img0_ori, vis], axis=1)
            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(vis_stacked, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )
            vis_jpg = buf.tobytes() if ok else b""

        if live:
            return InferResult(
                disp=disp,
                depth=np.empty(0, dtype=np.float32),
                vis_jpg=vis_jpg,
                scale_used=scale,
                forward_ms=forward_ms,
                forward_s=forward_s,
                post_s=time.time() - post_t0,
                load_s=load_s,
            )

        # Depth at the resized scale; intrinsics scaled accordingly.
        K_scaled = K.copy()
        K_scaled[:2] *= scale

        disp_for_depth = disp.copy()
        if remove_invisible:
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            us_right = xx - disp_for_depth
            disp_for_depth[us_right < 0] = np.inf

        depth = K_scaled[0, 0] * baseline / np.where(
            disp_for_depth > 0, disp_for_depth, np.inf)
        depth = np.where((depth > 0) & (depth <= z_far), depth, 0.0).astype(np.float32)
        post_s = time.time() - post_t0

        return InferResult(
            disp=disp,
            depth=depth,
            vis_jpg=vis_jpg,
            scale_used=scale,
            forward_ms=forward_ms,
            forward_s=forward_s,
            post_s=post_s,
            load_s=load_s,
        )
