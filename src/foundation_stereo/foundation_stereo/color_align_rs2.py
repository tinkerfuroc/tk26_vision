"""IR1 → color depth alignment via librealsense `rs.align` on a
`software_device`.

Why this exists
---------------
The forward-warp implementation in `color_align_legacy.py` produces a
sparse output (one IR1 pixel projects to exactly one color pixel, so
~89% of color pixels remain as holes). When that sparse depth is
overlaid on the color image, the per-pixel Sobel gradient between every
valid projection and its zero-hole neighbour creates "salt-and-pepper"
edges everywhere — the visual artifact users perceive as "background
depth bleeding into the right side of foreground objects" (see triage
at `debug_renders/2026-05-25-fs-vs-native-alignment/`).

This implementation pushes the IR1 depth through librealsense's
`rs.align(rs.stream.color)` processing block via a `rs.software_device`.
The internal C++ implementation does sub-pixel splatting and proper
occlusion handling, giving a dense (>95% coverage) output whose depth
edges follow the true silhouettes of objects in the color image.

Cost: ~3-5 ms per call (warm), versus ~5 ms for the legacy numpy
forward warp. The first call ("warmup") takes ~12 ms while
librealsense JIT-creates internal stream graphs.

Caveats / gotchas baked into this wrapper
-----------------------------------------
1. The `rs.syncer` does NOT emit a paired (D+C) frameset on the first
   push when only one pair is in the queue. It needs at least two pairs
   in flight before it begins to release matched framesets. We work
   around this by pushing the SAME depth+color twice with sequential
   frame numbers and timestamps spaced by 1 s; the first push primes
   the syncer and emerges as a paired frameset on the second one.
2. Librealsense distortion model on RealSense color streams is
   `inverse_brown_conrady` even when distortion coefficients are zero
   (e.g. D435 publishes zeros). For D405 the coefficients are non-zero.
   We replicate the model + coeffs exactly to match the firmware's
   alignment math; passing `distortion.none` would skip the
   undistortion step and introduce an extra ~2-3 px residual offset.
3. `add_video_stream` returns a generic `stream_profile`; the calls
   that take stream profiles (`open()`, `start()`) need the raw
   profile, while `software_video_frame.profile` needs a downcast
   `video_stream_profile`. We keep both references alive.
4. Do NOT call `software_device.create_matcher(...)` — letting
   librealsense pick the default matcher is what works. An explicit
   matcher call silently drops the depth frame.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import numpy as np
import pyrealsense2 as rs


def _intrinsics_from_K(K: np.ndarray, W: int, H: int,
                       D: Optional[Sequence[float]],
                       *, is_color: bool) -> rs.intrinsics:
    """Build rs.intrinsics. RealSense color streams use
    `inverse_brown_conrady` (even with zero coeffs); IR/depth use
    `brown_conrady`."""
    intr = rs.intrinsics()
    intr.width = int(W)
    intr.height = int(H)
    intr.fx = float(K[0, 0])
    intr.fy = float(K[1, 1])
    intr.ppx = float(K[0, 2])
    intr.ppy = float(K[1, 2])
    intr.model = (rs.distortion.inverse_brown_conrady if is_color
                  else rs.distortion.brown_conrady)
    if D is None:
        intr.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        intr.coeffs = [float(D[i]) if i < len(D) else 0.0 for i in range(5)]
    return intr


def _extrinsics_to_rs(R: np.ndarray, T: np.ndarray) -> rs.extrinsics:
    """numpy row-major R + T → rs.extrinsics. librealsense stores
    rotation in column-major; transpose before flattening."""
    ex = rs.extrinsics()
    R_np = np.asarray(R, dtype=np.float64).reshape(3, 3)
    ex.rotation = list(R_np.T.ravel())
    ex.translation = [float(T[0]), float(T[1]), float(T[2])]
    return ex


class RealsenseAligner:
    """Stateful wrapper around `rs.software_device` + `rs.align(color)`.

    Construction is heavy (~12 ms; opens sensors, registers extrinsics);
    `align()` is the hot path (~3-5 ms warm). Hold one instance per
    distinct (K_ir, K_color, R, T, ir_hw, color_hw, D) tuple.
    """

    def __init__(self,
                 K_ir: np.ndarray,
                 K_color: np.ndarray,
                 R_ir_to_color: np.ndarray,
                 T_ir_to_color: np.ndarray,
                 ir_hw: Tuple[int, int],
                 color_hw: Tuple[int, int],
                 *,
                 D_color: Optional[Sequence[float]] = None,
                 D_ir: Optional[Sequence[float]] = None,
                 depth_units_m: float = 0.001):
        H_ir, W_ir = int(ir_hw[0]), int(ir_hw[1])
        H_c, W_c = int(color_hw[0]), int(color_hw[1])
        self._H_ir, self._W_ir = H_ir, W_ir
        self._H_c, self._W_c = H_c, W_c
        self._depth_units_m = float(depth_units_m)

        self._dev = rs.software_device()
        # NB: do NOT call create_matcher(). Letting librealsense pick
        # the default matcher is what makes paired framesets work.

        d_sensor = self._dev.add_sensor("Depth")
        c_sensor = self._dev.add_sensor("Color")

        # Depth stream
        d_vs = rs.video_stream()
        d_vs.type = rs.stream.depth
        d_vs.fmt = rs.format.z16
        d_vs.uid = 1
        d_vs.index = 0
        d_vs.width = W_ir
        d_vs.height = H_ir
        d_vs.fps = 30
        d_vs.bpp = 2
        d_vs.intrinsics = _intrinsics_from_K(K_ir, W_ir, H_ir, D_ir,
                                             is_color=False)
        d_prof_raw = d_sensor.add_video_stream(d_vs)
        self._d_prof = d_prof_raw.as_video_stream_profile()

        # Color stream — pixel content is unused by align, only geometry matters
        c_vs = rs.video_stream()
        c_vs.type = rs.stream.color
        c_vs.fmt = rs.format.rgb8
        c_vs.uid = 2
        c_vs.index = 0
        c_vs.width = W_c
        c_vs.height = H_c
        c_vs.fps = 30
        c_vs.bpp = 3
        c_vs.intrinsics = _intrinsics_from_K(K_color, W_c, H_c, D_color,
                                             is_color=True)
        c_prof_raw = c_sensor.add_video_stream(c_vs)
        self._c_prof = c_prof_raw.as_video_stream_profile()

        # depth → color extrinsics, registered against the video profiles
        self._d_prof.register_extrinsics_to(
            self._c_prof, _extrinsics_to_rs(R_ir_to_color, T_ir_to_color))

        self._sync = rs.syncer()
        d_sensor.open(d_prof_raw)
        c_sensor.open(c_prof_raw)
        d_sensor.start(self._sync)
        c_sensor.start(self._sync)
        self._d_sensor = d_sensor
        self._c_sensor = c_sensor

        self._align = rs.align(rs.stream.color)

        # Re-usable stub color buffer (zeros; pixels don't affect align)
        self._color_buf = np.zeros((H_c, W_c, 3), dtype=np.uint8)
        self._frame_n = 0

    def align(self, depth_ir_m: np.ndarray,
              timestamp_ms: float = 0.0) -> np.ndarray:
        """Align a float32-m depth on the IR1 grid to the color grid.

        Args:
            depth_ir_m: (H_ir, W_ir) float32, metres. Zeros = invalid.
            timestamp_ms: ignored — we use synthetic monotonic stamps
              spaced 1 s apart because the syncer needs ≥ 2 frames in
              its window to emit a paired frameset.

        Returns:
            (H_color, W_color) float32, metres. Zeros where the warp
            didn't fill (rare with `rs.align` — coverage usually
            > 95%).
        """
        assert depth_ir_m.shape == (self._H_ir, self._W_ir), (
            f"depth shape {depth_ir_m.shape} != configured "
            f"({self._H_ir}, {self._W_ir})")

        z16 = np.clip(
            np.round(depth_ir_m / self._depth_units_m), 0, 65535
        ).astype(np.uint16)

        # Workaround: push twice with sequential frame numbers + 1s-apart
        # timestamps so the syncer flushes the first pair through.
        for _ in range(2):
            self._frame_n += 1
            ts = 1000.0 * self._frame_n  # ms

            d_frame = rs.software_video_frame()
            d_frame.pixels = z16
            d_frame.bpp = 2
            d_frame.stride = self._W_ir * 2
            d_frame.timestamp = ts
            d_frame.domain = rs.timestamp_domain.hardware_clock
            d_frame.frame_number = self._frame_n
            d_frame.profile = self._d_prof
            d_frame.depth_units = self._depth_units_m
            self._d_sensor.on_video_frame(d_frame)

            c_frame = rs.software_video_frame()
            c_frame.pixels = self._color_buf
            c_frame.bpp = 3
            c_frame.stride = self._W_c * 3
            c_frame.timestamp = ts
            c_frame.domain = rs.timestamp_domain.hardware_clock
            c_frame.frame_number = self._frame_n
            c_frame.profile = self._c_prof
            self._c_sensor.on_video_frame(c_frame)

        # Drain framesets until we see one with both streams.
        depth_in = None
        for _ in range(8):
            try:
                fs = self._sync.wait_for_frames(2000)
            except Exception:
                continue
            for i in range(fs.size()):
                f = fs[i]
                st = f.profile.stream_type()
                if st == rs.stream.depth and depth_in is None:
                    fs_with_depth = fs
                    depth_in = f
            if depth_in is not None and fs_with_depth.size() >= 2:
                break

        if depth_in is None:
            raise RuntimeError(
                "rs.align via software_device: syncer never produced a "
                "paired depth+color frameset")

        aligned = self._align.process(fs_with_depth).as_frameset()
        depth_out = aligned.get_depth_frame()
        if not depth_out:
            for i in range(aligned.size()):
                f = aligned[i]
                if f.profile.stream_type() == rs.stream.depth:
                    depth_out = f.as_depth_frame()
                    break
            if not depth_out:
                raise RuntimeError("rs.align returned no depth frame")

        arr = np.frombuffer(depth_out.get_data(), dtype=np.uint16)
        arr = arr.reshape(self._H_c, self._W_c)
        return arr.astype(np.float32) * self._depth_units_m
