# Triage — FS color-alignment bleed — 2026-05-25

User-visible symptom: in FS aligned depth, depth from the table behind a
bottle bleeds into the **right** side of the bottle (and analogously the
left side picks up bottle depth where it should be background). Looks
like a small horizontal offset between FS depth and color image.

## Triage chain

| Phase | Question | Result |
|---|---|---|
| **0** | Empirical pixel shift via 41×41 binary-edge xcorr | (0,0) — said no global shift. **Wrong; the surface was dominated by aligned horizontal table/floor edges, masking bottle-edge misalignment.** |
| **G1** | MAE-vs-dx sweep over ±15 px (whole image, integer + parabolic sub-pixel) | color: **+3 px / sub +2.97 px**, MAE 47→47 mm.  IR1: **0 px / sub −0.3 px**, MAE flat. *FS in IR1 frame agrees with native IR1 depth sub-pixel.* |
| **H1** | Feed **ground-truth native IR1 depth** (no FS in loop) through `color_align.reproject_ir_to_color`, compare to ASIC `aligned_depth_to_color`. | **+2 / +2.39 px shift** — proves the misalignment is in our reprojection math, not in FS. |
| **H2** | Same as H1 but binned by Z. | **D435**: sign change with Z — +2..+3 px at near Z, −1..−2 px at far Z. Classic Tx-error signature.  **D405**: **constant −3 px across all Z bins**. No Z dependence (because D405 publishes Tx ≈ 0). |
| **H3** | Linear-regress `dx ≈ a + b/Z` on H2 results. | **D435**: a = −2.5 px (constant) + b = +2.33 px·m (≡ Tx underestimate of +3.83 mm out of 14.66 mm published).  **D405**: a = −3.33 px + b ≈ 0 (no Tx component). |

## What this means

1. **The bug is in `src/foundation_stereo/foundation_stereo/color_align.py`** — specifically, the forward-projection it implements does not match the librealsense `rs2::align` math that produces the ASIC's `aligned_depth_to_color`.
2. **The discrepancy has two components**, separable by the D405 test:
   * **Constant ~2.5–3 px** offset (visible on both cameras, including D405 where Tx ≈ 0). This is purely an algorithmic difference — likely rounding convention, sub-pixel splatting, or distortion handling.
   * **Z-dependent** component on D435 only (because Tx ≠ 0 there). The empirical fit suggests the ASIC behaves as if Tx were +3.83 mm larger than the topic-published 14.66 mm.
3. **FS itself is innocent.** Phase G1's IR1-frame sweep shows FS depth agrees with native IR1 depth to sub-pixel. The misalignment is introduced entirely by the IR1→color reprojection.

## Why the user sees "table bleeds into right side of bottles"

At a bottle Z of ~0.5 m on D435: the published-extrinsics-based projection
shifts IR1 depth right by ~17.8 px; the ASIC effectively shifts by ~22 px
(empirical fit). So **our FS-color depth is positioned ~4-5 px LEFT** of
where the bottle is in the color image at that range. The bottle's
*right* edge in our FS depth ends up 4-5 px to the left of the bottle's
right edge in color. To the right of our FS-bottle silhouette there's a
narrow band where: native ASIC depth says "still bottle", but our FS
depth says "table" (because IR1 depth values at columns one-bottle-width-
to-the-right of where we placed them happen to be table). Same artifact,
mirrored, on the left side — though it's masked there by the bottle's
own foreground depth winning the np.minimum.at.

## Recommended fixes (ordered)

1. **Replace `reproject_ir_to_color` with `pyrealsense2.rs2.align`.** Wrap
   FS depth in a synthetic `rs2::frame` against the same intrinsics +
   extrinsics that librealsense already has registered for the active
   device, and let librealsense do the warp. Cost: depends on whether
   we can construct an rs2::frame outside the SDK's normal pipeline —
   librealsense supports software_device for exactly this.
2. **Implement BACKWARD warping in `color_align.py`.** For each color
   pixel, ray-cast back into IR1 frame and sample IR1 depth there with
   bilinear interpolation. Eliminates the np.minimum.at occlusion mess.
   Algorithmically cleaner; matches what rs2::align does internally.
   Sub-pixel by construction.
3. **Empirical per-camera calibration.** Add a `color_align_offset_px`
   parameter (or two-parameter `a_px + b_pxm/Z` correction) measured
   once per device and applied as a post-shift on the projected
   `(u_c, v_c)` floats before rounding. Cheap; works without changing
   the algorithm.

## Files in this dir

| File | What |
|---|---|
| `triage_alignment.py` | full Phase A-F triage (matrices, IR1-frame compare, edge xcorr, bottle profile). Runs against D435. |
| `triage_phase_g.py` | MAE-vs-dx sweep (whole image) + vertical-edge xcorr + per-ROI xcorr + cross-modal color-vs-depth xcorr. |
| `triage_phase_h.py` | feed native IR1 → `reproject_ir_to_color` → compare to ASIC native_color, with Z-binned dx sweep. |
| `triage_phase_h_d405.py` | same as triage_phase_h.py but targets `/camera/head_camera/` (D405). |
| `triage/h2_z_binned.png` | D435 Z-binned best-dx bar chart (sign-changes around Z=1m). |
| `triage_h/h2_z_binned.png` | D435 — same data, second run. |
| `triage_h/h_sweep.png` | D435 — H1 global sweep, MAE vs dx. |
| `triage_h/h_panel.png` | D435 — ASIC vs ours vs diff side-by-side. |
| `triage_h_head_camera/h2_z_binned.png` | **D405 — constant-across-Z signature.** |
| `triage_h/h_data.json` / `triage_h_head_camera/h_data.json` | raw numbers + fit coefficients. |
