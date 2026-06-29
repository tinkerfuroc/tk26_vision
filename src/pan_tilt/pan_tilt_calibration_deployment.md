# Pan-tilt extrinsic calibration → deployment (the 2026-06-27 camera_mount fix)

**Audience:** anyone deploying a fresh pan-tilt hand-eye calibration to the robot,
or debugging "vision points land in the wrong place" (behind the robot, below the
floor, or a constant angular offset in seat-recommend / arm-pointing).

This documents a real bug we hit on 2026-06-27 and the rule that prevents it.
The decision-side consequence (HRI arm-pointing) is documented in
`tk25_decision/src/behavior_tree/behavior_tree/HRI/pan_tilt_pointing.md`.

---

## Symptom

HRI seat-recommendation had the xArm point ~12° clockwise of the seat shown
(correctly) in the vision log, and the angle got *worse* the further off-axis the
seat was. The pan-tilt had just been freshly calibrated and "should" have been
producing correct TFs.

The deeper symptom (visible if you `ros2 run tf2_ros tf2_echo base_link
camera_color_optical_frame`, or project a known-forward point): the camera was
modelled facing roughly **backward and up** — seat/person centroids transformed
into `base_link` landed **behind** the robot (x < 0) and high (z ≈ 3 m).

## Root cause — a degenerate calibration pair, half-applied

The pan-tilt extrinsic calibration solves a chain with a **degenerate pair**:

| parameter | physical meaning | lives in (runtime) |
|---|---|---|
| `tilt_offset_rad` (a.k.a. `theta_t_offset`) | tilt-servo zero offset | `src/pan_tilt/config/pan_tilt.yaml` |
| `T_B` = `camera_mount` rotation | camera body pose on the tilt link | the **URDF** (`tinker_urdf/src/pan_tilt.urdf.xacro`) |

These two trade off against each other: a calibration solution is only correct
when **both** halves are deployed together. The May calibration and the June-25
re-calibration (`calibration_data/wjy-0625-new/polish.json`) each produced a
matched `(tilt_offset, camera_mount)` pair.

**The bug:** the June-25 `polish.json` was **half-applied**. Its `tilt_offset_rad`
(`1.3306…`) reached `pan_tilt.yaml`, but its `camera_mount` `t_b` **never reached
the URDF** — the URDF kept the May `camera_mount_rpy = [0.041, -0.795, 3.083]`
("backward convention", yaw ≈ 176°). A **mismatched** `tilt_offset` + `camera_mount`
models the (physically forward) camera as looking ~180° backward. Vision points
then land behind the robot, and any downstream consumer is ~180° off.

In HRI this was *masked* by a `BtNode_PointTo(pan_bias=math.pi)` hack that
cancelled the ~180° in the arm's joint0, leaving only a tilt/position-dependent
residual (the "~12°, worse off-axis" symptom). See the decision-side doc.

## The fix (2026-06-27)

1. **Apply `polish.json`'s `camera_mount` to the URDF** (the missing half).
   `apply_to_urdf --results polish.json` against `tinker_urdf/src/pan_tilt.urdf.xacro`:
   ```
   camera_mount origin rpy:  0.0406528 -0.79457 3.0833   →   3.10886 1.01221 -0.0396491
   ```
   and synced `pan_tilt_standalone.urdf.xacro`'s arg defaults to match (its stale
   `attach_rpy`/`camera_mount` defaults otherwise re-introduce a ~4° residual via
   the standalone dev launch).
2. **Sync `pan_tilt.yaml`** to the polish values (already had the tilt offset; the
   source was brought in line): `tilt_offset_rad: -1.8735… → 1.3306144109`,
   `pan_offset_rad: 3.1443140208`.
3. **Raised `tilt_max_deg` 30 → 90** so the horizontal-forward firmware tilt is
   within the controller's allowed range.
4. **Decision side:** `pan_bias=math.pi → 0.0` at every seat/person-pointing call
   site (the hack is no longer needed and would now point the arm 180° off).

Verified live from the install build: at firmware tilt 45° the camera bearing is
**+1.38° forward** (was −174°); HRI seat-pointing residual dropped from 12°+ to
~2°.

## Deployment rules (read before deploying any new calibration)

- **Ship the `(tilt_offset_rad, camera_mount)` pair together.** Updating
  `pan_tilt.yaml` without the matching URDF `camera_mount` (or vice-versa) gives a
  ~180°-wrong camera. This is the single most common way to break it.
- **`apply_to_urdf` enforces a forward-camera invariant** (`|yaw| < π/2`): it
  **rejects** `chain.json`'s `t_b` (xyz-euler yaw −178°) and **accepts**
  `polish.json`'s `t_b` (yaw −2.27°). **Deploy `polish.json`, not `chain.json`.**
- **The runtime URDF path is the macro defaults.**
  `tk25_basic/src/tinker_urdf/.../tracer_mini_manipulator.urdf.xacro` instantiates
  `pan_tilt_macro` with **no args**, so grasp/HRI use the macro defaults in
  `pan_tilt.urdf.xacro` — the per-robot `urdf_overrides.yaml` path is only the
  standalone dev launch, and tinker1 has no overrides file. Patch the macro
  defaults, not just the standalone args.
- **Firmware tilt = 45° puts the optical axis horizontal-forward.** (An older note
  saying 30° is stale.)

## Affected files (this fix)

- `tk26_vision/src/pan_tilt/config/pan_tilt.yaml` — `pan_offset_rad`,
  `tilt_offset_rad`, `tilt_max_deg`.
- `tk25_basic/src/tinker_urdf/src/pan_tilt.urdf.xacro` &
  `pan_tilt_standalone.urdf.xacro` — `camera_mount` / `attach` origins.
- `tk25_decision/.../HRI/hri.py`, `HRI/point_at_seat.py`,
  `TemplateNodes/pointing_math.py`, `TemplateNodes/Manipulation.py` — `pan_bias=0`.
