# vision_bringup — design & node-selection rationale

Canonical design doc for the tk26 vision launch files. **All future
launch-related documentation lives in this `docs/` directory.**

- Date: 2026-06-23
- Package: `vision_bringup` (`src/tk26_vision/src/vision_bringup/`)
- Launch files: `vision_driver.launch.py`, `vision_bringup.launch.py`

## 1. Goal

Replace per-terminal `ros2 run` of ~20 vision nodes with two composed launches:

- **`vision_driver.launch.py`** — the sensor/hardware layer: pan-tilt head,
  Orbbec Femto Bolt, FoundationStereo (FFS) streaming depth.
- **`vision_bringup.launch.py`** — the BT-facing perception layer: only the
  vision nodes the behavior_tree actually calls, selected per RoboCup task.

The split keeps sensor-driver chatter (camera enumeration, FFS TRT warmup /
forward timings) in one terminal and perception nodes in another.

## 2. Node selection — methodology

The perception set was derived by auditing the **production** behavior_tree task
trees (entry points in `tk25_decision` `setup.py`, not test/demo files) and
mapping every invoked vision service/action/topic to its serving node, then
taking the transitive dependency closure. The mapping was cross-checked by
independent adversarial review (each include/exclude decision re-derived from
source); all decisions held. Scope: the tasks the user runs — **HRI (with Follow
integrated), GPSR, Restaurant, PickAndPlace.**

Key dependency edges discovered:

- `kimi_api/feature_recognition` and `feature_matching` are in-process service
  clients of `/object_detection_generalist` → pulling either requires
  `generalist_node`.
- `object_detection_new/yolo_seg_node` and `generalist_node` are *optional*
  clients of FFS `/foundation_stereo/get_depth` (`prefer_ffs` default true on
  the RealSense path); they fall back to native aligned depth if FFS is absent —
  so FFS is a soft, not hard, dependency.

## 3. Layer 1 — `vision_driver.launch.py`

| Node | Source | Notes |
|---|---|---|
| pan-tilt controller + state_publisher + URDF RSP | `pan_tilt.launch.py` | `launch_robot_state_publisher:=false` when grasp_bringup owns `/robot_description` |
| Orbbec Femto Bolt | `orbbec_camera/femto_bolt.launch.py` | `depth_registration:=true enable_colored_point_cloud:=true enable_ir:=false enable_frame_sync:=false` |
| FoundationStereo | `foundation_stereo.launch.py` | `stream_enabled:=true stream_align_to_color:=false` |

**RealSense is intentionally absent.** The manipulation launch
(`arm_bringup_cumotion` / grasp_bringup) owns the `xarm_camera` RealSense and is
the only launch that enables its IR pair (`enable_infra1:=true enable_infra2:=true`,
`_robot_moveit_common2_cumotion.launch.py:265-266`). Launching RealSense here
too would double-bind the same serial / node name. Consequence: FFS consumes
that manipulation-owned IR pair, so FFS streamed depth is non-empty **only when
the manipulation stack is up** (see §6).

### FastDDS SHM — whole launch, FFS included

`fastdds_shm.xml` makes same-host image topics negotiate shared memory (~30 Hz
vs ~3 Hz over UDP) and enlarges the FastDDS SHM segment to 20 MB. The whole
launch — pan-tilt, Orbbec, **and FFS** — runs under
`FASTRTPS_DEFAULT_PROFILES_FILE = fastdds_shm.xml`:

- the Orbbec *publisher* must offer SHM for the perception subscribers to use it;
- **FFS needs it too.** The RealSense IR pair FFS subscribes to (~0.82 MB
  combined) exceeds the *default* ~512 KB SHM segment, so a frame drops and FFS
  time-sync collapses (controlled-experiment root cause). The 20 MB segment
  fixes it.

> An earlier design fenced FFS *out* of SHM via a scoped `GroupAction` +
> `UnsetEnvironmentVariable`, on the belief that SHM corrupts cuMotion collision
> voxels. That belief was **experimentally refuted** — the 20 MB profile is
> data-safe (0 corruption across oversize + slow-reader tests), and fencing FFS
> onto the default ~512 KB segment is exactly what *starves* the IR pair. The
> fencing was removed; FFS runs under the blanket profile.
>
> Caveat: the segment size is set by the *writer*. The IR pair is published by
> the camera owner (manipulation `arm_bringup_cumotion.launch.py:247-268`,
> currently with no profile), so the full IR-reliability fix requires the
> *manipulation* launch to publish under the same 20 MB profile. Setting it on
> the FFS subscriber here removes the harmful fencing and is the vision-side half
> of the fix; the camera-owner half is tracked separately.

## 4. Layer 2 — `vision_bringup.launch.py`

Plain blanket SHM env (every node here is a camera subscriber that benefits; no
FFS to fence). Two always-on core nodes + per-task groups.

### Always-on core (default ON, ungated by task)

| Node | Serves | Why always-on |
|---|---|---|
| `object_detection_generalist/generalist_node` | `/object_detection_generalist` | used by PickAndPlace, GPSR, Restaurant directly + HRI transitively (operator decision: always) |
| `vision_util/door_detection` | `/door_detection_srv` | PickAndPlace enterArena; (operator decision: always) |

### Per-task node matrix (task flags default OFF)

| Node (pkg/exec) | hri¹ | gpsr | restaurant | pick_place |
|---|:--:|:--:|:--:|:--:|
| `object_detection_new/yolo_seg_node` | ✓ | ✓ | | |
| `vision_track/person_track_server` | ✓ | ✓ | | |
| `tk_vision_specialized/waving_person_server` | ✓ | ✓ | ✓ | |
| `kimi_api/feature_recognition` | ✓ | ✓ | | |
| `kimi_api/feature_matching` | ✓ | | | |
| `kimi_api/seat_recommend_bbox` | ✓ | | | |
| `pan_tilt/follow_head` | ✓ | | ✓ | |
| `vision_util/get_image` | | ✓ | | |

¹ `enable_hri` = HRI **+ Follow** (Follow is integrated into HRI as one task).

Shared nodes are OR-gated so they spawn once regardless of how many task flags
are set: `yolo_seg_node` ← hri∨gpsr; `person_track_server` ← hri∨gpsr;
`waving_person_server` ← hri∨gpsr∨restaurant; `feature_recognition` ← hri∨gpsr;
`follow_head` ← hri∨restaurant. OR via `IfCondition(PythonExpression(...))`.

`enable_pick_place` gates **no** task-specific node — PickAndPlace's only vision
deps (generalist + door) are already in the always-on core. The flag is kept for
symmetry and future-proofing.

### Per-task vision usage (evidence summary)

- **HRI+Follow** — `feature_extraction_service` + `feature_matching_service`
  (guest intake / two-way introduction, always), `seat_recommend_bbox_service`
  (escort-and-seat, always), `follow_head_action` (eye contact, always);
  Follow side: `track_person` + `/person_track_node/reseed_target`,
  `detect_waving_persons` (recovery scan), `object_detection_yolo`
  (help-me-carry pointed-luggage).
- **GPSR** — `object_detection_generalist` (find_object/person/count),
  `object_detection_yolo` (grasp branch), `track_person` (follow plan),
  `detect_waving_persons` (go-to-waving), `feature_extraction_service`
  (describe_person), `get_image_service` (vlm_fallback).
- **Restaurant** — `detect_waving_persons` (waving customers, both standard and
  simplified trees), `object_detection_generalist` (tray + fallback customer
  scan, always-on core), `follow_head_action` (eye contact, standard tree only).
- **PickAndPlace** — `object_detection_generalist` (ScanForGeneralist),
  `door_detection_srv` (enterArena). Both in the always-on core.

### Dropped nodes (no reachable caller in the four tasks)

`yolo_seg_default_node` (legacy COCO, dead), `spot_on_shelf_server`,
`object_match_server`, `object_match_all_server`, `placing_location_server`,
`grocery_categorize` (StoringGroceries-only), `get_point_cloud` (ServeBreakfast
only), `get_orbbec_pc`, `monocular_depth_pc`. These remain runnable via
`ros2 run` for their owning tasks; they are simply not in these launches.

## 5. Argument reference

`vision_driver.launch.py`: `enable_pan_tilt` (true), `enable_orbbec` (true),
`enable_ffs` (true), `device` (`/dev/ttyUSB0`), `launch_robot_state_publisher`
(true), `camera_profile` (d435), `ffs_stream_enabled` (true),
`ffs_stream_align_to_color` (false).

`vision_bringup.launch.py`: `enable_generalist` (true), `enable_door` (true),
`enable_hri` (false), `enable_gpsr` (false), `enable_restaurant` (false),
`enable_pick_place` (false).

## 6. Cross-launch contract with manipulation

1. Manipulation/grasp_bringup brings up the `xarm_camera` RealSense **with the
   IR pair enabled** and owns `/robot_description`.
2. `vision_driver.launch.py` runs with `launch_robot_state_publisher:=false`
   (so only the xArm RSP owns the pan/tilt chain) and brings up FFS, which
   consumes the manipulation RealSense IR pair. FFS depth is empty until step 1
   is up.
3. `vision_bringup.launch.py` runs the perception nodes; detection prefers FFS
   `get_depth` but degrades gracefully to Orbbec/native aligned depth (throttled
   warning, `status=1` empty scene, no crash) if FFS/RealSense are absent.

## 7. Caveats

- **kimi_api nodes** (`feature_recognition`, `feature_matching`,
  `seat_recommend_bbox`) raise at init without `OPENROUTER_API_KEY` /
  `DASHSCOPE_API_KEY`. Launch from the workspace root so `.env` is found, or
  leave `enable_hri`/`enable_gpsr` off.
- **FFS standalone** (no manipulation RealSense) spins with no input — harmless
  but loads the TRT engine. Use `enable_ffs:=false` for vision-only bench runs.
- **Re-source after building** — `vision_bringup` is a package; a new build
  requires `source install/setup.zsh` (or a new shell) before `ros2 launch`
  resolves it.

## 8. Build & validate

```bash
# build into the live workspace install tree (NOT src/tk26_vision/install)
WS_ROOT=/home/tinker/tk25_ws ./src/tk26_vision/scripts/build.sh \
    --packages-select vision_bringup
# validate (under bash; zsh chokes on the .bash setup files)
ros2 launch vision_bringup vision_driver.launch.py  --show-args
ros2 launch vision_bringup vision_bringup.launch.py --show-args
```
