# Orbbec HRI Fixed Higher Resolution — Design Spec

- **Date:** 2026-07-03
- **Status:** Approved design, pending implementation plan
- **Robot:** tinker1 / tinker2 (Orbbec Femto Bolt, head-mounted)
- **Author:** Claude (brainstormed with cindy)

## 1. Problem & Goal

The HRI (receptionist-style) behavior tree task uses the Orbbec Femto Bolt for
face/feature enrollment (host + guest scan, two-way introduction) and for
person following. The Orbbec's color stream is currently launched at the
vendored default, 1280×720 MJPG, which limits feature-match quality during
enrollment.

Goal: launch the Orbbec at a higher, **fixed** color resolution (1920×1080)
for the whole HRI session — no runtime switching — while ensuring nothing
else in the vision/behavior-tree stack silently breaks at the new
resolution.

### Why not dynamic switching

The original idea was to raise resolution only during enrollment and revert
before the follow phase. Investigation ruled this out as unnecessary
complexity:

- The Orbbec vendored driver (`orbbec_camera`, OrbbecSDK_ROS2 v2.7.6) has
  **no runtime-reconfigurable resolution**. `color_width`/`color_height` are
  wired into the SDK's dynamic-parameter framework with an intentionally
  empty update callback (`dynamic_params.cpp:22-40`, `ob_camera_node.cpp:129-141`)
  — a `ros2 param set` appears to succeed but does nothing, and the driver
  logs "can not be changed in runtime." The only way to change resolution is
  killing and relaunching the driver process.
- The person-tracking/following phase's compute is already fully decoupled
  from camera resolution: YOLO runs at a fixed `imgsz=736` letterbox
  (`vision_track/yolo_tracker.py:478-484`) and ReID crops resize to a fixed
  128×256 OSNet input (`vision_track/reid_backbone.py:57,183-189`) regardless
  of source resolution. There is therefore **no performance reason** to
  revert to a lower resolution once following starts.
- Given that, a launch-time-only resolution (decided once, before the whole
  HRI session starts) achieves the same enrollment-quality benefit with far
  less risk than a mid-task driver relaunch (no relaunch blips, no TF gaps,
  no orchestration node needed).

## 2. Decisions (from brainstorm)

| Topic | Decision |
|---|---|
| Switching mechanism | **None — launch-time only.** No BT node, no orchestrator service, no mid-task relaunch. |
| Target resolution | **1920×1080 MJPG** color, confirmed available at full 30fps via direct hardware query (`list_camera_profile_mode_node` against the live Femto Bolt). Depth stays **640×576@30fps** (unchanged). |
| Bug-fix scope | Fix `object_seg_yolo.py`'s hardcoded depth-reprojection grid (blocks this feature outright) and `door_detection.py`'s same-pattern bug (dead code today, fixed for completeness). **Not** fixing `follow_head.py`'s `CameraInfo` latch — not load-bearing without dynamic switching. |
| Point-cloud handling | Leave `enable_colored_point_cloud:=true` as-is; rely on the vendored driver's existing zero-subscriber early-return once the bug fixes land. |
| SHM segment size | Increase `fastdds_shm.xml`'s `segment_size` to accommodate larger raw color frames. |
| Launch surface | Fix in `vision_bringup/vision_driver.launch.py` (add `color_width`/`color_height` args, default unchanged) + `tmux_hri_vision.sh` (HRI-specific override). **Not** `master_hri2.sh` — confirmed not the live script; `master_hri.sh` → `tmux_hri_vision.sh` → `vision_driver.launch.py` is the real path. |

## 3. Root-cause context: the two bugs

### 3.1 `object_detection_new/object_seg_yolo.py` — hardcoded depth grid

`_pointcloud_to_array` (`:399-431`) reprojects the Orbbec's ordered colored
`PointCloud2` (`/camera/depth_registered/points`) into a 2D grid using the
*live* `CameraInfo` `K` matrix (correct) but writes into a buffer hardcoded
to `np.zeros((720, 1280, 3))` (`:405`) and clips `valid_coords` to
`x<1280`/`y<720` (`:422-425`). Meanwhile the RGB image (`rgb_img`) and YOLO
masks are read at the *actual* live resolution (`:786`). The moment color
resolution ≠ 1280×720, any detection whose bbox falls outside the stale
720×1280 window either gets silently dropped or throws an unhandled
`ValueError` (verified: mismatched array broadcast, e.g. `(200,200)` vs
`(200,0)`) inside `_calculate_centroid` (`:980-1041`), which is uncaught all
the way up through `_detect_objects` and `generalist_node.py:364`'s
`_yolo_pipeline(**ctx)` call — the offending service call hangs (caller has
no timeout on the `await`).

`object_detection_generalist` (`generalist_node.py:62`, subclasses
`YOLOSegmentationNode`) inherits this method unmodified
(`generalist_node.py:320`), so both `object_detection_new`'s specialist path
and `object_detection_generalist` — the service `BtNode_FeatureExtraction`
and `BtNode_FeatureMatching` actually call — are broken by this bug at any
resolution other than 1280×720.

**Fix:** rewrite the Orbbec depth path to reproject from
`/camera/depth/image_raw` (the SW-aligned, depth-registered Image topic) +
live `CameraInfo`, instead of the ordered colored `PointCloud2`. This
mirrors the already-correct, resolution-safe pattern used in
`kimi_api/seat_recommend_bbox.py` (`:606-621`), `vision_util/get_orbbec_pc.py`
(`:107`), and `vision_track/person_track_node.py` (`:840-870`) — all of
which read `depth_msg.height`/`depth_msg.width` live and rebuild any
cached meshgrid keyed on the actual shape.

Side benefit: this decouples `object_detection_generalist` from the ordered
colored point cloud entirely. Per `orbbec_diagnosis.md` (a prior crash
investigation in this tree), `publishColoredPointCloud` runs **synchronously
in the Orbbec driver's color-frame callback thread** — every color frame
triggers a full CPU xy-table reprojection of the whole cloud (~900k points
documented at 720p). When that reprojection got slow, it previously dragged
the color stream itself down to ~5Hz, not just the depth/cloud topics. Moving
`object_detection_generalist` off this path removes that coupling for the
one HRI-hot-path consumer that had it.

### 3.2 `vision_util/door_detection.py` — same bug pattern, independent copy

Its own standalone copy of the PointCloud2-to-grid logic hardcodes
`h, w = 720, 1280` (`:72`) and a center-crop window `W, H, L = 1280, 720, 10`
(`:108`). Not shared code with `object_seg_yolo.py` — needs its own fix,
same approach (dynamic dims from the depth Image + `CameraInfo`).
`BtNode_DoorDetection`'s call site is currently commented out in the live
`hri.py` (`:358-382`), so this is dead code on the production HRI path
today, but `door_detection` is part of HRI's always-on core launch group
(`enable_door` default true) and would misbehave the instant it's ever
re-enabled at non-720p. Fixed for completeness.

## 4. Launch-time resolution wiring

Two divergent launch paths exist for the Orbbec driver in this tree:

- `vision_bringup/launch/vision_driver.launch.py:115-126` — the "designed"
  2-layer launch package (driver layer: pan-tilt + Orbbec + FFS). Includes
  `orbbec_camera femto_bolt.launch.py` with `depth_registration`,
  `enable_colored_point_cloud`, `enable_ir`, `enable_frame_sync` overridden,
  but **no resolution override** — falls through to the vendored default.
- `src/tk25_basic/src/scripts/master_hri2.sh:57` — an alternate/stale script
  that bypasses `vision_driver.launch.py` entirely and launches
  `orbbec_camera femto_bolt.launch.py` directly in a tmux pane. **Confirmed
  not the live path** — not touched by this work.

The actual live path, confirmed by the user:

```
master_hri.sh → tmux_hri_vision.sh → ros2 launch vision_bringup vision_driver.launch.py ...
```

`tmux_hri_vision.sh` pane 0 (sensor/driver layer) has two branches (with and
without a detected pan-tilt device), both invoking `vision_driver.launch.py`.

**Changes:**

1. `vision_driver.launch.py` — add to the `args` list:
   ```python
   DeclareLaunchArgument('color_width', default_value='1280'),
   DeclareLaunchArgument('color_height', default_value='720'),
   ```
   and thread into the Orbbec `IncludeLaunchDescription`'s
   `launch_arguments` dict:
   ```python
   'color_width': LaunchConfiguration('color_width'),
   'color_height': LaunchConfiguration('color_height'),
   ```
   Defaults preserve today's behavior for every other task
   (GPSR/restaurant/bench runs) that launches this file without an override.
   The vendored `femto_bolt.launch.py` already declares `color_width`/
   `color_height`/`color_fps`/`color_format` (`:27-30`) — this just exposes
   the first two one layer up. `color_format` stays `MJPG` (default);
   confirmed available at 1920×1080@30fps via direct hardware query.

2. `tmux_hri_vision.sh` — both pane-0 branches add
   `color_width:=1920 color_height:=1080` to their existing
   `ros2 launch vision_bringup vision_driver.launch.py ...` command lines.

Depth stream (`depth_width`/`depth_height`) is left untouched at 640×576 —
confirmed both in vendor source (`ob_camera_node.cpp:2498-2504`,
`align_filter_->setAlignToStreamProfile` targets whatever color profile is
actually configured, not a hardcoded size) and by direct hardware query that
SW alignment already handles arbitrary color/depth size combinations. Raising
color resolution introduces no new alignment-mode requirement.

## 5. Bandwidth / SHM sizing

1920×1080 raw color `Image` messages (the Orbbec driver decodes MJPG to
RGB888 before publishing, per `ob_camera_node.cpp:4322-4323`) are
theoretically `1920 × 1080 × 3 ≈ 6.2 MB` uncompressed, roughly 2.25× the
720p frame size — a multi-MB-per-frame increase either way. Treat 6.2MB as a
sizing upper bound and confirm the real transmitted size empirically during
implementation (`ros2 topic bw` / message introspection), rather than
assuming the theoretical figure exactly.

`config/fastdds_shm.xml`'s `segment_size` (currently `20971520` = 20MB,
shared by the whole driver-layer launch: pan-tilt + Orbbec + FFS via
`SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', ...)` in
`vision_driver.launch.py:99-101`) needs to comfortably exceed the largest
single message with headroom for several in-flight frames across multiple
subscribers. This is a **strictly additive, previously-validated-safe**
change — this exact file's segment size was already raised once (512KB →
20MB) to fix RealSense IR sync drops, live-verified data-safe with no
corruption. Exact target size (proposed ~48-64MB, to be confirmed empirically
against `ros2 topic hz` + drop counters) is an implementation-time decision,
not fixed here.

**Point-cloud sizing is NOT the driver, contrary to first appearance.** The
ordered colored point cloud's width/height track the **color** stream
resolution (`ob_camera_node.cpp:2857-2858`,
`width = color_frame->getWidth()`), so at 1920×1080 it would be ~2.07M
points — tens of MB per message, dwarfing even a generously-sized segment.
However: `publishColoredPointCloud` early-returns when
`depth_registration_cloud_pub_->get_subscription_count() == 0`
(`ob_camera_node.cpp`, confirmed in the vendored driver). Once §3's bug
fixes land, **no live HRI BT node subscribes to
`/camera/depth_registered/points`** — `object_detection_generalist` moves
off it (§3.1), and `door_detection`/`waving_person_server`'s BT hooks are
both dormant in the live tree (confirmed: no `BtNode_ScanForWavingPerson` or
enabled `BtNode_DoorDetection` call sites in `hri.py`/`hri_2026.py`). So in
practice, the expensive point cloud should never be computed or published
during a real HRI run, and the SHM sizing target is really just the raw
color/depth Image topics + CameraInfo, not the point cloud. This assumption
must be verified live (see §6) rather than trusted blind.

## 6. Testing

No automated test can exercise Orbbec hardware; validated operator-in-the-loop
per this repo's existing `scripts/tests/` tier convention (T2 live-camera /
T3 interaction tiers). Implementation plan should include:

- `ros2 topic hz /camera/color/image_raw` at 1080p — confirm ~30Hz sustained,
  no drops, after the SHM segment bump.
- `ros2 topic hz /camera/depth_registered/points` during a live HRI run —
  confirm it's silent or near-zero-rate, validating the zero-subscriber
  theory in §5. If it's *not* silent, something still subscribes to it and
  the SHM sizing assumption needs revisiting.
- A full live HRI session through enrollment (host scan, guest intake,
  two-way introduction) and the follow phase, confirming feature
  extraction/matching and seat recommendation still succeed at 1080p and
  person tracking/following is unaffected.
- Check the Orbbec node's log for SHM segment-overflow / negotiation-failure
  warnings during the above.
- Re-run door_detection's service directly (`/door_detection_srv`) against
  live 1080p data to confirm the fix, even though it's not BT-invoked today.

## 7. Explicitly out of scope

- Any dynamic/runtime resolution-switching mechanism (relaunch
  orchestration, vendored-SDK live-reconfigure patch) — dropped in favor of
  launch-time-only configuration.
- `pan_tilt/follow_head.py`'s one-shot `CameraInfo` latch bug
  (`follow_head.py:427-432`) — real bug, but not load-bearing without
  dynamic switching (the single launch-time `CameraInfo` message is already
  correct and never changes for the life of the process). Left as a known,
  separately-tracked issue.
- Flipping `enable_colored_point_cloud` off — not needed given the
  zero-subscriber early-return (§5).
- Any change to `master_hri2.sh` (confirmed not the live script).
- Any change to Orbbec depth-stream resolution (stays 640×576@30fps).
