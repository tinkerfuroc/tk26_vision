# Changelog — pan_tilt

All notable changes to this package land here, newest first.

## [Unreleased]

### Changed
- `pan_tilt_state_publisher` now reads its calibration-derived pan/tilt
  joint offsets from the per-robot `tinker_robot_config` profile
  (`robots/<ROBOT_NAME>/pan_tilt/offsets.yaml`, keys
  `pan_tilt.offsets.{pan,tilt}_offset_rad`) instead of only the
  `config/pan_tilt.yaml` ROS params. Adds `_load_profile()` /
  `_load_per_robot_offsets(logger)` module-level helpers (isolated for test
  monkeypatching); any resolver failure (package absent, `ROBOT_NAME`
  unset, profile missing the keys) degrades to a logged WARN and falls back
  to the existing `pan_offset_rad`/`tilt_offset_rad` params, which
  `config/pan_tilt.yaml` now documents as a dev-machine fallback only (NOT
  per-robot). `pan_tilt.launch.py`'s deprecated
  `overrides_key='pan_tilt.urdf_overrides'` mapping (unrelated URDF-mount
  geometry sub-tree, not the joint offsets) is dropped in the same change —
  see README "Runtime Configuration" for the split between the two profile
  sub-trees. (Task 2 / Phase 1b)
- `pan_tilt.launch.py` publishes the URDF via the new
  `tinker_robot_config/robot_description.launch.py` wrapper, with
  `overrides_key='pan_tilt.urdf_overrides'` and the existing private-topic
  remappings (`/pan_tilt/robot_description`, `/pan_tilt/joint_states`)
  threaded through the wrapper's `remappings` arg. The xacro macro's
  hardcoded attach/camera_mount defaults remain as a last-ditch fallback
  for tools that don't go through the wrapper; runtime URDF now reflects
  `robots/<ROBOT_NAME>/pan_tilt/urdf_overrides.yaml`. Operational disable
  (`launch_robot_state_publisher:=false`) and `IfCondition` gating are
  preserved. Pairs with tk25_basic P6.1 + follow-up commits. (P6.2)
- `calib_web` and `calibrate_collect` now resolve `calibration.yaml` via
  `tinker_robot_config` — default to the per-robot file under
  `robots/<ROBOT_NAME>/pan_tilt/calibration.yaml`. Operators may still
  pass `-p config:=…` explicitly. The in-tree `config/calibration.yaml`
  is deleted because no consumer reads it.
- Updated `_resolve_source_tree_yaml` (calib_web) to map the new install
  path under `tinker_robot_config/share/.../robots/<robot>/pan_tilt/` to
  the canonical source under
  `tk25_basic/src/tinker_robot_config/robots/<robot>/pan_tilt/`. The
  legacy `pan_tilt/share/pan_tilt/config/` pattern is still recognised
  for back-compat with explicitly-passed paths. (P5a)
- Added `<exec_depend>tinker_robot_config</exec_depend>` to `package.xml`.
  (P5a)
- README gained a "Calibration" section describing the per-robot yaml
  location and the resolver-backed default. (P5a)

### Removed
- `src/pan_tilt/config/calibration.yaml` — moved to
  `tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/pan_tilt/calibration.yaml`
  during P2; removed here in P5a because nothing reads it anymore. The
  symlink shim originally planned for P5a was dropped after discovering
  that calib_web's rename-then-rewrite write paths are symlink-hostile;
  pointing calib_web directly at the per-robot location is cleaner.
  Consequently, P5b ("drop the symlink shim") is also dropped — there is
  no shim to remove. (P5a)
