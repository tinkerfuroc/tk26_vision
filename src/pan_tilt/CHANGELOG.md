# Changelog — pan_tilt

All notable changes to this package land here, newest first.

## [Unreleased]

### Changed
- Task-3 review follow-up: `calibration/readme.md` §8 rewritten to the
  per-robot apply flow (the shared-xacro `cp` + `--xacro`/`--no-yaml`
  instructions it still documented are gone); orphaned
  `calibration/yaml_targets.py` deleted (no importers left);
  `_require_robot_name` now rejects a `ROBOT_NAME` containing `/` or `..`
  (path-traversal hardening). (Task 3 / Phase 1c)
- **Calibration Preview/Apply is per-robot only** (Task 3 / Phase 1c).
  `apply_to_urdf` was rewritten around a single public entry
  `apply_calibration(params, basic_root=None, allow_flipped_camera=False)`
  that renders and atomically writes the COMPLETE contents of exactly two
  files in the tk25_basic SOURCE tree —
  `robots/<ROBOT_NAME>/pan_tilt/pan_tilt_overrides.xacro` + `offsets.yaml`
  — in lockstep with `.old-<ts>` backups (both-or-neither; rollback on
  partial failure). It refuses when `ROBOT_NAME` is unset
  (`CalibrationApplyError`, new alias `ApplyError`; calib_web surfaces it
  as HTTP 400 with `detail`). rpy values the solve didn't fit
  (trivial/absent rotvec) are copied from the current per-robot overrides
  file, never zeroed; the forward-camera invariant (`|yaw| < π/2` unless
  `allow_flipped_camera`) still guards fitted camera rotations. New
  helpers: `render_overrides_xacro`, `render_offsets_yaml`,
  `render_calibration`, `resolve_per_robot_dir` (workspace located by
  walking up from the module — never a filename glob, which is ambiguous
  across the three per-robot copies). CLI is now
  `python -m pan_tilt.calibration.apply_to_urdf --results <json>
  [--basic-root PATH] [--allow-flipped-camera]`; the old
  `--xacro/--yaml/--no-yaml/--allow-partial/--overrides-yaml/
  --no-overrides/--out` flags are gone. `urdf_targets.list_targets()` now
  returns the single per-robot target ("robots/<robot>/pan_tilt/
  (per-robot apply target)", build command `tkbuild tk25_basic
  --packages-select tinker_robot_config`); calib_web's
  `urdf_diff`/`urdf_apply` ignore a stray legacy `xacro_path` in POST
  bodies and keep the response keys the web UI reads (`applied`,
  `backup_path`, `yaml_path`/`yaml_applied`/`yaml_backup_path`,
  `pan_offset_rad`/`tilt_offset_rad`, `build_command`, `workspace_hint`,
  plus new `written`/`robot`). (Task 3 / Phase 1c)

### Removed
- The shared-source patching paths in `apply_to_urdf`: `_patched_macro` /
  `_patched_standalone` / `_patched_xacro` (with `JOINT_BLOCK_RE`,
  `ATTACH_XYZ_DEFAULT_RE`, `ATTACH_RPY_DEFAULT_RE`, `MACRO_DECL_RE`),
  `_patch_yaml_offsets`, `_patch_urdf_overrides`, `_resolve_overrides_yaml`,
  `_resolve_yaml_path`, and `resolve_source_path`. Rationale: the old
  regex patchers targeted the pre-`db1524a` macro format — running them
  against the new per-robot-include macro would silently destroy per-robot
  behavior, and `resolve_source_path`'s `**/<name>` glob became ambiguous
  (3 identically-named per-robot files) with a silent install-tree-write
  fallback. A tombstone test (`test_apply_source_tree.py::
  test_old_shared_source_patchers_are_gone`) keeps them deleted.
  (Task 3 / Phase 1c)

### Changed
- Docs correction (launch comment + README, review fix for the entry
  below): the dropped `overrides_key='pan_tilt.urdf_overrides'` mapping
  was URDF mount-geometry threading only (never the joint offsets), and
  its removal does not revert the standalone launch to hardcoded macro
  defaults — since tk25_basic `db1524a`, `pan_tilt.urdf.xacro` pulls
  per-robot mount geometry via a `ROBOT_NAME`-guarded `<xacro:include>`
  at xacro-parse time, independent of launch args. (Task 2 / Phase 1b)
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
