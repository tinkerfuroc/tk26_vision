# Changelog — pan_tilt

All notable changes to this package land here, newest first.

## [Unreleased]

### Changed
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
