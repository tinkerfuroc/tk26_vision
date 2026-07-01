# Multiple named Phase-1 hand-eye (CUSTOM) datasets

**Date:** 2026-06-28
**Status:** approved, implementing
**Scope:** `src/tk26_vision/src/pan_tilt/` — calibration web UI + collect + solver

## Problem

The pan-tilt extrinsic calibration supports exactly **one** "Phase 1 — hand-eye
(CUSTOM)" dataset: a single `phase1_waypoints_custom` waypoint list plus a single
operator-chosen park pose (`phase1_custom_park_pan_deg` / `phase1_custom_park_tilt_deg`).
Operators want **many** custom hand-eye datasets — each with its own park pose and
its own xArm waypoints — so they can collect/solve several independent hand-eye sets
(e.g. `high_shelf`, `seat`, `low_table`) and cross-check / merge them. The request:
add an "+ add custom dataset" affordance in the web UI under the xArm Waypoints
section, replacing the singleton.

## Decisions (from brainstorming)

1. **Full pipeline.** Each custom entry flows end-to-end: edit in web → collect →
   solve (`handeye_custom_<name>.json`) → usable in chain/polish.
2. **Operator labels.** Each entry has an operator-typed name (sanitized
   `[a-z0-9_]`, must start with a letter, ≤24 chars, unique). Filenames derive from
   the name.
3. **Auto-migrate, keep reading old keys.** Loaders fold the legacy
   `phase1_waypoints_custom` + `phase1_custom_park_*` into the new list as the first
   entry named `custom` when the new list is absent. Zero data loss; the legacy
   entry keeps the bare `handeye_custom.json` filename so existing solve/chain
   workflows and docs stay valid.

## Data model

Authoritative form in `calibration.yaml` (`collector:` section):

```yaml
phase1_custom_datasets:
- name: custom            # legacy/default entry (auto-migrated)
  park_pan_deg: 0.0
  park_tilt_deg: 0.0
  waypoints:
  - [ -0.7526, -0.9839, 1.3594, 0.8004, 0.9748, -1.1856, 0.3846 ]
  - ...
- name: high_shelf
  park_pan_deg: 15.0
  park_tilt_deg: 30.0
  waypoints: [ ... ]
```

**Migration (loader-side, both consumers):** if `phase1_custom_datasets` is absent
but legacy `phase1_waypoints_custom` is present and non-empty, synthesize
`[{name: "custom", park_pan_deg: <legacy or 0>, park_tilt_deg: <legacy or 0>,
waypoints: <legacy>}]`. The legacy flat keys (`phase1_waypoints_custom`,
`phase1_custom_park_pan_deg`, `phase1_custom_park_tilt_deg`) are dropped from the
serialized output — save/promote writes only the new list form. Both consumers
(`calib_web`, `calibrate_collect`) are updated, so nothing external reads the old keys.

## Shared helper (single source of truth for filenames)

New module `pan_tilt/calibration/custom_naming.py`:

```python
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")

def sanitize_custom_name(raw: str) -> str:
    """Lowercase, spaces/dashes -> underscore, strip illegal chars; validate."""

def custom_dataset_filenames(name: str) -> tuple[str, str]:
    """(phase1_collect_filename, handeye_solve_filename) for a custom entry.
    name == 'custom'  -> ('phase1_handeye_custom.json', 'handeye_custom.json')   # bare, legacy-compatible
    otherwise         -> ('phase1_handeye_custom_<name>.json', 'handeye_custom_<name>.json')
    """

def migrate_custom_datasets(collector: dict) -> list[dict]:
    """Return the normalized list of {name, park_pan_deg, park_tilt_deg, waypoints}
    from a collector dict, migrating legacy flat keys if the new list is absent.
    Pure; does not mutate `collector`."""
```

Used by `calibrate_collect.py`, `calib_web.py`, and `run_calibration.py` (filename
glob) so the convention lives in exactly one place.

## Collection — `calibrate_collect.py`

- `CollectConfig`: add `phase1_custom_datasets: list`. Keep the legacy fields
  (`phase1_waypoints_custom`, `phase1_custom_park_*`) so `_load_config`'s flat
  override still reads old YAMLs; after load, call `migrate_custom_datasets` to
  populate `phase1_custom_datasets` if empty.
- New ROS param `custom_name` (default `""`). When `phase:=phase1_custom`:
  resolve the dataset by `custom_name` (empty + exactly one entry ⇒ that entry;
  empty + multiple ⇒ error listing names; unknown name ⇒ error). Drive the head to
  *that entry's* park pose, run *its* waypoints (`label_prefix=f"phase1_custom_{name}"`),
  write to `custom_dataset_filenames(name)[0]`.
- `run_dry()` iterates `phase1_waypoints` + every custom dataset + `phase2_waypoints`.
- Startup config echo logs the dataset count + names.

## Web backend — `calib_web.py`

**Store.** `CalibWebNode` gains `self._custom_datasets: list[dict]` built at init via
`migrate_custom_datasets(self._loaded_cfg)`; legacy flat keys popped from
`_loaded_cfg`. `phase1_waypoints_custom` removed from the static `self._waypoints`
dict.

**Dynamic phase key.** Custom waypoints are addressed by phase key
`phase1_waypoints_custom:<name>`. `list_waypoints`/`set_waypoints` detect this prefix
and delegate to the dataset store; everything else unchanged. `_is_valid_phase()`
replaces the static `VALID_PHASES` set: accepts the static three +
`phase1_waypoints_custom:<sanitized-name>` that exists.

**Serialize / reload.** `_serialize_waypoints_yaml` drops legacy keys from `base`
and injects `phase1_custom_datasets` from the store. `reload_waypoints_from_yaml`
loads `phase1_custom_datasets` (or migrates legacy) into the store and reports
per-dataset counts. `dedupe_waypoints` also walks each dataset's waypoints.

**New endpoints (dataset lifecycle):**
- `GET  /api/calib/custom_datasets` → `[{name, park_pan_deg, park_tilt_deg, waypoints}]`
- `POST /api/calib/custom_datasets {name}` → create (sanitize, reject dupes)
- `DELETE /api/calib/custom_datasets/{name}` → remove
- `POST /api/calib/custom_datasets/{name}/park {pan_deg, tilt_deg}` → set park
  (envelope check ±30 / 0..30). The old singleton `/api/calib/phase1_custom_park`
  is removed (the UI no longer uses it).

`GET /api/waypoints` keeps returning the static three only; the custom section is
driven by `/api/calib/custom_datasets`. Per-dataset waypoint add/remove reuses
`POST /api/waypoints/phase1_waypoints_custom:<name>`.

**Run dispatch.**
- `collect_phase1_custom`: request carries `custom_name`; passed as
  `-p custom_name:=<name>`. Prereq/`phase_arg` derivation made name-aware.
- `handeye_custom`: request carries `custom_name`; solves
  `custom_dataset_filenames(name)[0]` with `--out-name custom_dataset_filenames(name)[1]`.
  Per-request prereq checks that the collect file exists.
- `chain` `--handeye` allowlist and `polish` `--phase1` allowlist become **dynamic**:
  built per-request from `{handeye.json}` / `{phase1_handeye.json}` plus every
  `handeye_custom*.json` / `phase1_handeye_custom*.json` present in the session dir,
  regex-guarded (`^(phase1_)?handeye(_custom(_[a-z0-9_]+)?)?\.json$`) against traversal.
- `GET /api/calib/session/{name}/file/{filename}` allowlist accepts the same regex
  for custom handeye files.

**Prune.** A resolver `_resolve_prune_phase(phase)` maps `phase1_waypoints_custom:<name>`
→ base `phase1_waypoints_custom` (for `PRUNE_*` lookups) + label prefix
`phase1_custom_<name>`. `_run_prune` / `prune_inputs` validate via the resolver;
`_build_payloads` / `_build_predictor` branch on the base phase and read the dataset's
waypoints via `node.list_waypoints(phase)`. `_build_pruned_collector` migrates the
source collector then edits the named dataset inside `phase1_custom_datasets` for the
custom case (vs the top-level key for the static phases).

## Solver — `run_calibration.py`

`--out-name` and multi-`--phase1` already exist; the only change is the T_ee_marker
sibling cross-check (currently hard-codes the `handeye.json ↔ handeye_custom.json`
pair): generalize so a `handeye_custom*.json` solve cross-checks against canonical
`handeye.json`, and a `handeye.json` solve cross-checks against any one existing
`handeye_custom*.json` (`sorted(out_dir.glob("handeye_custom*.json"))[0]`). Message
uses `sibling_path.name`. Preserves today's behavior when only the two canonical
files exist.

## Frontend — `app.js` + `index.html`

**Waypoints tab.** The `phase1_waypoints_custom` PHASES entry becomes a "custom
datasets" container rendered from `/api/calib/custom_datasets`:
- `+ Add custom dataset` button → name prompt → `POST /api/calib/custom_datasets`.
- One sub-group per dataset: header (name + `remove dataset`), per-entry park
  pan/tilt inputs + `Save park` (`POST .../{name}/park`), and the waypoint list
  (`+ add current joints` / `load` / `remove`) bound to phase key
  `phase1_waypoints_custom:<name>`. wpState seeds these dynamic keys from the
  datasets payload so add/remove → `pushPhase` works unchanged.

**Calibrate tab.** Session-scoped collect/solve stay here:
- Replace the single `Collect Phase 1 — custom` button with a dataset `<select>`
  (`#calib-custom-dataset-select`, populated from `/api/calib/custom_datasets`) +
  the collect button; sends `custom_name = select.value`.
- `handeye_custom` solve gets the same dataset selector; sends `custom_name`.
- `#chain-handeye-select` options and `#polish-phase1-checks` checkboxes are
  populated dynamically to include `handeye_custom[_<name>].json` /
  `phase1_handeye_custom[_<name>].json` per dataset (in addition to canonical).
- `#prune-phase` select gains one `phase1_waypoints_custom:<name>` option per dataset.

The `_phase1CustomPark` global + `_buildCustomParkControls` + `_updateCustomParkUI`
are replaced by the per-dataset rendering. Collection still requires an explicit
`confirm()` (moves the robot) and a selected session.

## Collection granularity

**Per-entry.** The operator collects one dataset at a time (each has a distinct park
pose and moves hardware). No "collect all customs" button — out of scope.

## Testing

- New `pan_tilt/test/test_custom_naming.py`: `sanitize_custom_name` rules,
  `custom_dataset_filenames` (legacy bare vs suffixed), `migrate_custom_datasets`
  (legacy → list, new list passthrough, empty).
- Existing `pan_tilt/test/test_calibration.py` (synthetic solver regression) must
  still pass after the sibling-check change.
- Build via `./scripts/build_pan_tilt_calib.sh`; smoke-check `calibrate_web` imports
  + serves `/api/calib/custom_datasets`. Hardware-in-the-loop collection/solve
  verified manually by the operator per the existing T-tiers.

## Out of scope

- "Collect all custom datasets" batch button.
- Auto-running `apply_to_urdf` per custom solve.
- Re-tuning prune default factors per dataset (they inherit the shared
  `phase1_waypoints_custom` defaults).
