"""Naming + migration helpers for multiple Phase-1 hand-eye CUSTOM datasets.

The pan-tilt calibration supports several independent "custom" hand-eye
datasets, each with its own operator-chosen park pose and xArm waypoint list.
This module is the SINGLE source of truth for:

  * how operator-typed names are sanitized,
  * how a dataset name maps to its on-disk JSON filenames, and
  * how a legacy single-custom calibration.yaml migrates to the named-list form.

Both ``calibrate_collect`` and ``calib_web`` (and the filename glob in
``run_calibration``) import from here so the convention never drifts between the
collector and the solver/UI.

Pure module: no ROS, no numpy, no I/O — trivially unit-testable.
"""

from __future__ import annotations

import re

# Sanitized names: start with a letter, then [a-z0-9_], total length 1..24.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")

# The reserved/default entry. Keeps the BARE legacy filenames so existing
# solve/chain workflows, docs, and prior session outputs stay valid.
LEGACY_NAME = "custom"


def sanitize_custom_name(raw: str) -> str:
    """Normalize an operator-typed dataset name to a filesystem-safe slug.

    Lowercases, folds runs of spaces/dashes/dots into single underscores, drops
    any remaining illegal characters, and validates against ``NAME_RE``.

    Raises ValueError if the result is empty, starts with a non-letter, or
    exceeds 24 characters — callers surface this as a 400 to the operator.
    """
    if not isinstance(raw, str):
        raise ValueError(f"custom dataset name must be a string, got {type(raw)!r}")
    s = raw.strip().lower()
    s = re.sub(r"[\s\-.]+", "_", s)      # spaces/dashes/dots -> underscore
    s = re.sub(r"[^a-z0-9_]", "", s)     # drop anything still illegal
    s = re.sub(r"_+", "_", s).strip("_")  # collapse + trim underscores
    if not NAME_RE.match(s):
        raise ValueError(
            f"invalid custom dataset name {raw!r} -> {s!r}: must start with a "
            "letter and contain only [a-z0-9_], max 24 chars"
        )
    return s


def custom_dataset_filenames(name: str) -> tuple[str, str]:
    """Return ``(phase1_collect_filename, handeye_solve_filename)`` for a dataset.

    The reserved ``custom`` entry keeps the historical bare filenames so nothing
    that already references them breaks; every other name is suffixed.

        custom      -> ('phase1_handeye_custom.json', 'handeye_custom.json')
        high_shelf  -> ('phase1_handeye_custom_high_shelf.json',
                        'handeye_custom_high_shelf.json')
    """
    if name == LEGACY_NAME:
        return "phase1_handeye_custom.json", "handeye_custom.json"
    return (
        f"phase1_handeye_custom_{name}.json",
        f"handeye_custom_{name}.json",
    )


def normalize_dataset(entry: dict) -> dict:
    """Coerce a raw dataset dict into the canonical shape with defaulted fields."""
    name = sanitize_custom_name(str(entry.get("name", LEGACY_NAME)))
    wps = entry.get("waypoints") or []
    waypoints = [list(w) for w in wps] if isinstance(wps, list) else []
    return {
        "name": name,
        "park_pan_deg": float(entry.get("park_pan_deg", 0.0) or 0.0),
        "park_tilt_deg": float(entry.get("park_tilt_deg", 0.0) or 0.0),
        "waypoints": waypoints,
    }


def migrate_custom_datasets(collector: dict) -> list[dict]:
    """Return the normalized list of custom datasets from a collector dict.

    Resolution order (does NOT mutate ``collector``):
      1. If ``phase1_custom_datasets`` is present and non-empty, normalize it.
      2. Else if legacy ``phase1_waypoints_custom`` is present and non-empty,
         synthesize a single entry named ``custom`` carrying the legacy park
         pose (defaulting to 0/0).
      3. Else return ``[]``.
    """
    new_list = collector.get("phase1_custom_datasets")
    if isinstance(new_list, list) and new_list:
        return [normalize_dataset(e) for e in new_list if isinstance(e, dict)]

    legacy_wps = collector.get("phase1_waypoints_custom")
    if isinstance(legacy_wps, list) and legacy_wps:
        return [normalize_dataset({
            "name": LEGACY_NAME,
            "park_pan_deg": collector.get("phase1_custom_park_pan_deg", 0.0),
            "park_tilt_deg": collector.get("phase1_custom_park_tilt_deg", 0.0),
            "waypoints": legacy_wps,
        })]
    return []
