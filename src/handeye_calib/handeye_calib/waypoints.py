"""Waypoint store + per-robot YAML persistence for handeye_calib.

Pure Python + pyyaml — no ROS, no rclpy. Unit-testable without sourcing ROS.
Used by HandeyeWebNode (handeye_web.py) to record a sequence of arm joint
positions that the auto-capture state machine can cycle through.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml

YAML_SCHEMA_VERSION = 1


class WaypointStore:
    """In-memory ordered list of 7-DOF joint waypoints (radians).

    Thread safety: the store itself is NOT thread-safe. Callers (HandeyeWebNode)
    hold their own lock around all store operations.
    """

    def __init__(self):
        self._waypoints: list[list[float]] = []

    def list(self) -> list[list[float]]:
        """Return a shallow copy of the waypoint list."""
        return [list(w) for w in self._waypoints]

    def add(self, joints_rad: Sequence[float]) -> int:
        """Append a waypoint; returns the new index.

        Raises ValueError if ``joints_rad`` does not have exactly 7 elements.
        """
        joints = list(joints_rad)
        if len(joints) != 7:
            raise ValueError(
                f"expected 7 joint values, got {len(joints)}"
            )
        self._waypoints.append([float(j) for j in joints])
        return len(self._waypoints) - 1

    def delete(self, idx: int) -> bool:
        """Delete waypoint at ``idx``.

        Returns True on success, False when ``idx`` is out of range.
        """
        if not isinstance(idx, int) or idx < 0 or idx >= len(self._waypoints):
            return False
        self._waypoints.pop(idx)
        return True

    def clear(self) -> None:
        """Remove all waypoints."""
        self._waypoints.clear()

    def load_yaml(self, path: Path) -> int:
        """Replace the in-memory list from a YAML file; returns count loaded.

        Raises ValueError on schema mismatch or parse error.
        """
        text = Path(path).read_text()
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML parse error: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("expected a YAML mapping at top level")
        version = data.get("schema_version")
        if version != YAML_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version!r} "
                f"(expected {YAML_SCHEMA_VERSION})"
            )
        raw = data.get("waypoints", [])
        loaded: list[list[float]] = []
        for i, row in enumerate(raw):
            row_f = [float(v) for v in row]
            if len(row_f) != 7:
                raise ValueError(
                    f"waypoint {i}: expected 7 values, got {len(row_f)}"
                )
            loaded.append(row_f)
        self._waypoints = loaded
        return len(self._waypoints)

    def save_yaml(self, path: Path, recorded_for_robot: str = "") -> None:
        """Atomically write waypoints to ``path`` via write_with_backup.

        Existing files are backed up to ``<path>.old-<timestamp>``
        (consistent with T6 promote's backup discipline).
        """
        from handeye_calib import apply_handeye as ah  # local import — keeps module ROS-free

        payload = {
            "schema_version": YAML_SCHEMA_VERSION,
            "recorded_for_robot": str(recorded_for_robot),
            "waypoints": [list(w) for w in self._waypoints],
        }
        ah.write_with_backup(str(path), yaml.safe_dump(payload, sort_keys=False))


def resolve_waypoints_path(robot_name, basic_repo_root) -> "Path | None":
    """Resolve the per-robot waypoints YAML path.

    Returns ``<basic_repo_root>/src/tinker_robot_config/robots/<robot_name>/handeye_waypoints.yaml``
    when ``robot_name`` is truthy, else ``None``.

    Mirrors ``apply_handeye.resolve_robot_xacro_path`` exactly — same shape,
    same "does NOT check existence" contract.
    """
    if not robot_name:
        return None
    return (Path(basic_repo_root) / "src" / "tinker_robot_config"
            / "robots" / robot_name / "handeye_waypoints.yaml")
