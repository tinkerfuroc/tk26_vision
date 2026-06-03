"""GT annotation dataclasses + JSON load/save/validate.

One JSON file per clip describes operator-labeled ground truth: per-frame
presence, a 2D bounding box in color pixels, and an optional 3D centroid in the
camera optical frame. The on-disk shape stores bbox/centroid as arrays (or
``null``); the in-memory dataclasses use tuples.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

SCHEMA_VERSION = "1.1"
SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1")


class GtSchemaError(Exception):
    """Raised when a GtClip fails schema validation."""


@dataclass
class GtFrame:
    t_ns: int
    present: bool
    bbox: Optional[Tuple[float, float, float, float]]
    # Best-available estimate (mask + robust median) — the gate scores this.
    centroid_field: Optional[Tuple[float, float, float]] = None
    # Node-identical math (no mask) — reported as a diagnostic only.
    centroid_track: Optional[Tuple[float, float, float]] = None
    # transitional alias (parallel-change); removed in Phase 0 cleanup once all consumers use centroid_field/centroid_track
    centroid_3d: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        # Keep the transitional alias in sync with the canonical fields so old
        # GtFrame(centroid_3d=x) construction and old frame.centroid_3d reads
        # both work during the migration.
        if self.centroid_3d is not None:
            if self.centroid_field is None:
                self.centroid_field = self.centroid_3d
            if self.centroid_track is None:
                self.centroid_track = self.centroid_3d
        elif self.centroid_field is not None:
            self.centroid_3d = self.centroid_field


@dataclass
class GtClip:
    schema_version: str
    clip_id: str
    bag_path: str
    scenario: str
    color_topic: str = "/camera/color/image_raw"
    depth_topic: str = "/camera/depth/image_raw"
    camera_info_topic: str = "/camera/color/camera_info"
    fps_hint: float = 30.0
    notes: str = ""
    frames: list = field(default_factory=list)


def _is_finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate(clip: GtClip) -> None:
    """Validate a GtClip, raising GtSchemaError on the first problem found.

    Error messages contain the offending field name as a substring.
    """
    if clip.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise GtSchemaError(
            f"unsupported schema_version {clip.schema_version!r} "
            f"(expected one of {SUPPORTED_SCHEMA_VERSIONS!r})"
        )

    if not clip.frames:
        raise GtSchemaError("frames must be a non-empty list")

    prev_t = None
    for i, f in enumerate(clip.frames):
        if prev_t is not None and f.t_ns <= prev_t:
            raise GtSchemaError(
                f"frame {i}: t_ns must be strictly increasing "
                f"(got {f.t_ns} after {prev_t})"
            )
        prev_t = f.t_ns

        if f.bbox is not None:
            bbox = tuple(f.bbox)
            if len(bbox) != 4 or not all(_is_finite_number(v) for v in bbox):
                raise GtSchemaError(
                    f"frame {i}: bbox must be 4 finite numbers, got {f.bbox!r}"
                )
            x1, y1, x2, y2 = bbox
            if x2 <= x1:
                raise GtSchemaError(f"frame {i}: bbox x2 must be > x1, got {f.bbox!r}")
            if y2 <= y1:
                raise GtSchemaError(f"frame {i}: bbox y2 must be > y1, got {f.bbox!r}")

        if f.present and f.bbox is None:
            raise GtSchemaError(f"frame {i}: present=True requires a non-null bbox")

        for field_name in ("centroid_field", "centroid_track"):
            val = getattr(f, field_name)
            if val is not None:
                c = tuple(val)
                if len(c) != 3 or not all(_is_finite_number(v) for v in c):
                    raise GtSchemaError(
                        f"frame {i}: {field_name} must be 3 finite numbers, "
                        f"got {val!r}"
                    )


def _frame_to_dict(f: GtFrame) -> dict:
    return {
        "t_ns": f.t_ns,
        "present": f.present,
        "bbox": list(f.bbox) if f.bbox is not None else None,
        "centroid_field": list(f.centroid_field) if f.centroid_field is not None else None,
        "centroid_track": list(f.centroid_track) if f.centroid_track is not None else None,
    }


def _clip_to_dict(clip: GtClip) -> dict:
    return {
        "schema_version": clip.schema_version,
        "clip_id": clip.clip_id,
        "bag_path": clip.bag_path,
        "scenario": clip.scenario,
        "color_topic": clip.color_topic,
        "depth_topic": clip.depth_topic,
        "camera_info_topic": clip.camera_info_topic,
        "fps_hint": clip.fps_hint,
        "notes": clip.notes,
        "frames": [_frame_to_dict(f) for f in clip.frames],
    }


def _frame_from_dict(d: dict) -> GtFrame:
    bbox = d.get("bbox")
    # 1.1 carries centroid_field/centroid_track; 1.0 carries a single
    # centroid_3d that maps onto both.
    if "centroid_field" in d or "centroid_track" in d:
        cf = d.get("centroid_field")
        ct = d.get("centroid_track")
    else:
        legacy = d.get("centroid_3d")
        cf = legacy
        ct = legacy
    return GtFrame(
        t_ns=d["t_ns"],
        present=d["present"],
        bbox=tuple(bbox) if bbox is not None else None,
        centroid_field=tuple(cf) if cf is not None else None,
        centroid_track=tuple(ct) if ct is not None else None,
    )


def _clip_from_dict(d: dict) -> GtClip:
    return GtClip(
        schema_version=d["schema_version"],
        clip_id=d["clip_id"],
        bag_path=d["bag_path"],
        scenario=d["scenario"],
        color_topic=d.get("color_topic", "/camera/color/image_raw"),
        depth_topic=d.get("depth_topic", "/camera/depth/image_raw"),
        camera_info_topic=d.get("camera_info_topic", "/camera/color/camera_info"),
        fps_hint=d.get("fps_hint", 30.0),
        notes=d.get("notes", ""),
        frames=[_frame_from_dict(fd) for fd in d.get("frames", [])],
    )


def save_gt(clip: GtClip, path) -> None:
    """Write a GtClip to ``path`` as canonical JSON.

    Does not validate — ``load_gt`` is the validation boundary. This lets tools
    write partial/in-progress GT and lets tests round-trip deliberately-invalid
    fixtures without save-time rejection.
    """
    Path(path).write_text(json.dumps(_clip_to_dict(clip), indent=2))


def load_gt(path) -> GtClip:
    """Load a GtClip from ``path`` and validate it before returning."""
    raw = json.loads(Path(path).read_text())
    clip = _clip_from_dict(raw)
    validate(clip)
    return clip
