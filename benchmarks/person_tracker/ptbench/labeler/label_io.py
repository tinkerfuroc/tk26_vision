"""PURE labeling logic + a small self-contained rosbag2 reader.

This module deliberately re-implements a *minimal* rosbag2 color/depth/info
reader instead of importing :mod:`ptbench.replay.bag_io`. The labeler is an
offline tool that only needs to step frames and sample nearest-by-timestamp
depth (no slop window, no live sync), and ``replay`` is being built
concurrently. A future refactor can dedupe these readers against
``replay.bag_io`` once both have landed.

Everything except the rosbag2 reader is framework-free and fully unit-tested:

- :func:`propagate_default` — copy-the-box-forward default for the next frame.
- :func:`nearest_depth` — nearest depth frame by ``|t_ns|`` (no slop limit).
- :func:`build_gt_clip` — assemble a schema-valid :class:`GtClip` from a list of
  per-frame annotations, sampling 3D centroids from depth via
  :func:`ptbench.common.geometry.centroid_from_bbox_depth`.

The rosbag2 reader (:func:`read_color_frames`, :func:`read_depth_and_info`)
needs ``rosbag2_py`` + ``rclpy`` + ``sensor_msgs`` on the path; it is only
invoked by the CLI, never by the unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ptbench.common.geometry import centroid_from_bbox_depth
from ptbench.common.schema import GtClip, GtFrame

Bbox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Per-frame annotation (in-progress, in-memory)
# ---------------------------------------------------------------------------

@dataclass
class FrameAnnotation:
    """One labeled color frame, before depth sampling.

    ``bbox`` is ``(x1, y1, x2, y2)`` in color pixels, or ``None``. The schema
    invariant ``present=True ⇒ bbox is not None`` is enforced by
    :func:`build_gt_clip` (it drops the box on an inconsistent annotation).
    """

    t_ns: int
    present: bool
    bbox: Optional[Bbox]


# ---------------------------------------------------------------------------
# Pure labeling helpers
# ---------------------------------------------------------------------------

def propagate_default(prev: Optional[FrameAnnotation]) -> Optional[Bbox]:
    """Return the default bbox for the next frame.

    The labeler "copies the box forward": if the previous frame was present and
    had a box, that box seeds the next frame; otherwise the next frame starts
    with no box.
    """
    if prev is None:
        return None
    if not prev.present:
        return None
    return prev.bbox


def nearest_depth(
    depth_list: List[Tuple[int, np.ndarray]], t_ns: int
) -> Optional[np.ndarray]:
    """Return the depth image whose stamp is nearest ``t_ns`` by ``|Δt|``.

    No slop limit — the labeler is offline and always wants *some* depth frame
    to sample from. Returns ``None`` only for an empty ``depth_list``.
    """
    if not depth_list:
        return None
    best = min(depth_list, key=lambda item: abs(item[0] - t_ns))
    return best[1]


def build_gt_clip(
    annotations: List[FrameAnnotation],
    depth_list: List[Tuple[int, np.ndarray]],
    K,
    *,
    clip_id: str,
    bag_path: str,
    scenario: str,
    color_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    fps_hint: float = 30.0,
    notes: str = "",
) -> GtClip:
    """Assemble a schema-valid :class:`GtClip` from per-frame annotations.

    For each annotation, if it is present with a box, the 3D centroid is sampled
    from the nearest depth frame via
    :func:`ptbench.common.geometry.centroid_from_bbox_depth`. The centroid may
    come back ``None`` (sparse/invalid depth); that is allowed — ``present``
    stays ``True`` with ``centroid_3d=None`` (the schema only requires a non-null
    bbox for present frames, not a centroid).

    To keep the result schema-valid:

    - Frames are emitted in strictly-increasing ``t_ns`` order (sorted, with
      duplicate stamps dropped — keeping the first).
    - An annotation that is ``present`` but has no box is downgraded to
      ``present=False`` (the schema forbids present-without-bbox).

    The returned clip round-trips through ``save_gt``/``load_gt`` without raising.
    """
    # Sort by t_ns and drop duplicate stamps so the schema's strictly-increasing
    # invariant holds even if the UI produced out-of-order / repeated stamps.
    ordered: List[FrameAnnotation] = sorted(annotations, key=lambda a: a.t_ns)

    frames: List[GtFrame] = []
    seen_t: set = set()
    for ann in ordered:
        if ann.t_ns in seen_t:
            continue
        seen_t.add(ann.t_ns)

        bbox = ann.bbox
        present = bool(ann.present)
        if present and bbox is None:
            # Schema forbids present=True with a null bbox; downgrade.
            present = False

        centroid = None
        if present and bbox is not None:
            depth = nearest_depth(depth_list, ann.t_ns)
            if depth is not None:
                centroid = centroid_from_bbox_depth(depth, K, bbox)

        frames.append(
            GtFrame(
                t_ns=ann.t_ns,
                present=present,
                bbox=tuple(bbox) if (present and bbox is not None) else None,
                centroid_3d=tuple(centroid) if centroid is not None else None,
            )
        )

    return GtClip(
        schema_version="1.0",
        clip_id=clip_id,
        bag_path=bag_path,
        scenario=scenario,
        color_topic=color_topic,
        depth_topic=depth_topic,
        camera_info_topic=camera_info_topic,
        fps_hint=fps_hint,
        notes=notes,
        frames=frames,
    )


# ---------------------------------------------------------------------------
# Minimal rosbag2 reader (NOT unit-tested; needs the ROS stack)
# ---------------------------------------------------------------------------

def _stamp_ns(header) -> int:
    """ROS ``std_msgs/Header`` stamp -> int nanoseconds."""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _open_reader(bag_dir: str):
    """Open a SequentialReader on ``bag_dir`` (auto-detected storage id)."""
    import rosbag2_py

    storage_id = ""
    try:
        # Prefer the metadata's storage id; fall back to the default.
        info = rosbag2_py.Info()
        meta = info.read_metadata(str(bag_dir), "")
        storage_id = meta.storage_identifier
    except Exception:
        storage_id = rosbag2_py.get_default_storage_id()

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def read_color_frames(
    bag_dir: str, color_topic: str
) -> List[Tuple[int, np.ndarray]]:
    """Read all color frames from ``color_topic`` as ``(t_ns, bgr ndarray)``.

    Assumes ``bgr8`` encoding (HxWx3 uint8). ``t_ns`` is the color header stamp.
    Returns frames in stored order (typically time order).
    """
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    reader = _open_reader(bag_dir)
    out: List[Tuple[int, np.ndarray]] = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != color_topic:
            continue
        msg = deserialize_message(data, Image)
        arr = _image_to_ndarray(msg)
        out.append((_stamp_ns(msg.header), arr))
    return out


def read_depth_and_info(
    bag_dir: str, depth_topic: str, camera_info_topic: str
) -> Tuple[List[Tuple[int, np.ndarray]], Optional[np.ndarray]]:
    """Read depth frames + the first CameraInfo K from a bag.

    Returns ``(depth_list, K)`` where ``depth_list`` is ``[(t_ns, depth_mm)]``
    (depth as a 2D array, native dtype — ``16UC1`` millimeters is the expected
    case) and ``K`` is a length-9 row-major intrinsics array, or ``None`` if no
    CameraInfo was found on ``camera_info_topic``.

    Record depth as **16UC1 (millimetres)** — the Orbbec default and the only
    encoding the scorer (:mod:`ptbench.replay.bag_io`) accepts. A clip labeled
    from a non-16UC1 depth bag may not be scoreable later (and ``32FC1`` is
    returned raw, NOT converted to millimetres — see ``_image_to_ndarray``).
    """
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo, Image

    reader = _open_reader(bag_dir)
    depths: List[Tuple[int, np.ndarray]] = []
    K: Optional[np.ndarray] = None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == depth_topic:
            msg = deserialize_message(data, Image)
            depths.append((_stamp_ns(msg.header), _image_to_ndarray(msg)))
        elif topic == camera_info_topic and K is None:
            info = deserialize_message(data, CameraInfo)
            K = np.asarray(info.k, dtype=np.float64).reshape(-1)
    return depths, K


def _image_to_ndarray(msg) -> np.ndarray:
    """Decode a ``sensor_msgs/Image`` into a numpy array.

    Handles the encodings this tool cares about: ``bgr8``/``rgb8`` (HxWx3
    uint8) and depth (``16UC1`` -> uint16 HxW, ``32FC1`` -> float32 HxW). Other
    encodings are returned as a raw uint8 buffer reshaped to (H, W, -1).
    """
    enc = msg.encoding
    h, w = msg.height, msg.width
    if enc in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    elif enc in ("16UC1", "mono16"):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    elif enc == "32FC1":
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
    elif enc in ("mono8", "8UC1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
    else:
        # Best-effort: raw bytes shaped to the row stride.
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
    return np.array(arr)  # copy out of the read-only buffer
