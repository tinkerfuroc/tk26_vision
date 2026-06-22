"""Read a rosbag2 directory and yield time-synced color + depth + intrinsics.

This is the input side of the replay harness. Given a rosbag2 directory and the
three camera topics (color image, depth image, camera_info), it walks the bag in
recorded order and pairs each color message with the depth message whose header
stamp is nearest within ``slop_ns``. CameraInfo intrinsics (``k``) are latched
(most-recent-wins) so every emitted bundle carries the intrinsics in force at
that color frame.

Timestamps come from the message **header** stamp
(``sec * 1_000_000_000 + nanosec``), not the bag's receive time, so they line up
with the GT annotation ``t_ns`` (which is also a header stamp).

Only the Orbbec defaults are required: color ``bgr8`` and depth ``16UC1``. For
convenience color ``rgb8`` is also accepted (converted to BGR). Anything else
raises :class:`BagIoError`.

``rosbag2_py``, ``rclpy.serialization`` and the ``sensor_msgs`` types import
fine in the vision venv once the ROS environment is sourced
(``source /opt/ros/humble/setup.bash``). They are imported at module load —
this module is meant to be used (and tested) only with ROS available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image


class BagIoError(Exception):
    """Raised when a bag is missing a required topic or carries bad encodings."""


@dataclass
class FrameBundle:
    """One time-synced color+depth+intrinsics sample.

    Attributes:
        t_ns: color message header stamp in nanoseconds
            (``sec * 1_000_000_000 + nanosec``).
        color_bgr: HxWx3 uint8 image in BGR order.
        depth_mm: HxW uint16 depth image in millimetres.
        K: len-9 row-major camera intrinsics (``CameraInfo.k``).
    """

    t_ns: int
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    K: list


def _stamp_ns(msg) -> int:
    """Header stamp of an Image/CameraInfo message in nanoseconds."""
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _decode_color(msg: Image) -> np.ndarray:
    """Decode a color Image message to an HxWx3 uint8 BGR array."""
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding
    if enc == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    if enc == "rgb8":
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return rgb[:, :, ::-1].copy()  # RGB -> BGR
    raise BagIoError(
        f"unsupported color encoding {enc!r} (expected 'bgr8' or 'rgb8')"
    )


def _decode_depth(msg: Image) -> np.ndarray:
    """Decode a depth Image message to an HxW uint16 (millimetres) array."""
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding
    if enc != "16UC1":
        raise BagIoError(
            f"unsupported depth encoding {enc!r} (expected '16UC1')"
        )
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)


def _open_reader(bag_dir) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir), storage_id=""
    )  # empty storage_id => auto-detect from metadata
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def _check_topics(reader, color_topic, depth_topic, camera_info_topic) -> None:
    available = {t.name for t in reader.get_all_topics_and_types()}
    missing = [
        t
        for t in (color_topic, depth_topic, camera_info_topic)
        if t not in available
    ]
    if missing:
        raise BagIoError(
            f"bag is missing required topic(s): {missing!r}. "
            f"available topics: {sorted(available)!r}"
        )


def _pair_depth(
    color_t: int,
    depths: List[Tuple[int, Image]],
    cursor: int,
    slop_ns: int,
) -> Tuple[Optional[Image], int]:
    """Find the depth message whose stamp is nearest ``color_t`` within slop.

    ``depths`` is the time-ordered list of (stamp_ns, msg). ``cursor`` is a
    monotonic hint into ``depths`` (depths older than it cannot be nearest to
    any later color frame). Returns ``(depth_msg_or_None, new_cursor)``.
    """
    n = len(depths)
    # Advance the cursor while the next depth is strictly closer to color_t.
    i = cursor
    while i + 1 < n and abs(depths[i + 1][0] - color_t) <= abs(
        depths[i][0] - color_t
    ):
        i += 1
    if i >= n:
        return None, cursor
    best_t, best_msg = depths[i]
    if abs(best_t - color_t) <= slop_ns:
        return best_msg, i
    return None, i


def read_synced_frames(
    bag_dir,
    color_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    slop_ns: int = 50_000_000,
) -> Iterator[FrameBundle]:
    """Yield time-synced :class:`FrameBundle`s from a rosbag2 directory.

    For each color message in recorded (time) order, the depth message with the
    nearest header stamp is paired if it falls within ``slop_ns``; color frames
    with no depth in slop are skipped. The most recent ``CameraInfo.k`` seen so
    far supplies the intrinsics; color frames that arrive before any CameraInfo
    are skipped.

    Args:
        bag_dir: path to a rosbag2 directory (sqlite3 or mcap).
        color_topic: color image topic (``bgr8`` or ``rgb8``).
        depth_topic: depth image topic (``16UC1``).
        camera_info_topic: CameraInfo topic.
        slop_ns: max |color_stamp - depth_stamp| to accept a pairing (default
            50 ms).

    Yields:
        :class:`FrameBundle` in color-frame time order.

    Raises:
        BagIoError: if any required topic is absent from the bag, or a message
            carries an unsupported encoding.
    """
    reader = _open_reader(bag_dir)
    _check_topics(reader, color_topic, depth_topic, camera_info_topic)

    # First pass: collect color + depth messages (deserialized) in time order,
    # and latch CameraInfo intrinsics keyed by stamp so each color frame can
    # use the most-recent-as-of-that-frame K. Depth frames are decoded lazily on
    # pairing to avoid decoding ones that never match.
    colors: List[Tuple[int, Image]] = []
    depths: List[Tuple[int, Image]] = []
    cam_infos: List[Tuple[int, list]] = []

    while reader.has_next():
        topic, raw, _bag_t = reader.read_next()
        if topic == color_topic:
            msg = deserialize_message(raw, Image)
            colors.append((_stamp_ns(msg), msg))
        elif topic == depth_topic:
            msg = deserialize_message(raw, Image)
            depths.append((_stamp_ns(msg), msg))
        elif topic == camera_info_topic:
            msg = deserialize_message(raw, CameraInfo)
            cam_infos.append((_stamp_ns(msg), list(msg.k)))

    colors.sort(key=lambda x: x[0])
    depths.sort(key=lambda x: x[0])
    cam_infos.sort(key=lambda x: x[0])

    depth_cursor = 0
    info_cursor = 0
    latched_K: Optional[list] = None

    for color_t, color_msg in colors:
        # Latch the most-recent CameraInfo whose stamp is <= this color stamp;
        # if none precedes it, fall back to the very first CameraInfo.
        while (
            info_cursor + 1 < len(cam_infos)
            and cam_infos[info_cursor + 1][0] <= color_t
        ):
            info_cursor += 1
        if cam_infos:
            if cam_infos[info_cursor][0] <= color_t:
                latched_K = cam_infos[info_cursor][1]
            elif latched_K is None:
                latched_K = cam_infos[0][1]
        if latched_K is None:
            continue  # no intrinsics yet; cannot build a usable bundle

        depth_msg, depth_cursor = _pair_depth(
            color_t, depths, depth_cursor, slop_ns
        )
        if depth_msg is None:
            continue  # no depth within slop; skip this color frame

        yield FrameBundle(
            t_ns=color_t,
            color_bgr=_decode_color(color_msg),
            depth_mm=_decode_depth(depth_msg),
            K=list(latched_K),
        )
