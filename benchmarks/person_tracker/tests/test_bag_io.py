"""Tests for ptbench.replay.bag_io — synced-frame reader over a synthetic bag.

Authors a tiny rosbag2 (sqlite3) in ``tmp_path`` with hand-controlled header
stamps: color (bgr8) + depth (16UC1) + CameraInfo. Some depth frames are offset
from their color frame by less than the slop (must still pair to nearest); one
color frame has no depth within slop (must be skipped). Then reads it back and
asserts FrameBundle count, shapes/dtypes, K, t_ns, and the nearest-depth pairing.

Requires the ROS environment sourced (``source /opt/ros/humble/setup.bash``) so
``rosbag2_py`` / ``rclpy.serialization`` / ``sensor_msgs`` import.
"""
from __future__ import annotations

import numpy as np
import pytest

# rosbag2_py / ROS message types are only importable with ROS sourced. Skip the
# whole module cleanly otherwise rather than erroring at collection.
rosbag2_py = pytest.importorskip("rosbag2_py")
serialization = pytest.importorskip("rclpy.serialization")
sensor_msgs = pytest.importorskip("sensor_msgs.msg")

from rclpy.serialization import serialize_message  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402

from ptbench.replay.bag_io import BagIoError, read_synced_frames  # noqa: E402

COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC = "/camera/color/camera_info"

H, W = 4, 6  # tiny image
MS = 1_000_000  # nanoseconds per millisecond


def _split_stamp(t_ns: int):
    return int(t_ns // 1_000_000_000), int(t_ns % 1_000_000_000)


def _make_color(t_ns: int, fill: int) -> Image:
    sec, nsec = _split_stamp(t_ns)
    msg = Image()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nsec
    msg.header.frame_id = "camera_color_optical_frame"
    msg.height = H
    msg.width = W
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = W * 3
    msg.data = (np.full((H, W, 3), fill, dtype=np.uint8)).tobytes()
    return msg


def _make_depth(t_ns: int, fill: int) -> Image:
    sec, nsec = _split_stamp(t_ns)
    msg = Image()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nsec
    msg.header.frame_id = "camera_depth_optical_frame"
    msg.height = H
    msg.width = W
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = W * 2
    msg.data = (np.full((H, W), fill, dtype=np.uint16)).tobytes()
    return msg


def _make_info(t_ns: int, k) -> CameraInfo:
    sec, nsec = _split_stamp(t_ns)
    msg = CameraInfo()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nsec
    msg.header.frame_id = "camera_color_optical_frame"
    msg.height = H
    msg.width = W
    msg.k = [float(v) for v in k]
    return msg


def _write_bag(bag_dir, records) -> None:
    """records: list of (topic, type_str, msg, bag_stamp_ns)."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    for topic, type_str in (
        (COLOR_TOPIC, "sensor_msgs/msg/Image"),
        (DEPTH_TOPIC, "sensor_msgs/msg/Image"),
        (INFO_TOPIC, "sensor_msgs/msg/CameraInfo"),
    ):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic, type=type_str, serialization_format="cdr"
            )
        )
    for topic, _type_str, msg, bag_t in records:
        writer.write(topic, serialize_message(msg), int(bag_t))
    # Closing flushes metadata so the reader can detect topics + storage id.
    del writer


K_REF = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]


@pytest.fixture
def synced_bag(tmp_path):
    """Author a bag with controlled pairing cases.

    Timeline (ms): info@0; 4 color frames at 0/100/200/300.
      - color@0   <- depth@0   (exact)
      - color@100 <- depth@110 (10 ms, within 50 slop -> nearest, beats @130)
      - color@200 : NO depth within slop (nearest depth is @130, 70 ms) -> skip
      - color@300 <- depth@290 (10 ms, within slop)
    Two depth frames straddle color@100 (@110 closer than @130) to exercise the
    nearest pick. Depth fill encodes which depth was paired.
    """
    bag_dir = tmp_path / "synthetic_bag"
    records = []
    # CameraInfo first so intrinsics are latched before any color frame.
    records.append((INFO_TOPIC, "info", _make_info(0, K_REF), 0))
    # Colors.
    records.append((COLOR_TOPIC, "img", _make_color(0 * MS, fill=10), 0 * MS))
    records.append((COLOR_TOPIC, "img", _make_color(100 * MS, fill=20), 100 * MS))
    records.append((COLOR_TOPIC, "img", _make_color(200 * MS, fill=30), 200 * MS))
    records.append((COLOR_TOPIC, "img", _make_color(300 * MS, fill=40), 300 * MS))
    # Depths. depth fill = its stamp-ms so we can verify which one paired.
    records.append((DEPTH_TOPIC, "img", _make_depth(0 * MS, fill=0), 0 * MS))
    records.append((DEPTH_TOPIC, "img", _make_depth(110 * MS, fill=110), 110 * MS))
    records.append((DEPTH_TOPIC, "img", _make_depth(130 * MS, fill=130), 130 * MS))
    records.append((DEPTH_TOPIC, "img", _make_depth(290 * MS, fill=290), 290 * MS))
    _write_bag(bag_dir, records)
    return bag_dir


class TestReadSyncedFrames:
    def test_count_skips_color_without_depth(self, synced_bag):
        bundles = list(
            read_synced_frames(
                synced_bag, COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC, slop_ns=50 * MS
            )
        )
        # color@200 has no depth within 50 ms -> skipped. 3 remain.
        assert len(bundles) == 3
        assert [b.t_ns for b in bundles] == [0 * MS, 100 * MS, 300 * MS]

    def test_shapes_and_dtypes(self, synced_bag):
        bundles = list(
            read_synced_frames(
                synced_bag, COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC, slop_ns=50 * MS
            )
        )
        for b in bundles:
            assert b.color_bgr.shape == (H, W, 3)
            assert b.color_bgr.dtype == np.uint8
            assert b.depth_mm.shape == (H, W)
            assert b.depth_mm.dtype == np.uint16

    def test_intrinsics_latched(self, synced_bag):
        bundles = list(
            read_synced_frames(
                synced_bag, COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC, slop_ns=50 * MS
            )
        )
        for b in bundles:
            assert list(b.K) == K_REF

    def test_nearest_depth_pairing(self, synced_bag):
        bundles = list(
            read_synced_frames(
                synced_bag, COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC, slop_ns=50 * MS
            )
        )
        by_t = {b.t_ns: b for b in bundles}
        # color@0 paired depth@0 (fill 0)
        assert int(by_t[0 * MS].depth_mm[0, 0]) == 0
        # color@100 paired the NEAREST of depth@110 / depth@150 -> @110 (fill 110)
        assert int(by_t[100 * MS].depth_mm[0, 0]) == 110
        # color@300 paired depth@290 (fill 290)
        assert int(by_t[300 * MS].depth_mm[0, 0]) == 290

    def test_color_fill_preserved(self, synced_bag):
        bundles = list(
            read_synced_frames(
                synced_bag, COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC, slop_ns=50 * MS
            )
        )
        by_t = {b.t_ns: b for b in bundles}
        assert int(by_t[0 * MS].color_bgr[0, 0, 0]) == 10
        assert int(by_t[100 * MS].color_bgr[0, 0, 0]) == 20
        assert int(by_t[300 * MS].color_bgr[0, 0, 0]) == 40

    def test_tighter_slop_drops_more(self, synced_bag):
        # With a 5 ms slop, only color@0 (exact) survives; @100/@300 are 10 ms off.
        bundles = list(
            read_synced_frames(
                synced_bag, COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC, slop_ns=5 * MS
            )
        )
        assert [b.t_ns for b in bundles] == [0 * MS]

    def test_missing_topic_raises(self, synced_bag):
        with pytest.raises(BagIoError) as exc:
            list(
                read_synced_frames(
                    synced_bag,
                    COLOR_TOPIC,
                    "/does/not/exist",
                    INFO_TOPIC,
                    slop_ns=50 * MS,
                )
            )
        assert "/does/not/exist" in str(exc.value)


class TestEncodings:
    def test_rgb8_color_converted_to_bgr(self, tmp_path):
        bag_dir = tmp_path / "rgb_bag"
        # Build an rgb8 color frame with a known per-channel value, paired depth.
        sec, nsec = _split_stamp(0)
        color = Image()
        color.header.stamp.sec = sec
        color.header.stamp.nanosec = nsec
        color.height = H
        color.width = W
        color.encoding = "rgb8"
        color.is_bigendian = 0
        color.step = W * 3
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[..., 0] = 1  # R
        rgb[..., 1] = 2  # G
        rgb[..., 2] = 3  # B
        color.data = rgb.tobytes()
        records = [
            (INFO_TOPIC, "info", _make_info(0, K_REF), 0),
            (COLOR_TOPIC, "img", color, 0),
            (DEPTH_TOPIC, "img", _make_depth(0, fill=5), 0),
        ]
        _write_bag(bag_dir, records)
        bundles = list(
            read_synced_frames(bag_dir, COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC)
        )
        assert len(bundles) == 1
        px = bundles[0].color_bgr[0, 0]
        # RGB (1,2,3) -> BGR (3,2,1)
        assert (int(px[0]), int(px[1]), int(px[2])) == (3, 2, 1)

    def test_bad_depth_encoding_raises(self, tmp_path):
        bag_dir = tmp_path / "bad_depth_bag"
        depth = _make_depth(0, fill=1)
        depth.encoding = "32FC1"  # unsupported
        records = [
            (INFO_TOPIC, "info", _make_info(0, K_REF), 0),
            (COLOR_TOPIC, "img", _make_color(0, fill=10), 0),
            (DEPTH_TOPIC, "img", depth, 0),
        ]
        _write_bag(bag_dir, records)
        with pytest.raises(BagIoError) as exc:
            list(read_synced_frames(bag_dir, COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC))
        assert "32FC1" in str(exc.value)
