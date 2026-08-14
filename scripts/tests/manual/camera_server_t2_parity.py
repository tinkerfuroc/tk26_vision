#!/usr/bin/env python3
"""T2 live-camera check for freshness, TF-at-stamp, and cloud parity.

Examples::

    python3 camera_server_t2_parity.py --server /head_camera_server \
        --legacy-pc-service /get_orbbec_pc \
        --driver-cloud /camera/depth_registered/points
    python3 camera_server_t2_parity.py --server /wrist_camera_server

The script does not start camera drivers.  Run it only after the relevant
bringup is already running; a failed live check is an operator action item.
"""
import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from tinker_vision_msgs_26.srv import (
    GetCameraPointCloud,
    GetCameraSnapshot,
    GetOrbbecPC,
    GetTransform,
)


RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"[camera_server_t2] {'PASS' if ok else 'FAIL'}: {name} {detail}")


def timestamp_ns(stamp):
    return Time.from_msg(stamp).nanoseconds


def centroid(cloud):
    count = 0
    sums = [0.0, 0.0, 0.0]
    for x, y, z in pc2.read_points(
        cloud, field_names=("x", "y", "z"), skip_nans=True
    ):
        sums[0] += float(x)
        sums[1] += float(y)
        sums[2] += float(z)
        count += 1
    if not count:
        return 0, (0.0, 0.0, 0.0)
    return count, tuple(value / count for value in sums)


class T2Node(Node):
    def __init__(self, server):
        super().__init__("camera_server_t2")
        self.snapshot = self.create_client(
            GetCameraSnapshot, server.rstrip("/") + "/get_snapshot"
        )
        self.cloud = self.create_client(
            GetCameraPointCloud, server.rstrip("/") + "/get_point_cloud"
        )
        self.transform = self.create_client(
            GetTransform, server.rstrip("/") + "/get_transform"
        )

    def call(self, client, request, timeout=15.0):
        try:
            if not client.wait_for_service(timeout_sec=min(5.0, timeout)):
                return None
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
            if not future.done():
                return None
            return future.result()
        except Exception as exc:  # operator-facing harness: report, don't traceback
            self.get_logger().warning("service call failed: %s", exc)
            return None


def wait_for_driver_cloud(node, topic, timeout=10.0):
    received = []

    def callback(message):
        if not received:
            received.append(message)

    subscription = node.create_subscription(
        PointCloud2, topic, callback, rclpy.qos.qos_profile_sensor_data
    )
    deadline = time.monotonic() + timeout
    while not received and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_subscription(subscription)
    return received[0] if received else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--legacy-pc-service", default="")
    parser.add_argument("--driver-cloud", default="")
    parser.add_argument("--target-frame", default="base_link")
    args = parser.parse_args()

    rclpy.init()
    node = T2Node(args.server)
    try:
        # 1. Newest pair is available and internally synchronized.
        first = node.call(node.snapshot, GetCameraSnapshot.Request())
        ok = first is not None and first.status == GetCameraSnapshot.Response.STATUS_OK
        record("snapshot returns OK", ok, f"(status={getattr(first, 'status', 'none')})")
        if ok:
            age = (node.get_clock().now().nanoseconds - timestamp_ns(first.stamp)) * 1e-9
            record("pair age < 1s", 0.0 <= age < 1.0, f"(age={age:.3f}s)")
            received_ns = timestamp_ns(first.received_at)
            transport_lag = (
                received_ns - timestamp_ns(first.stamp)
            ) * 1e-9
            record(
                "snapshot receive time present",
                received_ns > 0,
                f"(transport_lag={transport_lag:.3f}s)",
            )
            skew = abs(timestamp_ns(first.color.header.stamp) -
                       timestamp_ns(first.depth.header.stamp)) * 1e-9
            record("color/depth skew <= 0.1s", skew <= 0.1, f"(skew={skew:.3f}s)")
            record("camera info present", first.color_info.width > 0 and
                   first.depth_info.width > 0)
            record("snapshot stamp is pair minimum", timestamp_ns(first.stamp) ==
                   min(timestamp_ns(first.color.header.stamp),
                       timestamp_ns(first.depth.header.stamp)))

        # 2. captured_after must return a pair newer than the boundary.
        boundary = node.get_clock().now().to_msg()
        request = GetCameraSnapshot.Request()
        request.captured_after = boundary
        request.wait_timeout_sec = 3.0
        newer = node.call(node.snapshot, request, timeout=8.0)
        is_newer = (newer is not None and
                    newer.status == GetCameraSnapshot.Response.STATUS_OK and
                    timestamp_ns(newer.stamp) > timestamp_ns(boundary))
        record("captured_after yields newer pair", is_newer,
               f"(status={getattr(newer, 'status', 'none')})")

        # 3. The transform embedded in a snapshot must agree with an explicit
        # lookup at that exact pair stamp.
        request = GetCameraSnapshot.Request()
        request.target_frames = [args.target_frame]
        stamped = node.call(node.snapshot, request)
        transform_ok = (stamped is not None and stamped.status ==
                        GetCameraSnapshot.Response.STATUS_OK and
                        stamped.transforms_ok and stamped.transforms_ok[0])
        if transform_ok:
            embedded = stamped.transforms[0].transform.translation
            record("snapshot transform succeeds", True,
                   f"({args.target_frame}<-{stamped.frame_id})")
            explicit = GetTransform.Request()
            explicit.target_frame = args.target_frame
            explicit.source_frame = stamped.frame_id
            explicit.lookup_time = stamped.stamp
            explicit.timeout_sec = 0.5
            looked_up = node.call(node.transform, explicit)
            actual_ok = (looked_up is not None and
                         looked_up.status == GetTransform.Response.STATUS_OK)
            if actual_ok:
                actual = looked_up.transform.transform.translation
                delta = math.sqrt((actual.x - embedded.x) ** 2 +
                                  (actual.y - embedded.y) ** 2 +
                                  (actual.z - embedded.z) ** 2)
            else:
                delta = math.inf
            record("get_transform matches snapshot transform", actual_ok and
                   delta <= 1e-4, f"(translation delta={delta:.6f}m)")
        else:
            record("snapshot transform succeeds", False,
                   f"(error={getattr(stamped, 'error_msg', 'no response')})")

        # 4. Native XYZRGB cloud and optional parity against old consumers.
        cloud_request = GetCameraPointCloud.Request()
        cloud_request.include_color = True
        cloud = node.call(node.cloud, cloud_request, timeout=30.0)
        cloud_ok = (cloud is not None and
                    cloud.status == GetCameraPointCloud.Response.STATUS_OK and
                    cloud.points.width > 1000 and cloud.points.height > 0)
        record("get_point_cloud returns a populated cloud", cloud_ok,
               f"(points={getattr(getattr(cloud, 'points', None), 'width', 0)})")
        if cloud_ok:
            record(
                "cloud receive time present",
                timestamp_ns(cloud.received_at) > 0,
            )
            field_names = {field.name for field in cloud.points.fields}
            record("colored cloud has XYZRGB fields",
                   {"x", "y", "z", "rgb"}.issubset(field_names),
                   f"(fields={sorted(field_names)})")

        if cloud_ok and args.legacy_pc_service:
            legacy = node.create_client(GetOrbbecPC, args.legacy_pc_service)
            legacy_request = GetOrbbecPC.Request()
            legacy_request.stride = 1
            legacy_request.include_color = True
            old = node.call(legacy, legacy_request, timeout=30.0)
            old_ok = old is not None and old.status == GetOrbbecPC.Response.STATUS_OK
            record("legacy point-cloud service reachable", old_ok)
            if old_ok:
                new_count, new_centroid = centroid(cloud.points)
                old_count, old_centroid = centroid(old.points)
                count_delta = abs(new_count - old_count) / max(old_count, 1)
                center_delta = math.dist(new_centroid, old_centroid)
                record("legacy parity: count within 15%", count_delta < 0.15,
                       f"(new={new_count}, old={old_count})")
                record("legacy parity: centroid within 5cm", center_delta < 0.05,
                       f"(delta={center_delta * 100:.1f}cm)")

        if cloud_ok and args.driver_cloud:
            driver = wait_for_driver_cloud(node, args.driver_cloud)
            driver_ok = driver is not None
            record("driver cloud topic reachable", driver_ok,
                   f"({args.driver_cloud})")
            if driver_ok:
                new_count, new_centroid = centroid(cloud.points)
                driver_count, driver_centroid = centroid(driver)
                count_delta = abs(new_count - driver_count) / max(driver_count, 1)
                center_delta = math.dist(new_centroid, driver_centroid)
                record("driver parity: count within 15%", count_delta < 0.15,
                       f"(new={new_count}, driver={driver_count})")
                record("driver parity: centroid within 5cm", center_delta < 0.05,
                       f"(delta={center_delta * 100:.1f}cm)")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    failed = [name for name, passed in RESULTS if not passed]
    print(f"[camera_server_t2] {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
