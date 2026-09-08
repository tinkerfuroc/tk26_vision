"""One-shot ROS image-topic frame grabber for the object_scan WebUI.

Invoked as a subprocess by server.py so rclpy's init/shutdown lifecycle stays
isolated from the long-running HTTP server. Subscribes to a color Image topic
with sensor-data QoS (BEST_EFFORT — matches camera publishers), waits for one
frame, writes it as JPEG, and prints "OK <path>" (or an ERROR line to stderr
with a non-zero exit).

    python ros_grab.py --topic /camera/color/image_raw --out photos/x.jpg
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/camera/color/image_raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()

    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        import cv2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: ROS deps unavailable: {exc}", file=sys.stderr)
        return 3

    rclpy.init()
    node = rclpy.create_node("object_scan_ros_grab")
    bridge = CvBridge()
    holder = {"msg": None}

    def cb(msg):
        if holder["msg"] is None:
            holder["msg"] = msg

    node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)
    t0 = time.time()
    while holder["msg"] is None and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    if holder["msg"] is None:
        print(f"ERROR: no frame on {args.topic} within {args.timeout:.0f}s "
              "(is the camera launched and publishing?)", file=sys.stderr)
        return 2

    try:
        img = bridge.imgmsg_to_cv2(holder["msg"], desired_encoding="bgr8")
        cv2.imwrite(args.out, img)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: convert/write failed: {exc}", file=sys.stderr)
        return 4

    print(f"OK {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
