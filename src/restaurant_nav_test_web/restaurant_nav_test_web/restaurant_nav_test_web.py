"""ROS node + thread-safe bridge for the restaurant nav-test dashboard.
Mirrors vision_track/track_web.py. Subscribes the BT status topic + a color
camera topic (MJPEG), computes live robot->target distance from TF, derives
graph-based readiness, and drives the allowlisted ProcessManager.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

from restaurant_nav_test_web.process_manager import ProcessManager, load_registry
from restaurant_nav_test_web.restaurant_nav_test_web_app import create_app


def _share_config_path():
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("restaurant_nav_test_web")) / "config" / "processes.yaml"


class RestaurantNavTestWebNode(Node):
    def __init__(self):
        super().__init__("restaurant_nav_test_web")
        self.declare_parameter("bind", "0.0.0.0")
        self.declare_parameter("port", 8768)
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("status_topic", "/restaurant_nav_test/status")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("workspace_root", "/home/tinker/tk25_ws")
        self.bind_host = str(self.get_parameter("bind").value)
        self.bind_port = int(self.get_parameter("port").value)
        ws = str(self.get_parameter("workspace_root").value)

        self._lock = threading.RLock()  # reentrant: snapshot/latest_state call
        # _distance_to_target()/_readiness() which re-acquire the same lock.
        self._state = None
        self._state_seq = 0
        self._state_ts = 0.0
        self._jpeg = None
        self._jpeg_seq = 0

        try:
            registry, groups, stagger = load_registry(str(_share_config_path()))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"processes.yaml load failed: {exc}; empty allowlist")
            registry, groups, stagger = {}, {}, 1.5
        registry = {k: [a.replace("<WS>", ws) for a in argv] for k, argv in registry.items()}
        self.proc = ProcessManager(registry=registry, groups=groups, stagger_sec=stagger)

        cb = ReentrantCallbackGroup()
        self.create_subscription(
            String, str(self.get_parameter("status_topic").value),
            self._on_status, 10, callback_group=cb)
        self.create_subscription(
            Image, str(self.get_parameter("camera_topic").value),
            self._on_image, 1, callback_group=cb)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _on_status(self, msg: String):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._state = state
            self._state_seq += 1
            self._state_ts = time.time()

    def _on_image(self, msg: Image):
        if cv2 is None:
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception:  # noqa: BLE001
            return
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._jpeg_seq += 1

    def _readiness(self) -> dict:
        topics = dict(self.get_topic_names_and_types())
        services = dict(self.get_service_names_and_types())
        cam = str(self.get_parameter("camera_topic").value)
        with self._lock:
            cam_fresh = self._jpeg is not None
        return {
            "camera": cam in topics and cam_fresh,
            "pan_tilt": "/pan_tilt_controller/state" in topics,
            "waving": "/detect_waving_persons" in services,
            "goto": "/go_to_approach/_action/send_goal" in services,
        }

    def _distance_to_target(self):
        with self._lock:
            state = self._state
        if not state or not state.get("target"):
            return None
        tx, ty = state["target"]["x"], state["target"]["y"]
        try:
            t = self._tf_buffer.lookup_transform(
                str(self.get_parameter("target_frame").value),
                str(self.get_parameter("robot_frame").value),
                rclpy.time.Time())
            rx, ry = t.transform.translation.x, t.transform.translation.y
            return round(math.hypot(tx - rx, ty - ry), 2)
        except Exception:  # noqa: BLE001
            return None

    def snapshot(self):
        with self._lock:
            state, ts = self._state, self._state_ts
        return {
            "state": state,
            "state_age_s": (round(time.time() - ts, 1) if ts else None),
            "distance_m": self._distance_to_target(),
            "readiness": self._readiness(),
            "proc": self.proc.status_all(),
        }

    def latest_state(self):
        with self._lock:
            if self._state is None:
                return self._state_seq, None
            merged = dict(self._state)
            merged["distance_m"] = self._distance_to_target()
            return self._state_seq, merged

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg_seq, self._jpeg

    def start_test(self, mock: bool = False):
        if mock:
            os.environ["BT_MOCK_MODE"] = "true"
        else:
            os.environ.pop("BT_MOCK_MODE", None)
        return self.proc.start("test_bt")

    def stop_test(self):
        return self.proc.stop("test_bt")

    def proc_status(self):
        return self.proc.status_all()

    def proc_start(self, name):
        return self.proc.start(name)

    def proc_stop(self, name):
        return self.proc.stop(name)

    def proc_group_start(self, group):
        return self.proc.start_group(group)

    def proc_group_stop(self, group):
        return self.proc.stop_group(group)


def main():
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    rclpy.init()
    node = RestaurantNavTestWebNode()
    try:
        from ament_index_python.packages import get_package_share_directory
        webui_dir = Path(get_package_share_directory("restaurant_nav_test_web")) / "webui"
    except Exception:  # noqa: BLE001
        webui_dir = Path(__file__).resolve().parents[1] / "webui"
    app = create_app(node, webui_dir=webui_dir)

    import uvicorn
    config = uvicorn.Config(app, host=node.bind_host, port=node.bind_port,
                            log_level="info", access_log=False, loop="asyncio")
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve())

    thread = threading.Thread(target=_serve, daemon=True, name="uvicorn")
    thread.start()
    node.get_logger().info(
        f"restaurant_nav_test_web on http://{node.bind_host}:{node.bind_port}")

    executor = MultiThreadedExecutor(num_threads=4)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        try:
            node.proc.shutdown_all()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.try_shutdown()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
