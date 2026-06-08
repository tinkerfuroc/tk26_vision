"""track_web — live tracking dashboard + active-reID test bench (ROS side).

Run:
    ros2 run vision_track track_web --ros-args -p bind:=0.0.0.0 -p port:=8766

Bridges the person tracker's debug topics + TrackPerson action +
ReseedTarget/DetectWaving services to the FastAPI app in track_web_app.py.
Threading: rclpy.spin in the main thread, uvicorn in a daemon thread, all
shared state behind self._lock (the calib_web model). HTTP handlers run on
uvicorn worker threads and may block politely (wait_for_*/future polls) —
callbacks keep spinning on the executor in the main thread.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from tinker_vision_msgs_26.action import TrackPerson
from tinker_vision_msgs_26.srv import DetectWaving, ReseedTarget

from vision_track.track_web_app import create_app

_STALE_S = 1.0


def _resolve_webui_dir() -> Path:
    """Installed share/vision_track/webui, falling back to the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory("vision_track")) / "webui"
        if p.exists():
            return p
    except Exception:
        pass
    # Source-tree fallback (running uninstalled). From
    # src/vision_track/vision_track/track_web.py: parents[0] is the package
    # module dir, parents[1] is the ROS package root (next to setup.py /
    # package.xml) where webui/ lives. Mirror calib_web's parent.parent.
    return Path(__file__).resolve().parents[1] / "webui"


class TrackWebNode(Node):
    """ROS bridge implementing the track_web_app bridge contract."""

    def __init__(self):
        super().__init__("track_web")
        self.declare_parameter("bind", "127.0.0.1")
        self.declare_parameter("port", 8766)
        self.declare_parameter("tracker_node_name", "person_track_node")
        self.declare_parameter("waving_service", "detect_waving_persons")
        self.bind_host = str(self.get_parameter("bind").value)
        self.bind_port = int(self.get_parameter("port").value)
        tracker = str(self.get_parameter("tracker_node_name").value)
        waving = str(self.get_parameter("waving_service").value)
        # Rosbag recording: capture the tracker's INPUT topics (so a session can
        # be replayed offline into the tracker) plus its decisions for reference.
        self.declare_parameter("record_dir", "~/tk25_ws/rosbags")
        self.declare_parameter("record_topics", [
            "/camera/color/image_raw", "/camera/depth/image_raw",
            "/camera/color/camera_info", f"/{tracker}/debug_state",
            "/target_points"])
        self.record_dir = os.path.expanduser(
            str(self.get_parameter("record_dir").value))
        self.record_topics = [t for t in self.get_parameter("record_topics").value
                              if t]

        self._lock = threading.Lock()
        self._state = None          # latest debug_state dict
        self._state_seq = 0
        self._state_ts = 0.0
        self._gallery = None        # latest debug_gallery dict
        self._jpeg = None           # latest annotated frame as JPEG bytes
        self._jpeg_seq = 0
        self._goal_handle = None    # our bench goal (None = not held by us)
        self._rec_proc = None       # `ros2 bag record` subprocess (None = idle)
        self._rec_path = None       # output dir of the active/last recording

        cb = ReentrantCallbackGroup()
        self.create_subscription(
            String, f"/{tracker}/debug_state", self._on_state, 10,
            callback_group=cb)
        gallery_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, f"/{tracker}/debug_gallery", self._on_gallery, gallery_qos,
            callback_group=cb)
        self.create_subscription(
            Image, f"/{tracker}/debug_image", self._on_image, 1,
            callback_group=cb)

        self._action = ActionClient(self, TrackPerson, "track_person",
                                    callback_group=cb)
        self._reseed_cli = self.create_client(
            ReseedTarget, f"/{tracker}/reseed_target", callback_group=cb)
        self._wave_cli = self.create_client(DetectWaving, waving,
                                            callback_group=cb)
        self.get_logger().info(
            f"track_web bridging tracker '{tracker}', waving '{waving}'")

    # ---- subscription callbacks -------------------------------------------
    def _on_state(self, msg: String):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._state = state
            self._state_seq += 1
            self._state_ts = time.time()

    def _on_gallery(self, msg: String):
        try:
            gal = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._gallery = gal

    def _on_image(self, msg: Image):
        # bgr8 on the wire; encode once here, serve many times.
        if msg.encoding != "bgr8" or msg.step != msg.width * 3:
            # Fail loud rather than silently garbling a padded/foreign image.
            self.get_logger().warn(
                f"debug_image unexpected layout (encoding={msg.encoding!r}, "
                f"step={msg.step}, width={msg.width}); frame dropped",
                throttle_duration_sec=5.0)
            return
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as exc:
            self.get_logger().warn(f"debug_image decode failed: {exc}",
                                   throttle_duration_sec=5.0)
            return
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()
                self._jpeg_seq += 1

    # ---- bridge contract ---------------------------------------------------
    def snapshot(self):
        with self._lock:
            age = time.time() - self._state_ts if self._state is not None else None
            held = self._goal_handle is not None
            observer = (not held and self._state is not None
                        and age is not None and age < _STALE_S)
            rec_active = (self._rec_proc is not None
                          and self._rec_proc.poll() is None)
            return {"state": self._state, "state_age_s": age,
                    "goal": {"held": held, "observer": observer},
                    "recording": {"active": rec_active, "path": self._rec_path},
                    "gallery_version": (self._gallery or {}).get("version", -1)}

    def latest_state(self):
        with self._lock:
            return self._state_seq, self._state

    def latest_gallery(self):
        with self._lock:
            return self._gallery

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg_seq, self._jpeg

    def start_goal(self):
        with self._lock:
            if self._goal_handle is not None:
                return {"ok": False, "message": "bench goal already running"}
        if not self._action.wait_for_server(timeout_sec=2.0):
            return {"ok": False,
                    "message": "track_person action server unavailable"}
        goal = TrackPerson.Goal()  # all image-return flags default False
        future = self._action.send_goal_async(goal)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            return {"ok": False, "message": "goal send timed out"}
        handle = future.result()
        if not handle.accepted:
            return {"ok": False,
                    "message": "goal REJECTED (another client is tracking?)"}
        with self._lock:
            self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_goal_done)
        return {"ok": True, "message": "tracking goal accepted"}

    def _on_goal_done(self, _future):
        with self._lock:
            self._goal_handle = None

    def stop_goal(self):
        with self._lock:
            handle = self._goal_handle
            # Release the slot now rather than waiting for _on_goal_done: if the
            # action server died mid-goal the result callback never fires and
            # `held` would latch True forever, blocking every future start.
            # _on_goal_done only writes None, so a late callback is harmless.
            self._goal_handle = None
        if handle is None:
            return {"ok": False, "message": "no bench goal to stop"}
        handle.cancel_goal_async()
        return {"ok": True, "message": "cancel requested"}

    def _call(self, client, request, timeout=10.0, name="service"):
        if not client.wait_for_service(timeout_sec=2.0):
            return None, f"{name} unavailable"
        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not future.done():
            return None, f"{name} timed out after {timeout:.0f}s"
        return future.result(), None

    def reseed(self, bbox):
        req = ReseedTarget.Request()
        req.bbox.x_offset = max(0, int(bbox[0]))
        req.bbox.y_offset = max(0, int(bbox[1]))
        req.bbox.width = max(0, int(bbox[2] - bbox[0]))
        req.bbox.height = max(0, int(bbox[3] - bbox[1]))
        req.frame_id = ""
        resp, err = self._call(self._reseed_cli, req, name="reseed_target")
        if err:
            return {"success": False, "target_track_id": -1, "message": err}
        return {"success": bool(resp.success),
                "target_track_id": int(resp.target_track_id),
                "message": str(resp.message)}

    def wave(self):
        # Default request (min_waving_persons=0) deliberately runs the fast
        # MediaPipe heuristic only — the VLM fallback is a server-side knob the
        # bench doesn't need; 30s timeout still covers it if enabled there.
        resp, err = self._call(self._wave_cli, DetectWaving.Request(),
                               timeout=30.0, name="detect_waving_persons")
        if err:
            return {"status": -1, "boxes": [], "points": [], "error": err}
        boxes = [[int(b.x_offset), int(b.y_offset),
                  int(b.x_offset + b.width), int(b.y_offset + b.height)]
                 for b in resp.waving_boxes]
        points = [[float(p.point.x), float(p.point.y), float(p.point.z)]
                  for p in resp.waving_persons]
        out = {"status": int(resp.status), "boxes": boxes, "points": points}
        # Wave-to-resume: an UNAMBIGUOUS single waver auto-reseeds (no click) so a
        # raise-hand resumes tracking on its own. Multiple wavers stay manual
        # (the operator picks the box) to keep the re-lock precise.
        if int(resp.status) == 0 and len(boxes) == 1:
            out["reseed"] = self.reseed(boxes[0])
            out["auto_reseeded"] = bool(out["reseed"].get("success"))
        return out

    def record_start(self):
        with self._lock:
            if self._rec_proc is not None and self._rec_proc.poll() is None:
                return {"ok": False, "message": "already recording",
                        "path": self._rec_path}
        if not self.record_topics:
            return {"ok": False, "message": "no record_topics configured",
                    "path": ""}
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        path = os.path.join(self.record_dir, f"track_{stamp}")
        cmd = ["ros2", "bag", "record", "-o", path, *self.record_topics]
        try:
            os.makedirs(self.record_dir, exist_ok=True)
            # start_new_session: a SIGINT we send on stop reaches the recorder,
            # and a Ctrl-C in the launching terminal won't tear the bag mid-write.
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        except Exception as exc:  # never crash the node on a spawn failure
            self.get_logger().error(f"rosbag record failed to start: {exc}")
            return {"ok": False, "message": f"failed to start: {exc}", "path": ""}
        with self._lock:
            self._rec_proc = proc
            self._rec_path = path
        self.get_logger().info(
            f"rosbag recording -> {path}  topics={self.record_topics}")
        return {"ok": True, "message": f"recording to {path}", "path": path,
                "topics": self.record_topics}

    def record_stop(self):
        with self._lock:
            proc = self._rec_proc
            path = self._rec_path
            self._rec_proc = None
        if proc is None or proc.poll() is not None:
            return {"ok": False, "message": "not recording", "path": path or ""}
        # SIGINT lets `ros2 bag record` finalize metadata/index cleanly; escalate
        # to kill if it doesn't exit promptly.
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        self.get_logger().info(f"rosbag recording stopped -> {path}")
        return {"ok": True, "message": f"saved {path}", "path": path or ""}


def main():
    # Mirror calib_web: avoid the SHM-discovery stall on a live robot.
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    os.environ.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)

    rclpy.init()
    node = TrackWebNode()
    webui_dir = _resolve_webui_dir()
    node.get_logger().info(f"web UI static dir: {webui_dir}")
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
        f"track_web listening on http://{node.bind_host}:{node.bind_port}")

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        node.record_stop()  # finalize any in-progress bag before teardown
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
