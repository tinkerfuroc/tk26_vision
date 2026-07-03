"""Stdlib http.server backend for the object_scan tuning WebUI.

No third-party web framework — runs under `.venv-vision-main` (needs only
openai + python-dotenv, already installed). Serves the single-page UI, stores
captured/uploaded photos under ./photos, and runs the batched VLM scan +
batch-size sweep via scan_core.

    python server.py [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import scan_core

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(HERE, "photos")
INDEX_HTML = os.path.join(HERE, "index.html")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_TOPIC = re.compile(r"^[A-Za-z0-9_/]+$")

# Named presets shown in the UI; any valid topic can be typed as "custom".
ROS_TOPIC_PRESETS = [
    {"label": "orbbec (head) — /camera/color/image_raw",
     "topic": "/camera/color/image_raw"},
    {"label": "realsense (arm) — /camera/xarm_camera/color/image_raw",
     "topic": "/camera/xarm_camera/color/image_raw"},
]

os.makedirs(PHOTOS_DIR, exist_ok=True)


class RosCamera:
    """Lazy in-process ROS image subscriber shared across requests.

    Spins one rclpy node in a background thread; subscribes to color Image
    topics on demand and keeps the latest JPEG-encoded frame per topic. Used
    for both the live MJPEG preview and instant capture (the previewed frame
    is already cached). Degrades gracefully: if rclpy/ROS is unavailable the
    webcam + upload + scan paths keep working.
    """

    def __init__(self):
        self.available = None      # None=unknown, True/False after first ensure
        self.err = ""
        self._node = None
        self._bridge = None
        self._exec = None
        self._subs = {}            # topic -> subscription
        self._frames = {}          # topic -> (jpeg_bytes, stamp)
        self._lock = threading.Lock()

    def _ensure(self) -> bool:
        with self._lock:
            if self.available is not None:
                return self.available
            try:
                import rclpy
                from cv_bridge import CvBridge
                from rclpy.executors import SingleThreadedExecutor
                if not rclpy.ok():
                    rclpy.init(args=None)
                self._node = rclpy.create_node("object_scan_webui_cam")
                self._bridge = CvBridge()
                self._exec = SingleThreadedExecutor()
                self._exec.add_node(self._node)
                threading.Thread(target=self._spin, daemon=True).start()
                self.available = True
            except Exception as exc:   # noqa: BLE001
                self.err = str(exc)
                self.available = False
            return self.available

    def _spin(self):
        try:
            self._exec.spin()
        except Exception:  # noqa: BLE001
            pass

    def subscribe(self, topic: str) -> bool:
        if not self._ensure():
            return False
        with self._lock:
            if topic in self._subs:
                return True
        try:
            from sensor_msgs.msg import Image
            from rclpy.qos import qos_profile_sensor_data
            sub = self._node.create_subscription(
                Image, topic, lambda m, t=topic: self._cb(m, t),
                qos_profile_sensor_data,
            )
            with self._lock:
                self._subs[topic] = sub
            return True
        except Exception as exc:   # noqa: BLE001
            self.err = str(exc)
            return False

    def _cb(self, msg, topic):
        try:
            import cv2
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self._lock:
                    self._frames[topic] = (buf.tobytes(), time.time())
        except Exception:  # noqa: BLE001
            pass

    def latest(self, topic: str):
        with self._lock:
            fr = self._frames.get(topic)
        return fr[0] if fr else None

    def grab(self, topic: str, timeout: float = 8.0):
        """Subscribe (if needed) and return the next/cached frame, or None."""
        if not self.subscribe(topic):
            return None
        t0 = time.time()
        while time.time() - t0 < timeout:
            d = self.latest(topic)
            if d is not None:
                return d
            time.sleep(0.05)
        return None


ROS_CAM = RosCamera()


def _data_url_to_bytes(data_url: str):
    """Return (bytes, ext) from a data: URL (jpeg/png)."""
    m = re.match(r"data:image/(png|jpe?g);base64,(.+)$", data_url, re.DOTALL)
    if not m:
        raise ValueError("expected data:image/{png,jpeg};base64,...")
    import base64
    ext = "png" if m.group(1) == "png" else "jpg"
    return base64.b64decode(m.group(2)), ext


class Handler(BaseHTTPRequestHandler):
    server_version = "ObjectScanWebUI/1.0"

    # -- helpers ----------------------------------------------------------
    def _send(self, code, body=b"", ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def _photo_path(self, name):
        if not _SAFE_NAME.match(name or ""):
            return None
        p = os.path.join(PHOTOS_DIR, name)
        return p if os.path.commonpath([PHOTOS_DIR, os.path.abspath(p)]) == PHOTOS_DIR else None

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    def _ros_stream(self, query: str):
        """Stream a ROS color topic as multipart MJPEG (live preview)."""
        topic = (parse_qs(query).get("topic", ["/camera/color/image_raw"])[0])
        if not _SAFE_TOPIC.match(topic):
            self._send(400, {"error": f"bad topic {topic!r}"})
            return
        if not ROS_CAM.subscribe(topic):
            self._send(503, {"error": f"ROS camera unavailable: {ROS_CAM.err}"})
            return
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        # brief wait for the first frame so the <img> doesn't error out
        deadline = time.time() + 5.0
        while time.time() < deadline and ROS_CAM.latest(topic) is None:
            time.sleep(0.05)
        try:
            while True:
                frame = ROS_CAM.latest(topic)
                if frame is not None and frame is not last:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode()
                        + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
                    last = frame
                time.sleep(0.05)   # ~20 fps ceiling
        except (BrokenPipeError, ConnectionResetError):
            pass  # client closed the preview <img>; end the thread

    # -- routing ----------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            try:
                with open(INDEX_HTML, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, {"error": "index.html missing"})
            return
        if path == "/api/vocabulary":
            try:
                self._send(200, {"vocabulary": scan_core.parse_vocabulary()})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"vocabulary load failed: {exc}"})
            return
        if path == "/api/photos":
            names = sorted(
                (n for n in os.listdir(PHOTOS_DIR)
                 if n.lower().endswith((".jpg", ".jpeg", ".png"))),
                reverse=True,
            )
            self._send(200, {"photos": names})
            return
        if path == "/api/ros_topics":
            self._send(200, {"presets": ROS_TOPIC_PRESETS})
            return
        if path == "/api/ros_stream":
            self._ros_stream(urlparse(self.path).query)
            return
        if path.startswith("/photos/"):
            name = path[len("/photos/"):]
            p = self._photo_path(name)
            if not p or not os.path.isfile(p):
                self._send(404, {"error": "not found"})
                return
            ctype = "image/png" if name.lower().endswith(".png") else "image/jpeg"
            with open(p, "rb") as f:
                self._send(200, f.read(), ctype)
            return
        self._send(404, {"error": f"no route {path}"})

    def do_HEAD(self):
        self.do_GET()

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/photos/"):
            name = path[len("/api/photos/"):]
            p = self._photo_path(name)
            if p and os.path.isfile(p):
                os.remove(p)
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
            return
        self._send(404, {"error": f"no route {path}"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"bad json: {exc}"})
            return

        if path == "/api/photos":
            try:
                raw, ext = _data_url_to_bytes(body["image"])
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": f"bad image: {exc}"})
                return
            ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            src = "cam" if body.get("source") == "cam" else "up"
            name = f"photo_{ts}_{src}.{ext}"
            with open(os.path.join(PHOTOS_DIR, name), "wb") as f:
                f.write(raw)
            self._send(200, {"name": name})
            return

        if path == "/api/ros_capture":
            topic = (body.get("topic") or "/camera/color/image_raw").strip()
            if not _SAFE_TOPIC.match(topic):
                self._send(400, {"error": f"bad topic {topic!r}"})
                return
            frame = ROS_CAM.grab(topic, timeout=float(body.get("timeout", 8.0)))
            if frame is None:
                self._send(502, {"error":
                    f"no frame on {topic} within timeout — is the camera "
                    f"launched and ROS sourced? {ROS_CAM.err}".strip()})
                return
            ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            name = f"photo_{ts}_ros.jpg"
            with open(os.path.join(PHOTOS_DIR, name), "wb") as f:
                f.write(frame)
            self._send(200, {"name": name})
            return

        if path in ("/api/scan", "/api/sweep"):
            name = body.get("photo", "")
            p = self._photo_path(name)
            if not p or not os.path.isfile(p):
                self._send(400, {"error": f"unknown photo {name!r}"})
                return
            try:
                vocab = body.get("vocabulary") or scan_core.parse_vocabulary()
                url = scan_core.path_to_data_url(p)
                use_qwen = bool(body.get("use_qwen_fallback", True))
                max_workers = int(body.get("max_workers", 0))  # 0 = all batches parallel
                if path == "/api/scan":
                    res = scan_core.scan_image(
                        url, vocab,
                        batch_size=int(body.get("batch_size", 8)),
                        max_workers=max_workers, use_qwen_fallback=use_qwen,
                        log=lambda m: print(f"[scan] {m}", flush=True),
                    )
                    self._send(200, {"photo": name, "result": res.to_dict()})
                else:
                    sizes = [int(x) for x in body.get("batch_sizes", [4, 8, 16])]
                    rows = scan_core.sweep_batch_sizes(
                        url, vocab, sizes,
                        repeats=int(body.get("repeats", 1)),
                        truth=body.get("truth") or None,
                        max_workers=max_workers, use_qwen_fallback=use_qwen,
                        log=lambda m: print(f"[sweep] {m}", flush=True),
                    )
                    self._send(200, {"photo": name, "sweep": rows})
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._send(500, {"error": f"scan failed: {exc}"})
            return

        self._send(404, {"error": f"no route {path}"})


def main():
    scan_core.load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    has_or = bool(scan_core.openrouter_api_key())
    has_qwen = bool(scan_core.dashscope_api_key())
    print(f"OPENROUTER_API_KEY: {'set' if has_or else 'MISSING'} | "
          f"DASHSCOPE/DASHCOPE_API_KEY: {'set' if has_qwen else 'MISSING'}")
    if not has_or:
        print("WARNING: Gemini calls will fail without OPENROUTER_API_KEY.")
    try:
        n = len(scan_core.parse_vocabulary())
        print(f"vocabulary: {n} classes from PickAndPlace/constants.json")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not load vocabulary: {exc}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  object_scan WebUI  ->  http://{args.host}:{args.port}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
