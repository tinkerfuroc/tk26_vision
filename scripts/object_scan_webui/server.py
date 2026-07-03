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
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import scan_core

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(HERE, "photos")
INDEX_HTML = os.path.join(HERE, "index.html")
ROS_GRAB = os.path.join(HERE, "ros_grab.py")
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
            ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            name = f"photo_{ts}_ros.jpg"
            out = os.path.join(PHOTOS_DIR, name)
            try:
                proc = subprocess.run(
                    [sys.executable, ROS_GRAB, "--topic", topic,
                     "--out", out, "--timeout", str(float(body.get("timeout", 8.0)))],
                    capture_output=True, text=True,
                    timeout=float(body.get("timeout", 8.0)) + 15.0,
                )
            except subprocess.TimeoutExpired:
                self._send(504, {"error": f"ROS grab timed out on {topic}"})
                return
            if proc.returncode == 0 and os.path.isfile(out):
                self._send(200, {"name": name})
            else:
                msg = (proc.stderr or proc.stdout or "unknown error").strip().splitlines()
                self._send(502, {"error": msg[-1] if msg else "ROS grab failed",
                                 "detail": (proc.stderr or "")[-800:]})
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
                max_workers = int(body.get("max_workers", 4))
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
                        url, vocab, sizes, max_workers=max_workers,
                        use_qwen_fallback=use_qwen,
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
