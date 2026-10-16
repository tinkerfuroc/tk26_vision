"""Small ROS-to-HTTP validation dashboard; no external web dependencies."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tinker_vision_msgs_26.msg import PanTiltCommand, PanTiltState
from tinker_vision_msgs_26.srv import DetectRgbScene


class SceneWeb(Node):
    def __init__(self):
        super().__init__('scene_detection_web')
        self.host = self.declare_parameter('host', '127.0.0.1').value
        self.port = self.declare_parameter('port', 8088).value
        self.topic = self.declare_parameter('rgb_topic', '/camera/color/image_raw').value
        self.target = self.declare_parameter('target_frame', '').value
        self.timeout = float(self.declare_parameter('service_timeout_s', 60.0).value)
        if not 1 <= self.timeout <= 300:
            raise ValueError('service_timeout_s must be in 1..300')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.jpeg = None
        self.received = 0.0
        self.stamp = None
        self.busy = set()
        self.pan_tilt = None
        self.pan_tilt_received = 0.0
        self.pan_tilt_pub = self.create_publisher(
            PanTiltCommand, '/pan_tilt_controller/cmd', 10)
        self.create_subscription(
            PanTiltState, '/pan_tilt_controller/state', self.on_pan_tilt, 10)
        self.detectors = {task: self.create_client(
            DetectRgbScene, self.declare_parameter(task + '_service',
                                                  '/' + task + '_detection/detect').value)
                        for task in ('behaviour', 'litter')}
        self.create_subscription(Image, self.topic, self.on_image, qos_profile_sensor_data)

    def on_image(self, msg):
        if time.monotonic() - self.received < 0.1:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
            ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.jpeg = data.tobytes()
                    self.received = time.monotonic()
                    self.stamp = {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec,
                                  'frame_id': msg.header.frame_id}
        except Exception as exc:
            self.get_logger().warning('Preview decode failed: ' + str(exc))

    def on_pan_tilt(self, msg):
        with self.lock:
            self.pan_tilt = {
                'pan_deg': math.degrees(msg.pan_rad),
                'tilt_deg': math.degrees(msg.tilt_rad),
                'connected': msg.connected, 'feedback_ok': msg.feedback_ok}
            self.pan_tilt_received = time.monotonic()

    def pan_tilt_state(self):
        # Caller holds self.lock.
        result = dict(self.pan_tilt or {})
        age = time.monotonic() - self.pan_tilt_received if self.pan_tilt else None
        result['ready'] = bool(
            age is not None and age < 1.0 and result.get('connected')
            and result.get('feedback_ok')
            and all(math.isfinite(result[k]) for k in ('pan_deg', 'tilt_deg'))
            and self.pan_tilt_pub.get_subscription_count() > 0)
        result['age_s'] = age
        for key in ('pan_deg', 'tilt_deg'):
            if key in result and not math.isfinite(result[key]):
                result[key] = None
        return result

    def move_pan_tilt(self, data):
        if not isinstance(data, dict):
            return 400, {'error': '请提供水平和俯仰角度'}
        values = []
        for key, low, high in (('pan_deg', -180, 180), ('tilt_deg', -30, 90)):
            value = data.get(key)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not low <= value <= high):
                return 400, {'error': f'{key} 必须是 {low} 到 {high} 之间的有限数值'}
            values.append(float(value))
        with self.lock:
            if not self.pan_tilt_state()['ready']:
                return 503, {'error': '云台未连接或反馈已过期，未发送动作'}
            msg = PanTiltCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.mode = PanTiltCommand.ABSOLUTE
            msg.pan_rad, msg.tilt_rad = map(math.radians, values)
            msg.speed_raw = 120
            msg.accel_raw = 20
            self.pan_tilt_pub.publish(msg)
        return 200, {'sent': True, 'pan_deg': values[0], 'tilt_deg': values[1]}

    def state(self):
        with self.lock:
            age = time.monotonic() - self.received if self.jpeg else None
            return {'rgb_topic': self.topic, 'image_age_s': age, 'stamp': self.stamp,
                    'camera_ready': age is not None and age < 2,
                    'services': {k: c.service_is_ready() for k, c in self.detectors.items()},
                    'busy': sorted(self.busy), 'pan_tilt': self.pan_tilt_state()}

    def detect(self, task):
        client = self.detectors[task]
        with self.lock:
            if task in self.busy:
                return 409, {'error': '该检测仍在进行，请等待完成'}
            if not client.service_is_ready():
                return 503, {'error': 'ROS 检测服务未启动或尚未发现'}
            self.busy.add(task)
        started = time.monotonic()
        event = threading.Event()
        try:
            request = DetectRgbScene.Request()
            request.target_frame = self.target
            future = client.call_async(request)
        except Exception:
            with self.lock:
                self.busy.discard(task)
            raise

        def done(_):
            with self.lock:
                self.busy.discard(task)
            event.set()

        future.add_done_callback(done)
        # A timed-out HTTP request does not cancel the actual VLM inference.
        # Keep the per-task gate closed until its ROS future really completes.
        if not event.wait(self.timeout):
            return 504, {'error': '等待 ROS 服务超时；后台可能仍在检测，请勿重复提交'}
        response = future.result()
        result = {'task': task, 'status': response.status, 'error': response.error_msg,
                  'detected': response.detected, 'vlm_elapsed_s': response.vlm_elapsed_s,
                  'service_elapsed_s': time.monotonic() - started,
                  'provider': response.vlm_provider, 'model': response.vlm_model,
                  'stamp': {'sec': response.header.stamp.sec,
                            'nanosec': response.header.stamp.nanosec,
                            'frame_id': response.header.frame_id},
                  'log_json': response.log_json, 'observations': []}
        for row in response.observations:
            p = row.position
            result['observations'].append({
                'labels': list(row.labels), 'description': row.description,
                'bbox': [row.bbox.x_offset, row.bbox.y_offset,
                         row.bbox.x_offset + row.bbox.width, row.bbox.y_offset + row.bbox.height],
                'position_valid': row.position_valid, 'position_error': row.position_error,
                'position': {'frame_id': p.header.frame_id,
                             'xyz': [p.point.x, p.point.y, p.point.z]} if row.position_valid else None})
        for name in ('rgb_image', 'annotated_image'):
            msg = getattr(response, name)
            if msg.width and msg.height:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                ok, data = cv2.imencode('.jpg', frame)
                if ok:
                    result[name] = 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')
        return 200, result


def make_handler(node):
    page = Path(__file__).with_name('scene_web.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, mime='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                self.send(200, page, 'text/html; charset=utf-8')
            elif path == '/api/state':
                self.send(200, node.state())
            elif path == '/api/frame.jpg':
                with node.lock:
                    data = node.jpeg if time.monotonic() - node.received < 2 else None
                self.send(200, data, 'image/jpeg') if data else self.send(503, {'error': '无新鲜 RGB 图像'})
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            # Require same-origin JSON requests; do not expose a paid API via cross-site forms.
            origin = self.headers.get('Origin')
            if (origin and urlsplit(origin).netloc != self.headers.get('Host')) or self.headers.get('Content-Type') != 'application/json':
                self.send(403, {'error': 'Only same-origin JSON requests are allowed'})
                return
            if self.path == '/api/pan_tilt':
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 1024:
                        raise ValueError('invalid body size')
                    data = json.loads(self.rfile.read(size))
                except (ValueError, UnicodeError):
                    self.send(400, {'error': '无效的角度请求'})
                    return
                try:
                    status, result = node.move_pan_tilt(data)
                    self.send(status, result)
                except Exception as exc:
                    node.get_logger().error('Pan-tilt command failed: ' + type(exc).__name__)
                    self.send(500, {'error': '云台命令发送失败'})
                return
            task = {'/api/detect/behaviour': 'behaviour', '/api/detect/litter': 'litter'}.get(self.path)
            if task is None:
                self.send(404, {'error': 'Not found'})
                return
            try:
                status, result = node.detect(task)
                self.send(status, result)
            except Exception as exc:
                node.get_logger().error('Web detection failed: ' + type(exc).__name__)
                self.send(500, {'error': '服务调用失败：' + type(exc).__name__})

    return Handler


def main(args=None):
    rclpy.init(args=args)
    node = SceneWeb()
    server = ThreadingHTTPServer((node.host, node.port), make_handler(node))
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    node.get_logger().info(f'Scene validation: http://{node.host}:{node.port}')
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=5)
        node.destroy_node()
