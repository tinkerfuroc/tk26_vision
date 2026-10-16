"""ROS-host tests of HTTP routes, result serialization and in-flight gates."""
import json
import math
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

rclpy = pytest.importorskip('rclpy')
from rclpy.task import Future
from tinker_vision_msgs_26.srv import DetectRgbScene
from tinker_vision_msgs_26.msg import PanTiltState, PanTiltCommand
from tk_vision_specialized.scene_detection_web import SceneWeb, make_handler


@pytest.fixture
def node():
    rclpy.init(args=[])
    n = SceneWeb()
    yield n
    n.destroy_node()
    rclpy.shutdown()


class Client:
    def __init__(self):
        self.future = Future()

    def service_is_ready(self):
        return True

    def call_async(self, request):
        self.request = request
        return self.future


class Publisher:
    def __init__(self):
        self.messages = []
        self.subscribers = 1

    def get_subscription_count(self):
        return self.subscribers

    def publish(self, msg):
        self.messages.append(msg)


def test_pan_tilt_validates_bounds_and_requires_live_feedback(node):
    pub = Publisher()
    node.pan_tilt_pub = pub
    target = {'pan_deg': 12.5, 'tilt_deg': 45.0}
    assert node.move_pan_tilt(target)[0] == 503
    state = PanTiltState(connected=True, feedback_ok=True,
                         pan_rad=0.0, tilt_rad=math.pi / 4)
    node.on_pan_tilt(state)
    assert node.state()['pan_tilt']['tilt_deg'] == pytest.approx(45)
    for invalid in (None, {}, {'pan_deg': True, 'tilt_deg': 0},
                    {'pan_deg': float('nan'), 'tilt_deg': 0},
                    {'pan_deg': 181, 'tilt_deg': 0},
                    {'pan_deg': 0, 'tilt_deg': -31},
                    {'pan_deg': 0, 'tilt_deg': 91}):
        assert node.move_pan_tilt(invalid)[0] == 400
    assert not pub.messages
    assert node.move_pan_tilt(target)[0] == 200
    msg, = pub.messages
    assert msg.mode == PanTiltCommand.ABSOLUTE
    assert math.degrees(msg.pan_rad) == pytest.approx(12.5)
    assert math.degrees(msg.tilt_rad) == pytest.approx(45)
    assert (msg.speed_raw, msg.accel_raw) == (120, 20)
    node.pan_tilt_received -= 2
    assert node.move_pan_tilt(target)[0] == 503
    node.on_pan_tilt(state)
    pub.subscribers = 0
    assert node.move_pan_tilt(target)[0] == 503
    pub.subscribers = 1
    state.feedback_ok = False
    node.on_pan_tilt(state)
    assert node.move_pan_tilt(target)[0] == 503
    assert len(pub.messages) == 1


def test_timeout_keeps_gate_until_ros_finishes(node):
    client = Client()
    node.detectors['litter'] = client
    node.timeout = 0.01
    assert node.detect('litter')[0] == 504
    assert node.detect('litter')[0] == 409
    client.future.set_result(DetectRgbScene.Response())
    assert not node.busy


def test_response_preserves_vlm_timing_and_failure(node):
    client = Client()
    node.detectors['behaviour'] = client
    response = DetectRgbScene.Response()
    response.status = response.STATUS_VLM_ERROR
    response.error_msg = 'VLM failed'
    response.vlm_elapsed_s = 1.25
    response.vlm_model = 'glm-5.3-flash'
    response.header.stamp.sec = 123
    client.future.set_result(response)
    code, result = node.detect('behaviour')
    assert code == 200
    assert result['status'] == 2 and result['error'] == 'VLM failed'
    assert result['vlm_elapsed_s'] == 1.25
    assert result['stamp']['sec'] == 123
    assert result['model'] == 'glm-5.3-flash'


def test_http_page_stale_preview_and_origin(node):
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(node))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = 'http://127.0.0.1:' + str(server.server_port)
    try:
        with urlopen(root) as response:
            assert '检测人物行为' in response.read().decode()
        with urlopen(root + '/api/state') as response:
            assert json.load(response)['camera_ready'] is False
        node.jpeg = b'old jpeg'
        node.received = time.monotonic() - 10
        with pytest.raises(HTTPError) as error:
            urlopen(root + '/api/frame.jpg')
        assert error.value.code == 503
        req = Request(root + '/api/detect/litter', data=b'{}',
                      headers={'Content-Type': 'application/json', 'Origin': 'http://other-host'})
        with pytest.raises(HTTPError) as error:
            urlopen(req)
        assert error.value.code == 403
        req = Request(root + '/api/pan_tilt', data=b'{"pan_deg":181,"tilt_deg":0}',
                      headers={'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as error:
            urlopen(req)
        assert error.value.code == 400
        req = Request(root + '/api/pan_tilt', data=b'{"pan_deg":0,"tilt_deg":0}',
                      headers={'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as error:
            urlopen(req)
        assert error.value.code == 503
        req = Request(root + '/api/pan_tilt', data=b'{}',
                      headers={'Content-Type': 'application/json', 'Origin': 'http://other-host'})
        with pytest.raises(HTTPError) as error:
            urlopen(req)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
