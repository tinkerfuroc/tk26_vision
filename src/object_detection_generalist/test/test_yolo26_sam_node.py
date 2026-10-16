"""Contract tests for YOLO26 bbox detection followed by MobileSAM."""

from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
import torch
from rclpy.parameter import Parameter
from std_msgs.msg import Header
from tinker_vision_msgs_26.srv import ObjectDetectionGeneralist as Srv
from ultralytics.engine.results import Results

from object_detection_generalist.yolo26_sam_node import (
    Yolo26SamNode, drop_duplicate_boxes,
)


NAMES = {0: 'cup', 1: 'bottle', 2: 'dining table'}


class FakeDetectionModel:
    names = NAMES
    task = 'detect'

    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        detections = [
            [5, 5, 25, 25, 0.9, 0],
            [40, 5, 60, 25, 0.8, 1],
        ]
        requested = kwargs.get('classes')
        if requested is not None:
            detections = [
                row for row in detections if int(row[5]) in requested
            ]
        return [Results(
            image,
            path='',
            names=self.names,
            boxes=torch.tensor(detections),
        )]


class FakeSamPredictor:
    instances = []

    def __init__(self, weights_path, device, logger=None):
        self.weights_path = weights_path
        self.device = device
        self.calls = []
        self.instances.append(self)

    def segment(self, image, bboxes):
        self.calls.append(list(bboxes))
        masks = []
        for x1, y1, x2, y2 in bboxes:
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
        return masks, 0.01


@pytest.fixture
def node(monkeypatch):
    import threading
    import object_detection_generalist.yolo26_sam_node as module

    FakeSamPredictor.instances.clear()

    def init_model(self):
        self.device = 'cpu'
        self.model = FakeDetectionModel()
        self._yolo_model_lock = threading.Lock()
        self._yolo_loaded_device = self.device

    monkeypatch.setattr(Yolo26SamNode, '_init_model', init_model)
    monkeypatch.setattr(module, 'SamPredictor', FakeSamPredictor)
    monkeypatch.setattr(module, 'resolve_weights', lambda path: path)

    rclpy.init()
    n = Yolo26SamNode(parameter_overrides=[
        Parameter('camera_backend', value='subscription'),
        Parameter('vision_logging_enabled', value=False),
        Parameter('prefer_ffs', value=False),
    ])
    rgb = np.zeros((64, 96, 3), np.uint8)
    depth = np.ones((64, 96), np.float32)
    points = np.zeros((64, 96, 3), np.float32)
    points[..., 2] = 3.0
    points[:, 40:60, 2] = 1.0
    rgb_msg = n.bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
    depth_msg = n.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
    monkeypatch.setattr(
        n, '_wait_for_recent_frame', lambda camera: (rgb_msg, depth_msg))
    monkeypatch.setattr(n, '_get_intrinsic', lambda camera: object())

    def process(*args):
        return rgb, points, np.ones((64, 96), bool), Header(frame_id='camera')

    monkeypatch.setattr(n, '_process_orbbec_data', process)
    monkeypatch.setattr(n, '_process_realsense_data', process)
    monkeypatch.setattr(
        n, '_lookup_centroid_transform', lambda *a: (None, True))
    monkeypatch.setattr(n, '_orbbec_response_depth_intake', SimpleNamespace(
        latest=lambda: SimpleNamespace(depth_msg=depth_msg)))
    try:
        yield n
    finally:
        n.destroy_node()
        rclpy.shutdown()


def call(node, prompt='', **kwargs):
    return node._generalist_service_callback(
        Srv.Request(camera='orbbec', prompt=prompt, **kwargs), Srv.Response())


def test_defaults_and_service_abi(node):
    assert node.model_path == 'yolo26s.pt'
    assert node.sam_weights == 'mobile_sam.pt'
    assert (
        node.get_parameter('service_name').value
        == 'object_detection_yolo26_sam'
    )
    assert node.model.task == 'detect'


def test_bbox_model_then_sam_returns_aligned_segments(node):
    response = call(node, sort_closest=True, return_segments=True)

    assert response.status == 0
    assert response.detection_source == 'yolo'
    assert [obj.cls for obj in response.objects] == ['bottle', 'cup']
    assert [obj.object_id for obj in response.objects] == [1, 0]
    assert [obj.centroid.z for obj in response.objects] == [1.0, 3.0]
    assert node.model.calls[-1]['classes'] is None
    assert node.model.calls[-1]['conf'] == 0.25
    assert FakeSamPredictor.instances[0].calls[-1] == [
        (5, 5, 25, 25), (40, 5, 60, 25),
    ]

    masks = [node.bridge.imgmsg_to_cv2(msg) for msg in response.segments]
    assert all(msg.encoding == '8UC1' for msg in response.segments)
    assert masks[0][10, 45] == 255 and masks[0][10, 10] == 0
    assert masks[1][10, 10] == 255 and masks[1][10, 45] == 0


def test_exact_label_filter_reaches_detector_and_sam(node):
    response = call(node, 'BOTTLE', return_segments=True)
    assert [obj.cls for obj in response.objects] == ['bottle']
    assert node.model.calls[-1]['classes'] == [1]
    assert FakeSamPredictor.instances[0].calls[-1] == [(40, 5, 60, 25)]


def test_unknown_label_stops_before_detector_and_sam(node):
    response = call(node, 'unknown', force_vlm_sam=True,
                    use_vlm_sam_fallback=True)
    assert response.status == 1
    assert response.detection_source == 'none'
    assert 'Unknown checkpoint label' in response.error_msg
    assert not node.model.calls
    assert not FakeSamPredictor.instances[0].calls


def test_sam_failure_is_reported_without_crashing_service(node, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('simulated MobileSAM failure')

    monkeypatch.setattr(node._sam, 'segment', fail)
    response = call(node)
    assert response.status == 1
    assert response.detection_source == 'none'
    assert 'simulated MobileSAM failure' in response.error_msg


def test_duplicate_labels_on_one_object_keep_highest_conf():
    # Boxes from a real 2026-10-16 scan: banana and coke on one object.
    bboxes = [(694, 261, 913, 501), (691, 265, 744, 421),
              (535, 261, 725, 499), (691, 263, 744, 421)]
    confs = [0.62, 0.62, 0.47, 0.38]
    assert drop_duplicate_boxes(bboxes, confs, 0.7) == [0, 1, 2]
    assert drop_duplicate_boxes(bboxes, confs, 0.0) == [0, 1, 2, 3]


class DuplicateLabelModel(FakeDetectionModel):
    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        return [Results(image, path='', names=self.names, boxes=torch.tensor([
            [5, 5, 25, 25, 0.5, 0],
            [5, 6, 25, 25, 0.9, 1],
            [40, 5, 60, 25, 0.8, 1],
        ]))]


def test_duplicate_filter_runs_before_sam(node):
    node.model = DuplicateLabelModel()
    response = call(node)
    assert sorted((o.cls, round(o.conf, 2)) for o in response.objects) == [
        ('bottle', 0.8), ('bottle', 0.9)]
    assert FakeSamPredictor.instances[0].calls[-1] == [
        (5, 6, 25, 25), (40, 5, 60, 25)]
