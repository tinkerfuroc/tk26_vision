"""Contract tests for checkpoint labels, real geometry and generalist payloads."""

from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
import torch
from geometry_msgs.msg import Point
from rclpy.parameter import Parameter
from std_msgs.msg import Header
from tinker_vision_msgs_26.srv import ObjectDetectionGeneralist as Srv
from ultralytics.engine.results import Results

from object_detection_generalist.yolo26_seg_node import Yolo26SegNode, select_labels


NAMES = {0: 'cup', 1: 'bottle', 2: 'dining table'}


@pytest.mark.parametrize('prompt', ['', '  ', '\n'])
def test_empty_prompt_means_all(prompt):
    assert select_labels(prompt, NAMES) is None


@pytest.mark.parametrize('prompt', [
    'cup . bottle', 'CUP, bottle', 'cup\nbottle', 'cup，bottle',
    'cup; bottle', 'cup | bottle', 'cup . cup . bottle',
])
def test_multiple_exact_labels(prompt):
    assert select_labels(prompt, NAMES) == {'cup', 'bottle'}


def test_multiword_and_custom_checkpoint_labels():
    assert select_labels('Dining Table', NAMES) == {'dining table'}
    assert select_labels('red_widget', {7: 'red_widget'}) == {'red_widget'}


@pytest.mark.parametrize('prompt', ['mug', 'cup, mug', 'cupboard', ', ;'])
def test_unknown_or_separator_only_prompt_is_rejected(prompt):
    with pytest.raises(ValueError):
        select_labels(prompt, NAMES)


class FakeModel:
    names = NAMES
    task = 'segment'

    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        masks = torch.zeros((2, *image.shape[:2]))
        masks[0, 5:25, 5:25] = 1
        masks[1, 5:25, 40:60] = 1
        return [Results(
            image, path='', names=self.names,
            boxes=torch.tensor([
                [5, 5, 25, 25, 0.9, 0],
                [40, 5, 60, 25, 0.8, 1],
            ]),
            masks=masks,
        )]


@pytest.fixture
def node(monkeypatch):
    import threading

    def init_model(self):
        self.model = FakeModel()
        self._yolo_model_lock = threading.Lock()

    monkeypatch.setattr(Yolo26SegNode, '_init_model', init_model)
    rclpy.init()
    n = Yolo26SegNode(parameter_overrides=[
        Parameter('camera_backend', value='subscription'),
        Parameter('vision_logging_enabled', value=False),
        Parameter('prefer_ffs', value=False),
    ])
    rgb = np.zeros((64, 96, 3), np.uint8)
    depth = np.ones((64, 96), np.float32)
    points = np.zeros((64, 96, 3), np.float32)
    points[..., 2] = 3.0
    points[:, 40:60, 2] = 1.0
    n.test_points = points
    rgb_msg = n.bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
    depth_msg = n.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
    n.test_depth = depth_msg
    monkeypatch.setattr(n, '_wait_for_recent_frame', lambda camera: (rgb_msg, depth_msg))
    monkeypatch.setattr(n, '_get_intrinsic', lambda camera: object())
    def process(*args):
        return rgb, points, np.ones((64, 96), bool), Header(frame_id='camera')
    monkeypatch.setattr(n, '_process_orbbec_data', process)
    monkeypatch.setattr(n, '_process_realsense_data', process)
    monkeypatch.setattr(n, '_lookup_centroid_transform', lambda *a: (None, True))
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


def test_all_labels_sorted_with_aligned_payloads(node):
    res = call(node, sort_closest=True, return_segments=True,
               return_rgb_image=True, return_depth_image=True)
    assert res.status == 0
    assert res.detection_source == 'yolo'
    assert res.person_id == 0
    assert [o.cls for o in res.objects] == ['bottle', 'cup']
    assert [o.object_id for o in res.objects] == [1, 0]
    assert [o.centroid.z for o in res.objects] == [1.0, 3.0]
    assert (res.rgb_image.height, res.rgb_image.width) == (64, 96)
    assert res.depth_image == node.test_depth
    masks = [node.bridge.imgmsg_to_cv2(s) for s in res.segments]
    assert all(s.encoding == '8UC1' for s in res.segments)
    assert all(m.shape == (64, 96) for m in masks)
    assert masks[0][10, 45] == 255 and masks[0][10, 10] == 0
    assert masks[1][10, 10] == 255 and masks[1][10, 45] == 0


def test_filtered_then_empty_prompt_resets_predictor_classes(node):
    assert [o.cls for o in call(node, 'BOTTLE').objects] == ['bottle']
    assert node.model.calls[-1]['classes'] == [1]
    assert len(call(node).objects) == 2
    assert node.model.calls[-1]['classes'] is None
    assert node.model.calls[-1]['retina_masks'] is True
    assert node.model.calls[-1]['conf'] == 0.25


def test_multiclass_sort_and_optional_payloads(node):
    res = call(node, 'cup . bottle', sort_closest=True)
    assert [o.cls for o in res.objects] == ['bottle', 'cup']
    assert not res.segments
    assert res.rgb_image.height == res.depth_image.height == 0


def test_unknown_does_not_run_inference_or_fallback(node):
    res = call(node, 'cup, unknown', force_vlm_sam=True,
               use_vlm_sam_fallback=True)
    assert res.status == 1 and res.detection_source == 'none'
    assert 'unknown' in res.error_msg
    assert not node.model.calls


def test_legacy_fallback_flags_still_use_checkpoint(node):
    res = call(node, force_vlm_sam=True, use_vlm_sam_fallback=True)
    assert res.status == 0 and res.detection_source == 'yolo'


def test_sort_uses_camera_distance_after_target_frame_transform(node, monkeypatch):
    monkeypatch.setattr(node, '_lookup_centroid_transform', lambda *a: (object(), True))
    monkeypatch.setattr(node, '_apply_centroid_transform',
                        lambda p, *a: Point(x=p.x, y=p.y, z=p.z - 10.0))
    res = call(node, sort_closest=True, target_frame='map')
    assert res.header.frame_id == 'map'
    assert [o.cls for o in res.objects] == ['bottle', 'cup']
    assert [o.centroid.z for o in res.objects] == [-9.0, -7.0]


def test_highest_sort_and_closest_precedence(node, monkeypatch):
    monkeypatch.setattr(node._tf_helper, 'try_lookup', lambda *a, **kw: object())
    monkeypatch.setattr(node._tf_helper, 'transform_point', lambda point, tf: point)
    assert [o.cls for o in call(node, sort_highest=True).objects] == ['cup', 'bottle']
    assert [o.cls for o in call(node, sort_highest=True, sort_closest=True).objects] == [
        'bottle', 'cup']


def test_realsense_range_gate_preserves_mask_alignment(node):
    node.test_points[..., 2] = 1.5
    node.test_points[:, 40:60, 2] = 0.5
    res = node._generalist_service_callback(
        Srv.Request(camera='realsense', return_segments=True, return_depth_image=True),
        Srv.Response())
    assert [o.cls for o in res.objects] == ['bottle']
    assert res.objects[0].centroid.x == 0.5
    assert len(res.segments) == 1
    assert res.depth_image == node.test_depth


def test_no_frame_returns_generalist_failure(node, monkeypatch):
    monkeypatch.setattr(node, '_wait_for_recent_frame', lambda camera: None)
    res = call(node)
    assert res.status == 1 and not res.objects
    assert 'camera frame' in res.error_msg


def test_inference_failure_returns_error(node, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('simulated inference failure')
    monkeypatch.setattr(node, '_detect_objects', fail)
    res = call(node)
    assert res.status == 1
    assert 'simulated inference failure' in res.error_msg

