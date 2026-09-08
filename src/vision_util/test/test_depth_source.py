"""Unit tests for FoundationStereo-preferred depth acquisition."""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def _depth_source_module_with_fakes():
    callback_module = types.ModuleType('rclpy.callback_groups')

    class ReentrantCallbackGroup:
        pass

    class CvBridge:
        pass

    callback_module.ReentrantCallbackGroup = ReentrantCallbackGroup
    cv_bridge_module = types.ModuleType('cv_bridge')
    cv_bridge_module.CvBridge = CvBridge
    rclpy_module = types.ModuleType('rclpy')
    rclpy_module.callback_groups = callback_module
    modules = {
        'cv_bridge': cv_bridge_module,
        'rclpy': rclpy_module,
        'rclpy.callback_groups': callback_module,
    }
    missing = object()
    previous = {
        name: sys.modules.get(name, missing)
        for name in modules
    }
    sys.modules.update(modules)
    sys.modules.pop('vision_util.depth_source', None)
    try:
        imported = importlib.import_module('vision_util.depth_source')
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return imported


depth_source = _depth_source_module_with_fakes()


class FakeService:
    class Request:
        def __init__(self):
            self.align_to_color = False


class FakeFuture:
    def __init__(self, response=None, complete=True, error=None):
        self.response = response
        self.complete = complete
        self.error = error

    def add_done_callback(self, callback):
        if self.complete:
            callback(self)

    def result(self):
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, available=True, complete=True):
        self.srv_name = '/foundation_stereo/get_depth'
        self.response = response
        self.available = available
        self.complete = complete
        self.requests = []
        self.removed = []

    def wait_for_service(self, timeout_sec):
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        return FakeFuture(self.response, complete=self.complete)

    def remove_pending_request(self, future):
        self.removed.append(future)


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)


class FakeParameter:
    def __init__(self, value):
        self.value = value


class FakeNode:
    def __init__(self, client):
        self.client = client
        self.logger = FakeLogger()
        self.params = {}
        self.create_count = 0

    def declare_parameter(self, name, default):
        self.params.setdefault(name, default)

    def get_parameter(self, name):
        return FakeParameter(self.params[name])

    def create_client(self, service_type, service_name, callback_group=None):
        self.create_count += 1
        self.client.srv_name = service_name
        return self.client

    def destroy_client(self, client):
        pass

    def get_logger(self):
        return self.logger


class FakeBridge:
    def imgmsg_to_cv2(self, msg, desired_encoding='passthrough'):
        return msg.array


def _response(depth, status=0):
    image = types.SimpleNamespace(array=np.asarray(depth))
    return types.SimpleNamespace(status=status, depth_image=image)


def _source(client, native):
    node = FakeNode(client)
    source = depth_source.FfsPreferredDepthSource(
        node,
        native_depth_provider=lambda: native,
        bridge=FakeBridge(),
        service_type=FakeService,
    )
    node.params['ffs_call_timeout_s'] = 0.0
    return node, source


def test_prefers_ffs_and_passes_alignment_flag():
    client = FakeClient(response=_response([[1.25]], status=0))
    node, source = _source(
        client, np.array([[2000]], dtype=np.uint16)
    )

    depth, tag = source.acquire(align_to_color=True)

    assert tag == 'ffs'
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [[1.25]])
    assert client.requests[0].align_to_color is True
    assert not node.logger.warnings


def test_unavailable_ffs_falls_back_to_native_float64():
    client = FakeClient(available=False)
    node, source = _source(
        client, np.array([[2500]], dtype=np.uint16)
    )

    depth, tag = source.acquire(align_to_color=False)

    assert tag == 'native'
    assert depth.dtype == np.float64
    np.testing.assert_allclose(depth, [[2.5]])
    assert node.logger.warnings


def test_ffs_timeout_removes_request_and_falls_back():
    client = FakeClient(complete=False)
    _, source = _source(client, np.array([[500]], dtype=np.uint16))

    depth, tag = source.acquire(align_to_color=True)

    assert tag == 'native'
    np.testing.assert_allclose(depth, [[0.5]])
    assert len(client.removed) == 1


def test_prefer_ffs_false_skips_client():
    client = FakeClient(response=_response([[9.0]]))
    node, source = _source(client, np.array([[1000]], dtype=np.uint16))
    node.params['prefer_ffs'] = False

    depth, tag = source.acquire(align_to_color=True)

    assert tag == 'native'
    np.testing.assert_allclose(depth, [[1.0]])
    assert node.create_count == 0


def test_float_native_input_is_bug_compatibly_treated_as_millimetres():
    client = FakeClient(available=False)
    _, source = _source(client, np.array([[1.75]], dtype=np.float32))

    depth, tag = source.acquire(align_to_color=True)

    assert tag == 'native'
    assert depth.dtype == np.float64
    np.testing.assert_allclose(depth, [[0.00175]])
