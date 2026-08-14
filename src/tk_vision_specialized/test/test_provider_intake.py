"""Focused tests for inherited servers' provider-backed intake paths."""
from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace


_STUBBED_MODULES = []
if 'torch' not in sys.modules and importlib.util.find_spec('torch') is None:
    torch_stub = types.ModuleType('torch')
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch_stub
    _STUBBED_MODULES.append('torch')

if (
    'ultralytics' not in sys.modules
    and importlib.util.find_spec('ultralytics') is None
):
    ultralytics_stub = types.ModuleType('ultralytics')
    ultralytics_stub.YOLO = object
    sys.modules['ultralytics'] = ultralytics_stub
    _STUBBED_MODULES.append('ultralytics')

from tk_vision_specialized.object_match_server import (  # noqa: E402
    ObjectMatchServer,
)
from tk_vision_specialized.placing_location_server import (  # noqa: E402
    PlacingLocationServer,
)

for _module_name in _STUBBED_MODULES:
    sys.modules.pop(_module_name, None)


def test_inherited_servers_request_provider_frame_and_camera_info_once():
    bundle = SimpleNamespace(
        color_msg={'frame': 'color'},
        depth_msg={'frame': 'depth'},
    )
    info = {'camera': 'info'}

    class Intake:
        cfg = SimpleNamespace(backend='service')

        def __init__(self):
            self.calls = []

        def wait_fresh(self, **kwargs):
            self.calls.append(kwargs)
            return bundle

        def camera_info(self):
            return info

    for server_type in (ObjectMatchServer, PlacingLocationServer):
        intake = Intake()
        owner = SimpleNamespace(
            _camera_intakes={'orbbec': intake},
            img_sync_thres=0.2,
            sync_wait_time_limit=5,
        )

        pair = server_type._wait_for_recent_frame(owner, 'orbbec')
        intrinsic = server_type._get_intrinsic(owner, 'orbbec')

        assert pair == (bundle.color_msg, bundle.depth_msg)
        assert pair[0] is not bundle.color_msg
        assert intrinsic == info
        assert intrinsic is not info
        assert intake.calls == [{
            'max_age_s': 0.2,
            'timeout_s': 0.5,
            'on_timeout': 'fail',
        }]
