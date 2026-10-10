"""ROS-free artifact contract tests, exercising the real shared logger.

OpenCV drawing/encoding is mocked because this test needs only the artifact
and timestamp contract. It does not validate rendered image quality.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tk_vision_specialized.scene_logging import log_scene
from tk_vision_specialized.scene_vlm import Observation


@pytest.fixture
def logger(monkeypatch, tmp_path):
    def write(path, image):
        Path(path).write_bytes(b'mocked jpeg')
        return True

    cv = SimpleNamespace(imwrite=write, rectangle=lambda *a: None,
                         putText=lambda *a: None, FONT_HERSHEY_SIMPLEX=0, LINE_AA=0)
    monkeypatch.setitem(sys.modules, 'cv2', cv)
    source = Path(__file__).resolve().parents[2] / 'vision_util/vision_util/vision_logging.py'
    spec = importlib.util.spec_from_file_location('scene_test_logger', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('TINKER_VISION_SESSION_TS', '20261010_120000')
    return module.VisionLogger(None, True, str(tmp_path), tag='behaviour_detection')


def test_capture_timestamp_boxes_images_and_unique_names(logger):
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    obs = [Observation(('sitting_waving',), 'Hand raised.', (1, 2, 30, 35))]
    header = {'frame_id': 'camera_color_optical_frame', 'stamp': {'sec': 123, 'nanosec': 456}}
    paths = [log_scene(logger, image, obs, task='behaviour', capture_header=header,
                       status=0, model_info={'provider': 'qwen', 'model': 'test'}) for _ in range(2)]
    assert paths[0] != paths[1]
    data = json.loads(Path(paths[0]).read_text())
    assert data['capture_header'] == header
    assert data['completed_at_utc'].endswith('+00:00')
    assert data['detections'][0]['bbox'] == [1, 2, 30, 35]
    assert data['detections'][0]['labels'] == ['sitting_waving']
    assert len(list(Path(logger.run_dir).glob('*.jpg'))) == 4


@pytest.mark.parametrize('status,error', [(0, ''), (2, 'VLM failed')])
def test_negative_and_error_still_saved(logger, status, error):
    path = log_scene(logger, np.zeros((20, 20, 3), dtype=np.uint8), [],
                     task='litter', capture_header={'stamp': {'sec': 1, 'nanosec': 0}},
                     status=status, error=error)
    data = json.loads(Path(path).read_text())
    assert data['detected'] is False
    assert data['status'] == status and data['error_msg'] == error


def test_no_frame_json(logger):
    path = log_scene(logger, None, [], task='litter', capture_header=None,
                     status=1, error='No fresh RGB')
    assert json.loads(Path(path).read_text())['capture_header'] is None
    assert not list(Path(logger.run_dir).glob('*.jpg'))


def test_silent_imwrite_failure_is_error(logger, monkeypatch):
    monkeypatch.setattr(sys.modules['cv2'], 'imwrite', lambda *a: False)
    with pytest.raises(OSError, match='Missing/empty'):
        log_scene(logger, np.zeros((20, 20, 3), dtype=np.uint8), [],
                  task='litter', capture_header=None, status=0)


def test_returned_overlay_is_persisted(logger, monkeypatch):
    cv = sys.modules['cv2']
    original_write = cv.imwrite
    writes = []

    def record(path, image):
        writes.append((path, image.copy()))
        return original_write(path, image)

    monkeypatch.setattr(cv, 'imwrite', record)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    overlay = np.full_like(image, 123)
    log_scene(logger, image, [], task='litter', capture_header=None, status=0,
              annotated_image=overlay)
    assert '_overlay_' in str(writes[-1][0])
    assert np.array_equal(writes[-1][1], overlay)
