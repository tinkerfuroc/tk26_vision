"""YOLO-World multi-class prompts (GPSR sim battery 2026-09-30 fix5).

A ' . '-joined prompt used to be registered as ONE class, so every box came
back labelled with the whole prompt and a category count could not tell the
objects apart. Each segment is now its own class and ``last_labels`` carries
the class each box matched.
"""
import numpy as np

from object_detection_generalist.world_bbox import WorldDetector


class _T:
    def __init__(self, a):
        self._a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _Model:
    def __init__(self, result):
        self._result = result

    def predict(self, *a, **k):
        return [self._result]


def _detector(result):
    det = WorldDetector.__new__(WorldDetector)
    det._model = _Model(result)
    det._device = "cpu"
    det._conf_threshold = 0.05
    det._iou_threshold = 0.5
    det._last_classes = None
    det._logger = None
    det.registered = []
    det._set_classes_on_device = lambda classes: det.registered.append(list(classes))
    return det


def test_each_prompt_segment_is_its_own_class_and_labels_follow_cls_ids():
    det = _detector(_Result(_Boxes([[10, 10, 50, 50], [60, 10, 90, 60]], [0.8, 0.4], [2, 0])))
    boxes, confs, _ = det.detect(np.zeros((100, 100, 3), np.uint8),
                                 "red bowl . yellow mustard bottle . white bleach cleanser bottle")
    assert det.registered == [["red bowl", "yellow mustard bottle", "white bleach cleanser bottle"]]
    assert len(boxes) == 2 and confs == [0.8, 0.4]
    assert det.last_labels == ["white bleach cleanser bottle", "red bowl"]


def test_single_class_prompt_is_unchanged():
    det = _detector(_Result(_Boxes([[10, 10, 50, 50]], [0.9], [0])))
    det.detect(np.zeros((100, 100, 3), np.uint8), "person")
    assert det.registered == [["person"]]
    assert det.last_labels == ["person"]
