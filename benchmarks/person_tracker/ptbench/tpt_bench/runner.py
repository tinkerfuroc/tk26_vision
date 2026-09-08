"""Thin driver that runs vision_track's YOLOTracker over a TPT-Bench sequence.

This module is intentionally NOT unit-tested: it imports the heavy
``vision_track`` tracker (which loads a YOLO model + ReID weights) and reads
images from disk. The heavy import is *deferred into the function body* so that
``import ptbench.tpt_bench.runner`` succeeds even when ``vision_track`` / the
model is not importable (e.g. on CI without the ROS workspace sourced).

Requirements to actually run:

* The ROS 2 / colcon workspace must be sourced so ``vision_track`` is on the
  Python path::

      source /home/tinker/tk25_ws/install/setup.bash

  (``vision_track`` lives at
  ``/home/tinker/tk25_ws/src/tk26_vision/src/vision_track``.)
* ``opencv-python`` and the tracker's model weights must be available (they are
  in ``.venv-vision-main``).

Protocol: the tracker is force-initialised on frame 1's ground-truth box, then
``update`` is called for every subsequent frame. The first frame's prediction
is the init box itself (confidence 1.0).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .dataset import TptFrame

Box = Optional[Tuple[float, float, float, float]]


def _load_rgb(image_path: str):
    """Load an image as an RGB numpy array (deferred cv2 import)."""
    import cv2  # deferred: keep module import light-weight

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {image_path!r}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_tracker_on_sequence(
    frames: Sequence[TptFrame],
    *,
    imgsz: int = 1280,
    conf: float = 0.5,
) -> Tuple[List[Box], List[float]]:
    """Run the YOLO person tracker across an aligned list of frames.

    The tracker is initialised on the first frame that carries a ground-truth
    box (normally frame 0). Frames before that init point get a ``None``
    prediction with score 0.0.

    Args:
        frames: ordered :class:`TptFrame` list from
            :func:`ptbench.tpt_bench.dataset.load_sequence`.
        imgsz: inference image size passed to the tracker.
        conf: detection confidence threshold for the tracker.

    Returns:
        ``(pred_boxes, scores)`` aligned 1:1 with ``frames``. ``pred_boxes[i]``
        is the predicted xyxy box (or ``None`` when the tracker returns no
        target); ``scores[i]`` is the prediction confidence (``result.confidence``
        when available, else 1.0; 0.0 where there is no prediction).

    Raises:
        ImportError: if ``vision_track`` cannot be imported (workspace not
            sourced). The error is raised here, not at module import.
    """
    # Deferred heavy import — only when actually running.
    from vision_track.track_yolo import YOLOTracker

    tracker = YOLOTracker(confidence_threshold=conf, inference_size=imgsz)

    pred_boxes: List[Box] = []
    scores: List[float] = []
    initialized = False

    for frame in frames:
        rgb = _load_rgb(frame.image_path)

        if not initialized:
            # Cannot init until we have a ground-truth box to seed on.
            if frame.gt_bbox is None:
                pred_boxes.append(None)
                scores.append(0.0)
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in frame.gt_bbox)
            ok = tracker.initialize_tracking(
                rgb, target_bbox=(x1, y1, x2, y2), target_class="person"
            )
            initialized = True
            if ok:
                # The init box is our frame-1 prediction (full confidence).
                pred_boxes.append((float(x1), float(y1), float(x2), float(y2)))
                scores.append(1.0)
            else:
                pred_boxes.append(None)
                scores.append(0.0)
            continue

        result = tracker.update(rgb)
        if result is None or result.bbox is None:
            pred_boxes.append(None)
            scores.append(0.0)
        else:
            bx1, by1, bx2, by2 = result.bbox
            pred_boxes.append((float(bx1), float(by1), float(bx2), float(by2)))
            conf_val = getattr(result, "confidence", None)
            scores.append(float(conf_val) if conf_val is not None else 1.0)

    return pred_boxes, scores
