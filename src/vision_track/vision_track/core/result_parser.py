import logging
from typing import List

import cv2
import numpy as np

from .tracking_types import TrackingResult

logger = logging.getLogger(__name__)


def parse_yolo_results(result) -> List[TrackingResult]:
    """
    Parse YOLO results into TrackingResult objects.

    Args:
        result: YOLO result object

    Returns:
        List of TrackingResult objects
    """
    tracking_results: List[TrackingResult] = []

    if result.boxes is None or len(result.boxes) == 0:
        return tracking_results

    boxes = result.boxes
    masks = result.masks
    names = result.names

    xyxy_data = boxes.xyxy
    cls_data = boxes.cls
    conf_data = boxes.conf
    id_data = boxes.id

    xyxy_np = xyxy_data.cpu().numpy() if hasattr(xyxy_data, "cpu") else np.asarray(xyxy_data)
    cls_np = cls_data.cpu().numpy() if hasattr(cls_data, "cpu") else np.asarray(cls_data)
    conf_np = conf_data.cpu().numpy() if hasattr(conf_data, "cpu") else np.asarray(conf_data)
    id_np = None
    if id_data is not None:
        id_np = id_data.cpu().numpy() if hasattr(id_data, "cpu") else np.asarray(id_data)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy_np[i].astype(int)
        track_id = int(id_np[i]) if id_np is not None else -1
        confidence = float(conf_np[i])
        class_id = int(cls_np[i])
        class_name = names[class_id]

        mask = _extract_mask(masks, i, result)

        tracking_results.append(
            TrackingResult(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                mask=mask,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )
        )

    return tracking_results


def _extract_mask(masks, index: int, result):
    """Extract and resize mask for a single detection."""
    if masks is None or index >= len(masks):
        return None

    try:
        mask_obj = masks[index]
        if hasattr(mask_obj, "data"):
            mask_data = mask_obj.data
            mask_data = mask_data[0] if hasattr(mask_data, "__getitem__") else mask_data
            if hasattr(mask_data, "cpu"):
                mask = mask_data.cpu().numpy()
            elif isinstance(mask_data, np.ndarray):
                mask = mask_data
            else:
                mask = np.asarray(mask_data)
        elif hasattr(mask_obj, "xy"):
            return None
        else:
            mask = None
    except Exception as exc:
        logger.debug(f"Failed to extract mask: {exc}")
        return None

    if mask is None:
        return None

    mask = cv2.resize(
        mask.astype(np.float32),
        (result.orig_shape[1], result.orig_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return (mask > 0.5).astype(np.uint8)
