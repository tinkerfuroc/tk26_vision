"""Shared image helpers for kimi_api nodes."""

import base64

import cv2
import numpy as np


def bbox_from_mask(mask):
    """Tight bbox of a binary mask, returned as (y1, x1, y2, x2) in pixel coords.

    Order is (row_min, col_min, row_max, col_max) — i.e. y first, x second —
    matching numpy slicing `img[y1:y2, x1:x2]` directly.
    """
    nonzero = np.nonzero(mask)
    return (
        int(np.min(nonzero[0])),
        int(np.min(nonzero[1])),
        int(np.max(nonzero[0])),
        int(np.max(nonzero[1])),
    )


def encode_to_data_url(img) -> str:
    """Encode a BGR numpy image as a JPEG data URL, in-memory."""
    ok, buf = cv2.imencode('.jpg', img)
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return f'data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode("utf-8")}'
