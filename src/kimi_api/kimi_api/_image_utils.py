"""Shared image helpers for kimi_api nodes."""

import base64
import os
import tempfile

import cv2
import numpy as np


def bbox_from_mask(mask):
    nonzero = np.nonzero(mask)
    x1, y1, x2, y2 = np.min(nonzero[0]), np.min(nonzero[1]), np.max(nonzero[0]), np.max(nonzero[1])
    return x1, y1, x2, y2


def encode_to_data_url(img) -> str:
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, img)
        with open(tmp_path, 'rb') as f:
            data = f.read()
    finally:
        os.unlink(tmp_path)
    return f'data:image/jpg;base64,{base64.b64encode(data).decode("utf-8")}'
