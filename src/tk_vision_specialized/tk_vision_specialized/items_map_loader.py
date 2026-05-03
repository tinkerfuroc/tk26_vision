"""Items-map loader for the object_match service.

Reads ``items_map.yaml`` (key -> filename) once at startup, encodes each
reference JPEG as a base64 data URL, and serves both the data URL and the
raw numpy image on demand. The ten reference images are tiny (~100 KB each)
so we keep them resident in memory rather than re-reading per request.

Schema of items_map.yaml is intentionally minimal:

    biscuit: biscuit.jpg
    cookie: cookie.jpg
    ...

If a referenced JPEG is missing on disk the entry is dropped with a warning
so a single bad item never knocks the whole node out.
"""

from __future__ import annotations

import base64
import os
from typing import Iterable

import cv2
import numpy as np
import yaml


_JPEG_MIME = 'data:image/jpeg;base64,'
_PNG_MIME = 'data:image/png;base64,'


class ItemsMapLoader:
    """Resolve item-name -> reference image / data URL."""

    def __init__(self, items_dir: str, logger=None):
        self.items_dir = items_dir
        self._logger = logger
        self._cache: dict[str, dict] = {}
        self._load()

    def _log_info(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)

    def _log_warn(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.warning(msg)

    def _load(self) -> None:
        yaml_path = os.path.join(self.items_dir, 'items_map.yaml')
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(
                f'items_map.yaml not found at {yaml_path}'
            )
        with open(yaml_path, 'r', encoding='utf-8') as f:
            mapping = yaml.safe_load(f) or {}
        if not isinstance(mapping, dict):
            raise ValueError(
                f'items_map.yaml must be a mapping; got {type(mapping).__name__}'
            )

        for name, filename in mapping.items():
            if not isinstance(name, str) or not isinstance(filename, str):
                self._log_warn(
                    f'items_map: skipping non-string entry {name!r} -> {filename!r}'
                )
                continue
            img_path = os.path.join(self.items_dir, filename)
            try:
                with open(img_path, 'rb') as f:
                    raw_bytes = f.read()
            except OSError as exc:
                self._log_warn(
                    f'items_map: cannot read "{name}" at {img_path}: {exc}'
                )
                continue
            # Reference images on disk are already JPEG/PNG; base64-encode the
            # raw bytes directly to avoid a lossy cv2 decode -> re-encode pass.
            mime = (
                _PNG_MIME if filename.lower().endswith('.png')
                else _JPEG_MIME
            )
            data_url = mime + base64.b64encode(raw_bytes).decode('utf-8')
            self._cache[name] = {
                'filename': filename,
                'data_url': data_url,
                # Lazy: only decode to numpy if the consumer asks for the
                # image. The service callback only needs the data URL.
                '_raw_bytes': raw_bytes,
                'image': None,
            }

        self._log_info(
            f'ItemsMapLoader loaded {len(self._cache)} items from {self.items_dir}: '
            + ', '.join(sorted(self._cache.keys()))
        )

    def __contains__(self, name: str) -> bool:
        return name in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def keys(self) -> Iterable[str]:
        return self._cache.keys()

    def get_data_url(self, name: str) -> str:
        return self._cache[name]['data_url']

    def get_image(self, name: str) -> np.ndarray:
        entry = self._cache[name]
        img = entry['image']
        if img is None:
            arr = np.frombuffer(entry['_raw_bytes'], dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            entry['image'] = img
        return img

    def get_filename(self, name: str) -> str:
        return self._cache[name]['filename']
