"""Gemini match client (OpenRouter compatible endpoint)."""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np
from dotenv import load_dotenv
from vision_util.vlm_models import vision_vlm_model

from ._vlm_common import strip_fences, encode_data_url
from .nms import MatchRow, Bbox


# Load .env once at module-import time so pytest's monkeypatch.delenv is
# authoritative after first construction (the workspace .env carries a
# real OPENROUTER_API_KEY which would otherwise repopulate after delete).
# Same pattern Task 5's QwenMatchClient uses; matches kimi_api/_env.py.
load_dotenv(override=False)


_GEMINI_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'


def _gemini_default_model() -> str:
    return vision_vlm_model()


def _decode_bbox_pixels(
    bbox_xyxy, scene_w: int, scene_h: int,
) -> Bbox | None:
    if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(bbox_xyxy[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), scene_w - 1))
    y1 = max(0, min(int(round(y1)), scene_h - 1))
    x2 = max(0, min(int(round(x2)), scene_w - 1))
    y2 = max(0, min(int(round(y2)), scene_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def decode_gemini_response(
    body: str, *, scene_w: int, scene_h: int, allowed_labels: set[str],
) -> list[MatchRow]:
    try:
        parsed = json.loads(strip_fences(body))
    except (json.JSONDecodeError, ValueError):
        return []
    detections = (
        parsed.get('detections') if isinstance(parsed, dict) else None
    )
    if not isinstance(detections, list):
        return []

    rows: list[MatchRow] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = det.get('label')
        if not isinstance(label, str) or label not in allowed_labels:
            continue
        bbox = _decode_bbox_pixels(det.get('bbox_xyxy'), scene_w, scene_h)
        if bbox is None:
            continue
        try:
            conf = float(det.get('confidence', 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(conf, 1.0))
        rows.append(MatchRow(label=label, bbox=bbox, conf=conf))
    return rows


def _gemini_system_prompt(
    labels: list[str], scene_w: int, scene_h: int,
) -> str:
    label_list = ', '.join(f'"{lbl}"' for lbl in labels)
    return (
        "You are a visual-grounding assistant for a service robot. The "
        "user provides one SCENE image of size "
        f"{scene_w}x{scene_h} followed by {len(labels)} REFERENCE images "
        f"captioned with labels from this set: [{label_list}]. Find every "
        "visible instance of any reference item in the scene and return "
        "bounding boxes. Coordinates 'bbox_xyxy' are [x1, y1, x2, y2] in "
        f"absolute pixels over the {scene_w}x{scene_h} scene image. The "
        f"'label' field must be exactly one of [{label_list}]. Confidence "
        "is a subjective match score in [0.0, 1.0]. If no reference item "
        "is visible, return detections=[]. Output JSON only, with no "
        "commentary or markdown fences."
    )


class GeminiMatchClient:
    def __init__(
        self,
        model: str = '',
        base_url: str = '',
    ):
        # load_dotenv ran at module-import time; just read os.environ here
        self._api_key = os.environ.get('OPENROUTER_API_KEY', '')
        if not self._api_key:
            raise RuntimeError(
                'OPENROUTER_API_KEY not found in env '
                '(required for Gemini provider)'
            )
        self._model = model or _gemini_default_model()
        self._base_url = base_url or _GEMINI_DEFAULT_BASE_URL

    def match_batch(
        self,
        scene_bgr: np.ndarray,
        refs: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> list[MatchRow]:
        if not refs:
            return []

        from openai import OpenAI

        h, w = scene_bgr.shape[:2]
        labels = [label for label, _url in refs]
        allowed_labels = set(labels)

        scene_url = encode_data_url(scene_bgr)
        content: list[dict] = [
            {'type': 'image_url', 'image_url': {'url': scene_url}},
        ]
        for label, url in refs:
            content.append(
                {'type': 'image_url', 'image_url': {'url': url}},
            )
        content.append({
            'type': 'text',
            'text': (
                'Image 1 is the scene. The remaining images are reference '
                'photos, in order: '
                + ', '.join(
                    f'image {i+2} = "{lbl}"'
                    for i, lbl in enumerate(labels)
                )
                + '. Return all visible instances grouped by label.'
            ),
        })

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        try:
            for attempt in range(max(1, max_retries)):
                try:
                    completion = client.with_options(
                        timeout=timeout_s,
                    ).chat.completions.create(
                        model=self._model,
                        messages=[
                            {'role': 'system',
                             'content': _gemini_system_prompt(labels, w, h)},
                            {'role': 'user', 'content': content},
                        ],
                        response_format={'type': 'json_object'},
                    )
                    raw = completion.choices[0].message.content or ''
                    return decode_gemini_response(
                        raw,
                        scene_w=w, scene_h=h,
                        allowed_labels=allowed_labels,
                    )
                except Exception as exc:    # noqa: BLE001
                    if logger is not None:
                        logger.warning(
                            f'Gemini match attempt {attempt+1}/'
                            f'{max_retries} failed: {exc}'
                        )
            return []
        finally:
            try:
                client.close()
            except Exception:    # noqa: BLE001
                pass
