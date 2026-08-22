"""Provider-agnostic match client for object_match_all.

The match client takes a scene BGR image plus a list of (label,
ref_data_url) pairs and asks the VLM to ground every reference in the
scene, returning a list of MatchRow. Two backends ship in this module:

- QwenMatchClient: DashScope Qwen3-VL, normalized 0..1000 coords
- GeminiMatchClient: OpenRouter Gemini, absolute pixel coords (Task 6)

`build_match_client(provider, **opts)` is the factory the node uses."""

from __future__ import annotations

import json
import os
from typing import Protocol, Sequence

import numpy as np
from dotenv import load_dotenv
from vision_util.vlm_models import vision_qwen_model

from ._vlm_common import strip_fences, encode_data_url
from .nms import MatchRow, Bbox


_QWEN_DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
_QWEN_KEY_NAMES = ('DASHCOPE_API_KEY', 'DASHSCOPE_API_KEY')


def _qwen_default_model() -> str:
    return vision_qwen_model()


# Load `.env` once at module import so per-instance resolution can rely on
# `os.environ` exclusively — keeps pytest monkeypatch behaviour predictable.
load_dotenv(override=False)


class MatchClient(Protocol):
    def match_batch(
        self,
        scene_bgr: np.ndarray,
        refs: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> list[MatchRow]: ...


def _decode_bbox_normalized(
    box_2d, scene_w: int, scene_h: int,
) -> Bbox | None:
    """Decode a [x1, y1, x2, y2] 0..1000-normalized box to scene pixel xyxy."""
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(box_2d[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    px1 = int(round(x1 * scene_w / 1000.0))
    py1 = int(round(y1 * scene_h / 1000.0))
    px2 = int(round(x2 * scene_w / 1000.0))
    py2 = int(round(y2 * scene_h / 1000.0))
    px1 = max(0, min(px1, scene_w - 1))
    px2 = max(0, min(px2, scene_w - 1))
    py1 = max(0, min(py1, scene_h - 1))
    py2 = max(0, min(py2, scene_h - 1))
    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def decode_qwen_response(
    body: str, *, scene_w: int, scene_h: int, allowed_labels: set[str],
) -> list[MatchRow]:
    """Parse a Qwen3-VL match response body and return MatchRows.

    Drops rows whose label is not in `allowed_labels` (defensive against
    hallucinated labels). Clamps boxes to image bounds, drops degenerate
    ones, clamps confidence to [0, 1]."""
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
        bbox = _decode_bbox_normalized(det.get('box_2d'), scene_w, scene_h)
        if bbox is None:
            continue
        try:
            conf = float(det.get('confidence', 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(conf, 1.0))
        rows.append(MatchRow(label=label, bbox=bbox, conf=conf))
    return rows


def _qwen_system_prompt(labels: list[str]) -> str:
    label_list = ', '.join(f'"{lbl}"' for lbl in labels)
    return (
        "You are a visual-grounding assistant for a service robot. The "
        "user provides one SCENE image followed by "
        f"{len(labels)} REFERENCE images, each captioned with a label "
        f"from this set: [{label_list}]. Find every visible instance of "
        "any reference item in the scene and return bounding boxes. "
        "Coordinates 'box_2d' are [x1, y1, x2, y2] normalized to 0-1000 "
        "over the SCENE image dimensions, where (0,0) is the top-left "
        "and (1000,1000) is the bottom-right. The 'label' field must be "
        f"exactly one of [{label_list}]. Confidence is a subjective "
        "match score in [0.0, 1.0]. If no reference item is visible, "
        "return detections=[]. Output JSON only, with no commentary or "
        "markdown fences."
    )


class QwenMatchClient:
    """Qwen3-VL match client (DashScope OpenAI-compatible endpoint)."""

    @staticmethod
    def _resolve_api_key() -> str | None:
        """Resolve the DashScope key from `os.environ` only.

        `.env` discovery happens at module import time (see `load_dotenv()`
        below the class), so pytest's `monkeypatch.delenv` fully controls
        the environment without `load_dotenv()` re-populating slots."""
        for name in _QWEN_KEY_NAMES:
            val = os.environ.get(name)
            if val:
                return val
        return None

    def __init__(
        self,
        model: str = '',
        base_url: str = '',
    ):
        self._api_key: str | None = self._resolve_api_key()
        if not self._api_key:
            raise RuntimeError(
                'DashScope API key not found in env (looked for '
                f'{_QWEN_KEY_NAMES})'
            )
        self._model = model or _qwen_default_model()
        self._base_url = base_url or _QWEN_DEFAULT_BASE_URL

    def match_batch(
        self,
        scene_bgr: np.ndarray,
        refs: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> list[MatchRow]:
        if not refs:
            return []

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
                             'content': _qwen_system_prompt(labels)},
                            {'role': 'user', 'content': content},
                        ],
                        response_format={'type': 'json_object'},
                    )
                    raw = completion.choices[0].message.content or ''
                    return decode_qwen_response(
                        raw,
                        scene_w=w, scene_h=h,
                        allowed_labels=allowed_labels,
                    )
                except Exception as exc:    # noqa: BLE001
                    if logger is not None:
                        logger.warning(
                            f'Qwen match attempt {attempt+1}/'
                            f'{max_retries} failed: {exc}'
                        )
            return []
        finally:
            try:
                client.close()
            except Exception:    # noqa: BLE001
                pass


def build_match_client(provider: str, **opts) -> MatchClient:
    if provider == 'qwen':
        return QwenMatchClient(**opts)
    if provider == 'gemini':
        # Gemini backend implemented in Task 6.
        from .vlm_match_client_gemini import GeminiMatchClient
        return GeminiMatchClient(**opts)
    raise ValueError(f'Unknown provider: {provider!r}')


try:
    from openai import OpenAI
except ImportError:    # pragma: no cover
    OpenAI = None    # type: ignore[assignment]
