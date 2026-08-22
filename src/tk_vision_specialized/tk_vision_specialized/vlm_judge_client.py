"""Provider-agnostic judge client for object_match_all conflict resolution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from dotenv import load_dotenv
from vision_util.vlm_models import vision_vlm_model, vision_qwen_model

from ._vlm_common import strip_fences, encode_data_url


# Load .env once at module-import time so pytest's monkeypatch.delenv is
# authoritative after first construction. Same defensive pattern Tasks 5
# and 6 use; matches kimi_api/_env.py.
load_dotenv(override=False)


_QWEN_DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
_QWEN_KEY_NAMES = ('DASHCOPE_API_KEY', 'DASHSCOPE_API_KEY')


def _qwen_default_model() -> str:
    return vision_qwen_model()


_GEMINI_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'


def _gemini_default_model() -> str:
    return vision_vlm_model()


@dataclass(frozen=True)
class JudgeChoice:
    label: str    # one of competing labels, or '' for abstain
    conf: float


class JudgeClient(Protocol):
    def choose(
        self,
        crop_bgr: np.ndarray,
        competing: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> JudgeChoice | None: ...


def decode_judge_response(
    body: str, *, competing_labels: set[str],
) -> JudgeChoice | None:
    """Parse a judge response body.

    Returns None on JSON parse failure (caller will fall back).
    Returns JudgeChoice(label='', conf=0.0) on abstain or hallucinated label.
    """
    try:
        parsed = json.loads(strip_fences(body))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None

    label = parsed.get('label')
    if label is None or label == '' or not isinstance(label, str):
        return JudgeChoice(label='', conf=0.0)
    if label not in competing_labels:
        return JudgeChoice(label='', conf=0.0)

    try:
        conf = float(parsed.get('confidence', 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(conf, 1.0))
    return JudgeChoice(label=label, conf=conf)


def _judge_system_prompt(labels: list[str]) -> str:
    label_list = ', '.join(f'"{lbl}"' for lbl in labels)
    return (
        "You are a tie-breaking visual-grounding assistant. The user "
        "provides one SCENE CROP image followed by N REFERENCE images, "
        "each captioned with a label from this set: "
        f"[{label_list}]. Choose the single label that best matches the "
        "object in the scene crop. If none of the references match what "
        "is in the crop, return label = null to abstain. The 'label' "
        "field must be exactly one of the input labels or null. "
        "Confidence is your match score in [0.0, 1.0]. Output JSON only, "
        "with no commentary or markdown fences."
    )


class _BaseJudgeClient:
    """Shared HTTP plumbing; subclasses set provider-specific config."""

    _api_key: str
    _model: str
    _base_url: str

    def choose(
        self,
        crop_bgr: np.ndarray,
        competing: Sequence[tuple[str, str]],
        *, timeout_s: float, max_retries: int, logger=None,
    ) -> JudgeChoice | None:
        if not competing:
            return None
        labels = [label for label, _url in competing]
        allowed = set(labels)

        crop_url = encode_data_url(crop_bgr)
        content: list[dict] = [
            {'type': 'image_url', 'image_url': {'url': crop_url}},
        ]
        for label, url in competing:
            content.append(
                {'type': 'image_url', 'image_url': {'url': url}},
            )
        content.append({
            'type': 'text',
            'text': (
                'Image 1 is the scene crop. The remaining images are '
                'reference photos, in order: '
                + ', '.join(
                    f'image {i+2} = "{lbl}"'
                    for i, lbl in enumerate(labels)
                )
                + '. Choose the best matching label or return null to '
                'abstain.'
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
                             'content': _judge_system_prompt(labels)},
                            {'role': 'user', 'content': content},
                        ],
                        response_format={'type': 'json_object'},
                    )
                    raw = completion.choices[0].message.content or ''
                    return decode_judge_response(
                        raw, competing_labels=allowed,
                    )
                except Exception as exc:    # noqa: BLE001
                    if logger is not None:
                        logger.warning(
                            f'Judge attempt {attempt+1}/{max_retries} '
                            f'failed: {exc}'
                        )
            return None
        finally:
            try:
                client.close()
            except Exception:    # noqa: BLE001
                pass


class QwenJudgeClient(_BaseJudgeClient):
    def __init__(self, model: str = '', base_url: str = ''):
        # load_dotenv ran at module-import time; read os.environ here
        self._api_key = ''
        for name in _QWEN_KEY_NAMES:
            val = os.environ.get(name)
            if val:
                self._api_key = val
                break
        if not self._api_key:
            raise RuntimeError(
                'DashScope API key not found in env '
                f'(looked for {_QWEN_KEY_NAMES})'
            )
        self._model = model or _qwen_default_model()
        self._base_url = base_url or _QWEN_DEFAULT_BASE_URL


class GeminiJudgeClient(_BaseJudgeClient):
    def __init__(self, model: str = '', base_url: str = ''):
        # load_dotenv ran at module-import time; read os.environ here
        self._api_key = os.environ.get('OPENROUTER_API_KEY', '')
        if not self._api_key:
            raise RuntimeError(
                'OPENROUTER_API_KEY not found in env '
                '(required for Gemini provider)'
            )
        self._model = model or _gemini_default_model()
        self._base_url = base_url or _GEMINI_DEFAULT_BASE_URL


def build_judge_client(provider: str, **opts) -> JudgeClient:
    if provider == 'qwen':
        return QwenJudgeClient(**opts)
    if provider == 'gemini':
        return GeminiJudgeClient(**opts)
    raise ValueError(f'Unknown provider: {provider!r}')


try:
    from openai import OpenAI
except ImportError:    # pragma: no cover
    OpenAI = None    # type: ignore[assignment]
