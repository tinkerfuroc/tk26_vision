"""Shared env helpers for kimi_api nodes.

Loads `.env` via python-dotenv if available; otherwise falls back to plain
`os.environ`. Provides a small helper that fails fast if the API key is not
set, so each node raises a clear error at __init__ rather than failing on the
first API call.
"""

import os


def load_env():
    """Best-effort load of a `.env` file from CWD and parents."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except ImportError:
        # python-dotenv is optional — env vars can still be exported by a wrapper
        pass


def require_api_key() -> str:
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise RuntimeError(
            'OPENROUTER_API_KEY is not set. Copy kimi_api/.env.example to .env '
            'and fill it in, or export the variable in your shell before '
            '`ros2 run kimi_api ...`.'
        )
    return key


def base_url() -> str:
    return os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')


def default_model() -> str:
    return os.environ.get('LLM_MODEL', 'google/gemini-2.5-pro')


def default_flash_model() -> str:
    return os.environ.get('FLASH_MODEL', 'google/gemini-2.5-flash')


def gemini_api_key() -> str:
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise RuntimeError(
            'GEMINI_API_KEY is not set. Add it to .env at the workspace root '
            'or export it in your shell before running the direct-Gemini path.'
        )
    return key


def dashscope_base_url() -> str:
    """DashScope OpenAI-compatible endpoint.

    Defaults to the China-mainland (bailian) host. Override with
    `DASHSCOPE_BASE_URL` for the international account, e.g.
    `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`.
    """
    return os.environ.get(
        'DASHSCOPE_BASE_URL',
        'https://dashscope.aliyuncs.com/compatible-mode/v1',
    )


def require_dashscope_api_key() -> str:
    """Return the DashScope API key, failing fast if unset.

    Accepts the correct spelling `DASHSCOPE_API_KEY` first, then the
    legacy/typo'd `DASHCOPE_API_KEY` that older `.env` files carried, so
    existing setups keep working without an edit.
    """
    key = (
        os.environ.get('DASHSCOPE_API_KEY')
        or os.environ.get('DASHCOPE_API_KEY')
    )
    if not key:
        raise RuntimeError(
            'DASHSCOPE_API_KEY is not set. Add it to .env at the workspace '
            'root (or export it) before using a dashscope/ VLM model.'
        )
    return key


_VALID_QWEN_BACKENDS = ('dashscope', 'openrouter')

# Provisional default for qwen_api_backend='openrouter' — OpenRouter does not
# carry the exact 'qwen3-vl-plus' slug this codebase defaults to on DashScope.
# This is the SAFER open-weight starting bet (same Qwen3-VL family as the
# calibrated bbox decoder in object_detection_generalist/vlm_bbox.py), NOT a
# verified final choice. Do not rely on this for a competition run until the
# benchmark task in docs/superpowers/specs/2026-07-03-qwen-openrouter-
# dashscope-toggle-design.md §"Default OpenRouter model" has actually run
# (image modality + known-position bbox format + regional latency) and this
# constant has been updated accordingly.
_OPENROUTER_QWEN_DEFAULT_MODEL = 'qwen/qwen3-vl-32b-instruct'

_DASHSCOPE_DEFAULT_MODEL = 'qwen3-vl-plus'


def resolve_qwen_target(
    backend: str,
    model_param_value: str,
    base_url_override: str = '',
) -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for a Qwen call on the given backend.

    `backend` is 'dashscope' or 'openrouter' — the caller's qwen_api_backend
    ROS param. `model_param_value` is the caller's own qwen-model ROS param:
    '' means "use this backend's default model"; any non-empty value is
    honored verbatim on either backend (never silently rewritten — an
    explicit value the operator set for the "wrong" backend raises instead
    of being swapped, since OpenRouter ids contain '/' and DashScope ids
    don't, and mixing them up is very likely a config mistake worth
    surfacing loudly). `base_url_override`, if non-empty, always wins over
    the backend's own base URL — it does not change which API key is
    required.

    Raises RuntimeError on: invalid backend, missing required key for the
    selected backend, or a model-id shape mismatch (see above).
    """
    if backend not in _VALID_QWEN_BACKENDS:
        raise RuntimeError(
            f'Invalid qwen_api_backend {backend!r}; expected one of '
            f'{_VALID_QWEN_BACKENDS}.'
        )

    model = model_param_value or ''

    if backend == 'dashscope':
        resolved_model = model or _DASHSCOPE_DEFAULT_MODEL
        if '/' in resolved_model:
            raise RuntimeError(
                f"qwen_api_backend='dashscope' but model {resolved_model!r} "
                "looks like an OpenRouter id (contains '/'). Pass a bare "
                "DashScope model id, or set qwen_api_backend='openrouter'."
            )
        api_key = require_dashscope_api_key()
        resolved_base_url = base_url_override or dashscope_base_url()
        return resolved_base_url, api_key, resolved_model

    # openrouter
    resolved_model = model or _OPENROUTER_QWEN_DEFAULT_MODEL
    if '/' not in resolved_model:
        raise RuntimeError(
            f"qwen_api_backend='openrouter' but model {resolved_model!r} "
            "looks like a bare DashScope id (no '/'). Pass an OpenRouter "
            "'org/name' id, or set qwen_api_backend='dashscope'."
        )
    api_key = require_api_key()
    resolved_base_url = base_url_override or base_url()
    return resolved_base_url, api_key, resolved_model
