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
