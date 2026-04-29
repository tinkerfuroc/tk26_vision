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
