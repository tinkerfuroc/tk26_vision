"""Precedence tests for vision_util.vlm_models (pure os.environ, no dotenv)."""
import pytest

from vision_util import vlm_models as vm


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in vm.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_are_the_legacy_literals():
    assert vm.vision_vlm_model() == 'google/gemini-2.5-pro'
    assert vm.vision_flash_model() == 'google/gemini-2.5-flash'
    assert vm.vision_qwen_model() == 'qwen3-vl-plus'


def test_vision_keys_win(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'openai/gpt-5.6-luna')
    monkeypatch.setenv('FLASH_MODEL', 'google/gemini-3.5-flash-lite')
    monkeypatch.setenv('VISION_VLM_MODEL', 'google/gemini-3.1-pro-preview')
    monkeypatch.setenv('VISION_VLM_FLASH_MODEL', 'google/gemini-3.7-flash')
    monkeypatch.setenv('VISION_QWEN_MODEL', 'qwen3.7-plus')
    assert vm.vision_vlm_model() == 'google/gemini-3.1-pro-preview'
    assert vm.vision_flash_model() == 'google/gemini-3.7-flash'
    assert vm.vision_qwen_model() == 'qwen3.7-plus'


def test_gpsr_keys_are_the_middle_fallback(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'openai/gpt-5.6-luna')
    monkeypatch.setenv('FLASH_MODEL', 'google/gemini-3.5-flash-lite')
    assert vm.vision_vlm_model() == 'openai/gpt-5.6-luna'
    assert vm.vision_flash_model() == 'google/gemini-3.5-flash-lite'
    assert vm.vision_qwen_model() == 'qwen3-vl-plus'   # no GPSR key for qwen


def test_empty_and_whitespace_count_as_unset(monkeypatch):
    monkeypatch.setenv('VISION_VLM_MODEL', '   ')
    monkeypatch.setenv('LLM_MODEL', '')
    assert vm.vision_vlm_model() == 'google/gemini-2.5-pro'
    monkeypatch.setenv('VISION_QWEN_MODEL', ' qwen3-vl-flash ')
    assert vm.vision_qwen_model() == 'qwen3-vl-flash'   # stripped
