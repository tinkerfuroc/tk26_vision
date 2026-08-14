"""Wiring tests for object_match_all_server client construction."""

from tk_vision_specialized import object_match_all_server as omas


def test_judge_client_receives_vlm_base_url(monkeypatch):
    """The judge client must share the vlm_base_url seam with the match
    client; otherwise object_match_all falls back to the real OpenRouter
    endpoint when vlm_base_url points at a local provider."""
    seen_match = {}
    seen_judge = {}

    def fake_match(provider, **opts):
        seen_match.update(opts)
        return 'match'

    def fake_judge(provider, **opts):
        seen_judge.update(opts)
        return 'judge'

    monkeypatch.setattr(omas, 'build_match_client', fake_match)
    monkeypatch.setattr(omas, 'build_judge_client', fake_judge)

    match_client, judge_client = omas._build_vlm_clients(
        provider='qwen',
        judge_provider='qwen',
        model='m',
        judge_model='jm',
        base_url='http://127.0.0.1:18080',
    )

    assert seen_match.get('base_url') == 'http://127.0.0.1:18080'
    assert seen_judge.get('base_url') == 'http://127.0.0.1:18080'
    assert match_client == 'match'
    assert judge_client == 'judge'


def test_judge_client_empty_base_url_keeps_default(monkeypatch):
    """Empty vlm_base_url must not change the existing default behaviour."""
    seen_judge = {}

    def fake_judge(provider, **opts):
        seen_judge.update(opts)
        return 'judge'

    monkeypatch.setattr(omas, 'build_match_client', lambda *a, **k: 'match')
    monkeypatch.setattr(omas, 'build_judge_client', fake_judge)

    omas._build_vlm_clients(
        provider='qwen', judge_provider='qwen', model='m', judge_model='jm',
        base_url='',
    )

    assert seen_judge.get('base_url') == ''
