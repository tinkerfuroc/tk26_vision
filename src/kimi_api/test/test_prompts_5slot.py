"""Prompt-content locks for the 2026-07-02 five-slot feature cut.

Spec: docs/superpowers/specs/2026-07-02-feature-5slot-cut-design.md.
The description is one free-text string end-to-end; these tests pin the
five requested attributes (hair color [+optional length], gender,
approximate age, glasses, upper-body wear) and the removal of the old
multi-clothing / facial-features asks, without asserting the full prompt
verbatim (wording may be tuned; slots may not drift).
"""
from kimi_api.feature_recognition import FEATURE_SYS_PROMPT


def test_extraction_prompt_requests_exactly_five_slots():
    for term in (
        'gender',
        'age in years',
        'hair color',
        'glasses',
        'upper-body garment',
    ):
        assert term in FEATURE_SYS_PROMPT, f'missing slot ask: {term}'
    # Old prompt asks that must be gone:
    assert 'pieces of clothing' not in FEATURE_SYS_PROMPT
    assert 'facial features' not in FEATURE_SYS_PROMPT


def test_extraction_prompt_keeps_spoken_sentence_template():
    # The Receptionist speaks this sentence verbatim (customNodes.py:148).
    assert 'years old' in FEATURE_SYS_PROMPT
    assert 'wearing a [color] [garment]' in FEATURE_SYS_PROMPT
    # Age-in-words convention retained:
    assert 'not numeric numerals' in FEATURE_SYS_PROMPT


def test_extraction_prompt_excludes_everything_else():
    assert 'Do not mention lower-body clothing' in FEATURE_SYS_PROMPT
