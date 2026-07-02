"""Prompt-content locks for the 2026-07-02 five-slot feature cut.

Spec: docs/superpowers/specs/2026-07-02-feature-5slot-cut-design.md.
The description is one free-text string end-to-end; these tests pin the
five requested attributes (hair color [+optional length], gender,
approximate age, glasses, upper-body wear) and the removal of the old
multi-clothing / facial-features asks, without asserting the full prompt
verbatim (wording may be tuned; slots may not drift).
"""
from kimi_api.feature_recognition import FEATURE_SYS_PROMPT
from kimi_api.feature_matching import build_matching_sys_prompt


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


def test_extraction_prompt_forces_binary_gender_and_allows_are():
    # 2026-07-02 replay finding: ambiguous crops yielded "They is/are
    # gender-neutral..." — ungrammatical and useless for matching. The
    # prompt must force a male/female commitment and permit "are" for
    # pronoun-verb agreement.
    assert 'male or female' in FEATURE_SYS_PROMPT
    assert 'gender-neutral' in FEATURE_SYS_PROMPT  # named in order to ban it
    assert '"are"' in FEATURE_SYS_PROMPT


def test_matching_prompt_text_only_cites_five_slots_not_body_shape():
    p = build_matching_sys_prompt(5, 3, True)
    for term in ('gender', 'hair color', 'glasses', 'apparent age', 'upper-body'):
        assert term in p, f'missing evidence term: {term}'
    assert 'body shape' not in p
    assert 'posture' not in p
    # Structural contract unchanged:
    assert '(0..4)' in p
    assert 'length 3' in p
    assert 'EVERY description MUST be matched' in p
    assert 'NEVER use -1' in p


def test_matching_prompt_image_mode_cites_five_slots_not_posture():
    p = build_matching_sys_prompt(4, 2, False)
    for term in ('gender', 'hair color', 'glasses', 'apparent age', 'upper-body'):
        assert term in p, f'missing evidence term: {term}'
    assert 'body shape' not in p
    assert 'posture' not in p
    # Structural contract unchanged:
    assert 'SAME' in p
    assert 'length 2' in p
    assert 'EVERY reference MUST be matched' in p
    assert 'tiebreaker hint' in p
    assert 'NEVER use -1' in p
