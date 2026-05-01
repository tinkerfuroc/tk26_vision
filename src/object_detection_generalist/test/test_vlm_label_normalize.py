"""Unit tests for VLM label → prompt-class normalization.

Pure-Python coverage of `GeneralistDetectionNode._parse_prompt_classes` and
`_normalize_vlm_label`. The helpers are `@staticmethod`, so we never
instantiate the node — the heavy ROS / torch / ultralytics imports happen
once at module import time, but no rclpy.init or CUDA context is needed.

Run:
    cd /home/tinker/tk25_ws && \
        python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_vlm_label_normalize.py -v
"""

import pytest

from object_detection_generalist.generalist_node import GeneralistDetectionNode

parse_prompt = GeneralistDetectionNode._parse_prompt_classes
normalize = GeneralistDetectionNode._normalize_vlm_label


@pytest.mark.parametrize(
    'prompt, expected',
    [
        ('apple . banana . pear', ['apple', 'banana', 'pear']),
        ('person', ['person']),
        ('ice cream . cone', ['ice cream', 'cone']),
        ('  apple  .  banana  ', ['apple', 'banana']),
        ('   ', ['']),
        ('apple', ['apple']),
    ],
)
def test_parse_prompt_classes(prompt, expected):
    assert parse_prompt(prompt) == expected


@pytest.mark.parametrize(
    'prompt, label, expected',
    [
        # Plan-listed cases
        ('apple . banana . pear', 'green apple', 'apple'),
        ('apple . banana . pear', 'yellow banana, ripe', 'banana'),
        ('ice cream . cone', 'chocolate ice cream cone', 'ice cream'),
        ('apple . banana', 'orange', 'apple . banana'),
        ('person', 'person in red shirt', 'person'),
        ('apple . banana . pear', '', 'apple . banana . pear'),
        # Specificity-over-ordering: 'red apple' beats 'apple' on a label that
        # contains both, even though 'apple' comes first in the prompt.
        ('apple . red apple', 'red apple, ripe', 'red apple'),
        # Single-word prompt, label has no overlap → fall back.
        ('person', 'orange chair', 'person'),
        # Punctuation in label tokenizes correctly.
        ('apple . banana', 'apple, red, shiny.', 'apple'),
        # Tie on token count → first-occurrence wins (banana not pear).
        ('banana . pear', 'banana pear smoothie', 'banana'),
    ],
)
def test_normalize_vlm_label(prompt, label, expected):
    classes = parse_prompt(prompt)
    assert normalize(label, classes, prompt) == expected


def test_normalize_only_punctuation_label_falls_back():
    """A label with no alphanumeric tokens cannot match anything."""
    prompt = 'apple . banana'
    classes = parse_prompt(prompt)
    assert normalize('...,,!', classes, prompt) == prompt
