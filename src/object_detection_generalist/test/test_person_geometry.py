"""Unit tests for person_geometry (pure numpy; no ROS node, no model).

Run:
    cd /home/tinker/tk25_ws && bash -c 'source /opt/ros/humble/setup.bash && \
        source install/setup.bash && \
        source src/tk26_vision/.venv-vision-main/bin/activate && \
        python -m pytest src/tk26_vision/src/object_detection_generalist/test/test_person_geometry.py -v'
"""
import pytest

from object_detection_generalist.person_geometry import (
    is_person_phrase,
    person_class_id,
)


@pytest.mark.parametrize('text', [
    'person',
    'Person',
    'PERSON WAVING',
    'person pointing to the left',
    'the person pointing to the left',
    'standing person',
    'person standing',
    'waving person',
    'a man in a red shirt',
    'woman',
    'people',
    'guest',
    'child',
    'person holding a cup',
    'person sitting on the sofa',
])
def test_person_phrases(text):
    assert is_person_phrase(text)


@pytest.mark.parametrize('text', [
    '',
    '   ',
    'red bowl',
    'chair',
    'cup next to the person',
    'cup held by the person',
    "the person's bag",
    "person's bag",
    'bag of the person',
    'bowl on the table near a man',
    'mannequin',
    'manipulator',
    'humanoid robot',
])
def test_non_person_phrases(text):
    assert not is_person_phrase(text)


def test_person_class_id_dict_names():
    assert person_class_id({0: 'person', 1: 'bicycle'}) == 0


def test_person_class_id_list_names():
    assert person_class_id(['cup', 'person']) == 1


def test_person_class_id_missing():
    assert person_class_id({0: 'cup'}) is None
