"""Person geometry for the generalist's open-vocab paths.

YOLO-World and VLM boxes are segmented by box-prompted MobileSAM. On person
boxes clipped by the image edge, SAM often segments the wall behind the
person, so the 3D centroid lands on the wall (GPSR sim 2026-10-04/05, report
``tinker-sim/6.0.1/docs/issue_reports/2026-10-08-gpsr-nav-bug-report.md``
section 4). For person boxes the generalist instead takes the mask of the
COCO ``person`` instance that the pretrained YOLO-seg model finds in the
same box. Everything here is pure numpy so it can be tested without a node
or a model.
"""

from __future__ import annotations

import re

PERSON_WORDS = frozenset({
    'person', 'persons', 'people', 'man', 'men', 'woman', 'women',
    'human', 'humans', 'guest', 'guests', 'child', 'children', 'kid',
    'kids', 'boy', 'boys', 'girl', 'girls', 'lady', 'ladies',
    'gentleman', 'gentlemen', 'operator', 'individual',
})

# A person word names the target only when it comes before any of these:
# "person next to the cup" is a person, "cup next to the person" is a cup.
RELATION_WORDS = frozenset({
    'with', 'next', 'near', 'beside', 'by', 'on', 'in', 'at', 'of',
    'behind', 'under', 'above', 'below', 'from', 'for', 'to', 'belonging',
    'between', 'beneath', 'inside', 'outside', 'over', 'against', 'around',
    'along', 'across',
})


def is_person_phrase(text: str) -> bool:
    """True when ``text`` names a person rather than an object near one."""
    tokens = re.findall(r'[a-z]+', (text or '').lower())
    for i, tok in enumerate(tokens):
        if tok in RELATION_WORDS:
            return False
        if tok in PERSON_WORDS:
            # "person's bag": the possessive person word modifies the next noun.
            return not (i + 1 < len(tokens) and tokens[i + 1] == 's')
    return False


def person_class_id(names) -> int | None:
    """Index of the ``person`` class in an Ultralytics ``model.names``."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for cls_id, name in items:
        if name == 'person':
            return int(cls_id)
    return None
