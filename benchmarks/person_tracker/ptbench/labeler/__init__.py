"""ptbench.labeler — offline GT-labeling tool for person-tracker clips.

Splits into:

- ``label_io``  — PURE, unit-tested: a small self-contained rosbag2 reader plus
  the box-propagation / GtClip-assembly logic.
- ``label_cli`` — thin cv2 UI loop (not unit-tested; cv2 imported lazily so this
  package imports cleanly headless).
"""
