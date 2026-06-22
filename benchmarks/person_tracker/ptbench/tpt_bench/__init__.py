"""TPT-Bench offline scorer for Tinker's vision_track YOLO tracker.

Self-contained sub-package (does NOT import :mod:`ptbench.common`). Runs the
``vision_track`` single-target person tracker against the TPT-Bench dataset
(robot-egocentric, LaSOT-style single-target person tracking) as an external
regression smoke-test.

See ``DOWNLOAD.md`` for the dataset, its annotation layout, and the run
command. Pure-logic modules (:mod:`.dataset`, :mod:`.metrics`) are unit-tested;
:mod:`.runner` defers the heavy tracker import.
"""
