"""ptbench.replay — replay a recorded rosbag through Tinker's person tracker.

Three thin layers:

* :mod:`ptbench.replay.bag_io` — read a rosbag2 directory and yield time-synced
  color + depth + intrinsics (:class:`~ptbench.replay.bag_io.FrameBundle`).
  Pure-ish and unit-tested against a synthetic bag.
* :mod:`ptbench.replay.runner` — produce a prediction stream from a bag, either
  offline (drive ``vision_track``'s ``YOLOTracker`` directly) or via a live
  ``/track_person`` action server. Heavy imports are deferred into the
  functions so importing this package never requires the tracker / ROS.
* :mod:`ptbench.replay.score_cli` — CLI glue: load GT, run a backend, align,
  compute metrics, score, print/dump. The align→metrics→score wiring is the
  testable :func:`~ptbench.replay.score_cli.score_preds`.
"""
