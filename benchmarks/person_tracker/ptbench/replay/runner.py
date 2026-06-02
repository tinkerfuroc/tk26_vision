"""Produce a prediction stream from a bag by running the real person tracker.

Two backends, each returning ``(list[PredFrame], throughput_hz)``:

* :func:`run_offline` — drives ``vision_track``'s ``YOLOTracker`` directly over
  the bag's color+depth frames (no ROS graph, no live server). This is the
  primary backend for CI-style offline scoring.
* :func:`run_action` — replays the bag onto a **live** ``/track_person`` action
  server and collects its feedback. Requires a running server + ROS graph.

Both backends defer their heavy/optional dependencies (``vision_track`` + the
YOLO model / ``rclpy`` action client + the action types / ``cv2``) **into the
function body**, so::

    import ptbench.replay.runner

does not require ``vision_track``, the YOLO model, or a live ``/track_person``
server — those errors surface only when you actually call a backend. Note that
``bag_io`` imports ROS message libraries (``rosbag2_py`` / ``rclpy`` /
``sensor_msgs``) at module load, so the ROS environment must still be sourced to
import this module; it is only the tracker/model/server deps that are deferred.

Requirements to actually run either backend:

* Source the colcon workspace so ``vision_track`` / ``tinker_vision_msgs_26``
  are importable::

      source /home/tinker/tk25_ws/install/setup.bash

* :func:`run_offline` additionally needs the YOLO model weights (bundled in the
  ``.venv-vision-main`` install of ``vision_track``).
* :func:`run_action` additionally needs a live ``person_track_server``::

      ros2 run vision_track person_track_server
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from ..common.align import PredFrame
from ..common.geometry import centroid_from_bbox_depth
from ..common.schema import GtClip
from .bag_io import read_synced_frames


def run_offline(
    bag_dir,
    gt_clip: GtClip,
    *,
    imgsz: int = 1280,
    conf: float = 0.5,
) -> Tuple[List[PredFrame], float]:
    """Run ``vision_track``'s YOLOTracker over a bag and emit a PredFrame stream.

    The tracker is force-initialised on the first synced frame (target class
    ``'person'``), then ``update`` is called on every subsequent frame. Each
    iteration yields one :class:`~ptbench.common.align.PredFrame`:

    * tracker returns ``None`` (or a result with no bbox) → ``target_lost=True``,
      ``target_track_id=-1``, ``point_xyz=None``.
    * otherwise → ``target_lost=False``, ``target_track_id=result.track_id``,
      ``point_xyz`` = :func:`centroid_from_bbox_depth` over the result's bbox
      (and segmentation mask if the tracker produced one). A bbox whose depth
      yields no valid centroid degrades to ``point_xyz=None`` but keeps
      ``target_lost=False`` (the tracker still has a 2D lock).

    Args:
        bag_dir: rosbag2 directory.
        gt_clip: GT clip — only its topic names are used here, to read the bag.
        imgsz: tracker inference image size.
        conf: tracker detection confidence threshold.

    Returns:
        ``(preds, throughput_hz)`` where ``preds`` is one PredFrame per synced
        frame and ``throughput_hz`` is ``n / wall_seconds`` over the update loop
        (``0.0`` if zero frames).

    Raises:
        ImportError: if ``vision_track`` cannot be imported (workspace not
            sourced). Raised on call, not at module import.
    """
    # Deferred heavy import: only when actually running.
    import cv2  # noqa: deferred — only needed at run time
    from vision_track.track_yolo import YOLOTracker

    tracker = YOLOTracker(confidence_threshold=conf, inference_size=imgsz)

    preds: List[PredFrame] = []
    initialized = False
    n = 0
    t_start = time.perf_counter()

    for bundle in read_synced_frames(
        bag_dir,
        gt_clip.color_topic,
        gt_clip.depth_topic,
        gt_clip.camera_info_topic,
    ):
        rgb = cv2.cvtColor(bundle.color_bgr, cv2.COLOR_BGR2RGB)

        if not initialized:
            tracker.initialize_tracking(rgb, target_class="person")
            initialized = True

        result = tracker.update(rgb)
        n += 1

        if result is None or getattr(result, "bbox", None) is None:
            preds.append(
                PredFrame(
                    t_ns=bundle.t_ns,
                    target_lost=True,
                    target_track_id=-1,
                    point_xyz=None,
                )
            )
            continue

        centroid = centroid_from_bbox_depth(
            bundle.depth_mm,
            bundle.K,
            result.bbox,
            mask=getattr(result, "mask", None),
        )
        preds.append(
            PredFrame(
                t_ns=bundle.t_ns,
                target_lost=False,
                target_track_id=int(result.track_id),
                point_xyz=centroid,
            )
        )

    elapsed = time.perf_counter() - t_start
    throughput_hz = (n / elapsed) if (n and elapsed > 0) else 0.0
    return preds, throughput_hz


def run_action(
    bag_dir,
    gt_clip: GtClip,
    *,
    target_point_topic: str = "",
    timeout_s: float = 120.0,
) -> Tuple[List[PredFrame], float]:
    """Replay a bag onto a LIVE ``/track_person`` server and collect feedback.

    This is the end-to-end backend: instead of driving the tracker class in
    process, it exercises the real ROS action server exactly as the robot would.
    It requires a running ``person_track_server`` and a ROS graph.

    Approach (implemented best-effort; see the deferred imports below):

    1. Spin up an ``rclpy`` node that **republishes** the bag's color, depth and
       camera_info messages onto ``gt_clip``'s topics (so the live server, which
       subscribes to those topics, sees the recorded scene). Publishing is paced
       to roughly the recorded inter-frame spacing.
    2. Send a ``tinker_vision_msgs_26/action/TrackPerson`` goal with
       ``target_frame='none'`` (camera frame) and all ``return_*`` flags False
       (we only need feedback, not the echoed images);
       ``target_point_topic`` is forwarded so the server can publish points.
    3. Collect each ``Feedback`` into a :class:`~ptbench.common.align.PredFrame`:
       ``feedback.target_lost``, ``feedback.target_track_id``,
       ``feedback.target_position.point`` → ``point_xyz``, and ``t_ns`` from
       ``feedback.target_position.header.stamp``.

    Args:
        bag_dir: rosbag2 directory to replay.
        gt_clip: GT clip; supplies the topic names to republish on.
        target_point_topic: forwarded into the goal's ``target_point_topic``
            field (empty → server default).
        timeout_s: overall wall-clock budget for the replay + collection.

    Returns:
        ``(preds, throughput_hz)`` where ``preds`` is one PredFrame per feedback
        message and ``throughput_hz`` is ``len(preds) / wall_seconds``.

    Raises:
        ImportError / RuntimeError: if ``rclpy`` / the action type are not
            importable, or no live server is reachable. Raised on call.
    """
    # Deferred imports: rclpy + action + message types only at run time.
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.serialization import deserialize_message

    import rosbag2_py
    from sensor_msgs.msg import CameraInfo, Image
    from tinker_vision_msgs_26.action import TrackPerson

    preds: List[PredFrame] = []

    def _stamp_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    rclpy_was_running = rclpy.ok()
    if not rclpy_was_running:
        rclpy.init()

    node: Optional[Node] = None
    t_start = time.perf_counter()
    try:
        node = rclpy.create_node("ptbench_replay_action")

        color_pub = node.create_publisher(Image, gt_clip.color_topic, 10)
        depth_pub = node.create_publisher(Image, gt_clip.depth_topic, 10)
        info_pub = node.create_publisher(
            CameraInfo, gt_clip.camera_info_topic, 10
        )

        client = ActionClient(node, TrackPerson, "/track_person")
        if not client.wait_for_server(timeout_sec=min(10.0, timeout_s)):
            raise RuntimeError(
                "no /track_person action server reachable — start "
                "`ros2 run vision_track person_track_server` first"
            )

        def _on_feedback(fb_msg) -> None:
            fb = fb_msg.feedback
            point = fb.target_position.point
            xyz = (
                None
                if fb.target_lost
                else (float(point.x), float(point.y), float(point.z))
            )
            preds.append(
                PredFrame(
                    t_ns=_stamp_ns(fb.target_position.header.stamp),
                    target_lost=bool(fb.target_lost),
                    target_track_id=int(fb.target_track_id),
                    point_xyz=xyz,
                )
            )

        goal = TrackPerson.Goal()
        goal.target_frame = "none"
        goal.target_point_topic = target_point_topic
        goal.return_rgb_img = False
        goal.return_depth_img = False
        goal.return_segment = False
        goal.debug = False

        send_future = client.send_goal_async(goal, feedback_callback=_on_feedback)
        rclpy.spin_until_future_complete(node, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("/track_person rejected or did not respond to goal")

        # Republish the bag, paced to recorded inter-frame spacing, spinning the
        # node between sends so feedback callbacks fire.
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=""),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )
        topics = {
            gt_clip.color_topic: (Image, color_pub),
            gt_clip.depth_topic: (Image, depth_pub),
            gt_clip.camera_info_topic: (CameraInfo, info_pub),
        }
        prev_bag_t: Optional[int] = None
        while reader.has_next() and (time.perf_counter() - t_start) < timeout_s:
            topic, raw, bag_t = reader.read_next()
            if topic not in topics:
                continue
            msg_type, pub = topics[topic]
            if prev_bag_t is not None:
                dt = max(0.0, (bag_t - prev_bag_t) / 1e9)
                # Pace, but spin so callbacks run during the wait.
                wait_end = time.perf_counter() + dt
                while time.perf_counter() < wait_end:
                    rclpy.spin_once(node, timeout_sec=0.005)
            prev_bag_t = bag_t
            pub.publish(deserialize_message(raw, msg_type))
            rclpy.spin_once(node, timeout_sec=0.0)

        # Cancel the goal and drain any trailing feedback.
        cancel_future = goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=5.0)
    finally:
        if node is not None:
            node.destroy_node()
        if not rclpy_was_running and rclpy.ok():
            rclpy.shutdown()

    elapsed = time.perf_counter() - t_start
    throughput_hz = (len(preds) / elapsed) if (preds and elapsed > 0) else 0.0
    return preds, throughput_hz
