"""Capture ChArUco frames for RGB intrinsic calibration.

There is no in-repo frame dumper, and intrinsic accuracy is dominated by how
well the captured frames cover the IMAGE PERIPHERY (the corners), which is
exactly where this rig's hand-eye calibration degrades. This node subscribes to
the live color stream, detects the ChArUco board each frame, and auto-saves a
frame only when it adds pose/coverage diversity — tracking a 3x3 image-region
grid so the operator can see which corners still need the board.

Usage (in .venv-calib, after sourcing ROS + install/setup):

    python -m pan_tilt.calibration.capture_intrinsic \
        --out /home/tinker/tk25_ws/calibration_data/intr_0625 \
        --n 40

Then fit:

    python -m pan_tilt.calibration.run_calibration intrinsic \
        /home/tinker/tk25_ws/calibration_data/intr_0625 \
        --out /home/tinker/tk25_ws/calibration_data/intr_0625

Hold the board by hand and sweep it slowly across the whole frame — push it into
all four CORNERS, at 0.4-1.0 m, and tilt it +/-30 deg in roll/pitch/yaw. The
node prints a coverage grid; aim for >=3 captures in every cell, including the
four corner cells, before it reaches --n.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from pan_tilt.calibration.aruco_detect import BoardSpec, build_board


def _build_detector(board):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), params)


class IntrinsicCapture(Node):
    def __init__(self, args):
        super().__init__("intrinsic_capture")
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        # Default BoardSpec already matches the physical board (5x5, 40mm, DICT_5X5_100).
        self.board = build_board(BoardSpec())
        self.detector = _build_detector(self.board)
        self.saved = 0
        self.last_center = None          # normalized (u, v) of last saved board centroid
        self.grid = np.zeros((3, 3), dtype=int)   # 3x3 coverage histogram
        self.show = args.show and self._display_ok()
        self.sub = self.create_subscription(
            Image, args.topic, self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Capturing to {self.out} from {args.topic}; target {args.n} frames, "
            f">={args.min_corners} corners, move-thresh {args.move_thresh:.2f}. "
            f"{'Preview ON' if self.show else 'Headless'}."
        )

    @staticmethod
    def _display_ok() -> bool:
        import os
        return bool(os.environ.get("DISPLAY"))

    def _on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cv_bridge failed: {exc}")
            return
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = self.detector.detectBoard(gray)
        n = 0 if ch_ids is None else len(ch_ids)

        center = None
        if n >= self.args.min_corners:
            center = ch_corners.reshape(-1, 2).mean(axis=0)
            center_n = (float(center[0] / w), float(center[1] / h))
            if self._should_save(center_n):
                self._save(bgr, center_n)

        if self.show:
            self._preview(bgr, ch_corners, ch_ids, n)

        if self.saved >= self.args.n:
            self.get_logger().info("Reached target frame count.")
            self._summary()
            rclpy.shutdown()

    def _should_save(self, center_n) -> bool:
        gx = min(2, int(center_n[0] * 3))
        gy = min(2, int(center_n[1] * 3))
        cell_starved = self.grid[gy, gx] < self.args.per_cell
        moved = (
            self.last_center is None
            or np.hypot(center_n[0] - self.last_center[0],
                        center_n[1] - self.last_center[1]) >= self.args.move_thresh
        )
        # Save if the pose moved enough, OR if this grid cell is still under-covered
        # (so corners get filled even if the operator lingers there).
        return moved or cell_starved

    def _save(self, bgr, center_n):
        path = self.out / f"img{self.saved:04d}.png"
        cv2.imwrite(str(path), bgr)
        self.saved += 1
        self.last_center = center_n
        gx = min(2, int(center_n[0] * 3))
        gy = min(2, int(center_n[1] * 3))
        self.grid[gy, gx] += 1
        self.get_logger().info(f"[{self.saved}/{self.args.n}] saved {path.name}")
        self._print_grid()

    def _print_grid(self):
        rows = []
        for gy in range(3):
            rows.append(" ".join(f"{self.grid[gy, gx]:2d}" for gx in range(3)))
        self.get_logger().info("coverage 3x3 (corners matter most):\n  " + "\n  ".join(rows))

    def _summary(self):
        self._print_grid()
        weak = [(gy, gx) for gy in range(3) for gx in range(3)
                if self.grid[gy, gx] < self.args.per_cell]
        if weak:
            self.get_logger().warn(
                f"{len(weak)} grid cell(s) under {self.args.per_cell} captures "
                f"(esp. corners) — fit may still be weak at the periphery: {weak}"
            )
        self.get_logger().info(
            f"Done: {self.saved} frames in {self.out}. Now run:\n"
            f"  python -m pan_tilt.calibration.run_calibration intrinsic {self.out} --out {self.out}"
        )

    def _preview(self, bgr, ch_corners, ch_ids, n):
        vis = bgr.copy()
        if ch_ids is not None and n > 0:
            cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
        cv2.putText(vis, f"saved {self.saved}/{self.args.n}  corners {n}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("intrinsic_capture", cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2)))
        cv2.waitKey(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output dir for img####.png")
    parser.add_argument("--topic", default="/camera/color/image_raw")
    parser.add_argument("--n", type=int, default=40, help="target frame count")
    parser.add_argument("--min-corners", type=int, default=12,
                        help="min ChArUco corners to accept a frame (board has 16)")
    parser.add_argument("--move-thresh", type=float, default=0.06,
                        help="normalized board-center move to count as a new pose")
    parser.add_argument("--per-cell", type=int, default=4,
                        help="target captures per 3x3 grid cell")
    parser.add_argument("--show", action="store_true", help="live preview (needs DISPLAY)")
    # Tolerate ROS-injected args when launched via ros2 run.
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    rclpy.init()
    node = IntrinsicCapture(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._summary()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
