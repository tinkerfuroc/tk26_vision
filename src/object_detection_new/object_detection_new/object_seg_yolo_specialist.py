"""Specialist-YOLO detection node.

Thin wrapper around `YOLOSegmentationNode` that advertises the specialist
`object_detection_yolo` service and filters out the 'person' class regardless
of what the (custom-trained) model emits. The competition model is expected
to be trained on arena items only, but this blacklist makes the guarantee
explicit at service boundary so downstream grasping logic never receives a
person detection.

Coexists with `yolo_seg_default_node` (pretrained YOLO, `/object_detection`,
no blacklist) and the generalist node (clean YOLO + VLM+SAM fallback).
"""

import rclpy
import cv2
from rclpy.parameter import Parameter

from .object_seg_yolo import YOLOSegmentationNode


def main(args=None):
    rclpy.init(args=args)
    node = YOLOSegmentationNode(
        node_name='yolo_segmentation_node',
        parameter_overrides=[
            Parameter('excluded_classes',
                      Parameter.Type.STRING_ARRAY, ['person']),
        ],
    )

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.visualization:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
