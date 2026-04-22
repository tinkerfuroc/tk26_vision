"""Default-YOLO detection node.

Thin wrapper around `YOLOSegmentationNode` that advertises the generalist
`object_detection` service with a stock pretrained YOLO model. Coexists with
`yolo_seg_node` (which advertises `object_detection_yolo` with a custom-trained
model). Used by downstream LLM nodes (kimi_api) that expect the legacy
`object_detection` service name.
"""

import rclpy
import cv2
from rclpy.parameter import Parameter

from .object_seg_yolo import YOLOSegmentationNode


def main(args=None):
    rclpy.init(args=args)
    node = YOLOSegmentationNode(
        node_name='yolo_segmentation_default_node',
        parameter_overrides=[
            Parameter('service_name', Parameter.Type.STRING, 'object_detection'),
            Parameter('model_path', Parameter.Type.STRING, 'yolo11n-seg.pt'),
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
