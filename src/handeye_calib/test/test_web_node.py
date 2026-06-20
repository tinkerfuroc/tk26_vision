import rclpy
from handeye_calib.handeye_web import HandeyeWebNode


def setup_module(_):
    rclpy.init()


def teardown_module(_):
    if rclpy.ok():
        rclpy.shutdown()


def test_node_constructs_and_safe_defaults():
    node = HandeyeWebNode()
    try:
        st = node.get_state_dict()
        assert st["camera_connected"] is False        # no camera in this env
        assert st["num_samples"] == 0
        jpg = node.latest_jpeg()
        assert isinstance(jpg, (bytes, bytearray)) and bytes(jpg[:2]) == b"\xff\xd8"
        cap = node.do_capture()
        assert cap["ok"] is False                      # nothing to capture
        sol = node.do_solve()
        assert sol["ok"] is False and "sample" in sol["reason"].lower()
    finally:
        node.destroy_node()


def test_do_move_validates_joint_count():
    node = HandeyeWebNode()
    try:
        bad = node.do_move([0.0, 0.0, 0.0])            # wrong arity
        assert bad["ok"] is False
    finally:
        node.destroy_node()
