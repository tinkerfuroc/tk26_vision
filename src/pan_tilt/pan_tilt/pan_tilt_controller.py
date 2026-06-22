"""Serial pan-tilt controller node."""

import json
import math
import threading
import time
from typing import Optional

import rclpy
import serial
from rclpy.node import Node
from std_srvs.srv import Trigger
from tinker_vision_msgs_26.msg import PanTiltCommand, PanTiltState
from tinker_vision_msgs_26.srv import SetTorque, SetZero


class PanTiltControllerNode(Node):
    def __init__(self):
        super().__init__('pan_tilt_controller')

        self.declare_parameter('device', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('startup_delay_sec', 4.0)
        self.declare_parameter('feedback_startup_timeout_sec', 2.0)
        self.declare_parameter('feedback_stale_timeout_sec', 0.5)
        self.declare_parameter('state_publish_rate_hz', 20.0)
        self.declare_parameter('pan_min_deg', -180.0)
        self.declare_parameter('pan_max_deg', 180.0)
        self.declare_parameter('tilt_min_deg', -30.0)
        self.declare_parameter('tilt_max_deg', 90.0)
        self.declare_parameter('default_speed_raw', 120)
        self.declare_parameter('default_accel_raw', 20)
        self.declare_parameter('send_force_gimbal_on_startup', True)
        self.declare_parameter('send_enable_feedback_on_startup', True)
        self.declare_parameter('invert_pan', False)
        self.declare_parameter('invert_tilt', False)
        self.declare_parameter('pan_trim_deg', 0.0)
        self.declare_parameter('tilt_trim_deg', 0.0)
        self.declare_parameter('initial_pan_deg', 0.0)
        self.declare_parameter('initial_tilt_deg', 0.0)

        self._device = self.get_parameter('device').value
        self._baudrate = int(self.get_parameter('baudrate').value)
        self._startup_delay_sec = float(self.get_parameter('startup_delay_sec').value)
        self._feedback_startup_timeout_sec = float(
            self.get_parameter('feedback_startup_timeout_sec').value,
        )
        self._feedback_stale_timeout_sec = float(
            self.get_parameter('feedback_stale_timeout_sec').value,
        )
        self._state_publish_rate_hz = float(
            self.get_parameter('state_publish_rate_hz').value,
        )
        self._pan_min_deg = float(self.get_parameter('pan_min_deg').value)
        self._pan_max_deg = float(self.get_parameter('pan_max_deg').value)
        self._tilt_min_deg = float(self.get_parameter('tilt_min_deg').value)
        self._tilt_max_deg = float(self.get_parameter('tilt_max_deg').value)
        self._default_speed_raw = int(self.get_parameter('default_speed_raw').value)
        self._default_accel_raw = int(self.get_parameter('default_accel_raw').value)
        self._send_force_gimbal_on_startup = bool(
            self.get_parameter('send_force_gimbal_on_startup').value,
        )
        self._send_enable_feedback_on_startup = bool(
            self.get_parameter('send_enable_feedback_on_startup').value,
        )
        self._invert_pan = bool(self.get_parameter('invert_pan').value)
        self._invert_tilt = bool(self.get_parameter('invert_tilt').value)
        self._pan_trim_deg = float(self.get_parameter('pan_trim_deg').value)
        self._tilt_trim_deg = float(self.get_parameter('tilt_trim_deg').value)
        initial_pan_deg = float(self.get_parameter('initial_pan_deg').value)
        initial_tilt_deg = float(self.get_parameter('initial_tilt_deg').value)

        self._serial_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_feedback = threading.Event()
        self._serial_connected = False
        self._last_feedback_monotonic: Optional[float] = None
        self._feedback_raw_pan_deg = self._desired_to_raw_pan(initial_pan_deg)
        self._feedback_raw_tilt_deg = self._desired_to_raw_tilt(initial_tilt_deg)
        self._command_raw_pan_deg = self._feedback_raw_pan_deg
        self._command_raw_tilt_deg = self._feedback_raw_tilt_deg

        self._state_pub = self.create_publisher(PanTiltState, '~/state', 10)
        self.create_subscription(PanTiltCommand, '~/cmd', self._handle_command, 10)
        self.create_service(SetTorque, '~/set_torque', self._handle_set_torque)
        self.create_service(SetZero, '~/set_zero', self._handle_set_zero)
        self.create_service(
            Trigger, '~/remap_servo_ids', self._handle_remap_servo_ids,
        )
        self.create_timer(
            1.0 / max(self._state_publish_rate_hz, 1.0),
            self._publish_state,
        )

        self._serial = serial.Serial(
            self._device,
            self._baudrate,
            timeout=0.1,
            write_timeout=0.5,
        )
        self._serial_connected = self._serial.is_open
        self.get_logger().info(
            f'Opened serial device {self._device} @ {self._baudrate}',
        )

        time.sleep(self._startup_delay_sec)

        self._reader_thread = threading.Thread(
            target=self._read_serial_loop,
            daemon=True,
        )
        self._reader_thread.start()

        if self._send_force_gimbal_on_startup:
            self._send_payload({'T': 4, 'cmd': 2})
        if self._send_enable_feedback_on_startup:
            self._send_payload({'T': 131, 'cmd': 1})

        if not self._first_feedback.wait(timeout=self._feedback_startup_timeout_sec):
            self.get_logger().warn(
                'No T:1001 feedback received during startup timeout; '
                'state will remain stale until the firmware responds.',
            )

    def _handle_command(self, msg: PanTiltCommand):
        with self._state_lock:
            if self._last_feedback_monotonic is None:
                base_raw_pan_deg = self._command_raw_pan_deg
                base_raw_tilt_deg = self._command_raw_tilt_deg
            else:
                base_raw_pan_deg = self._feedback_raw_pan_deg
                base_raw_tilt_deg = self._feedback_raw_tilt_deg

        base_pan_deg = self._raw_to_desired_pan(base_raw_pan_deg)
        base_tilt_deg = self._raw_to_desired_tilt(base_raw_tilt_deg)
        pan_delta_deg = math.degrees(msg.pan_rad)
        tilt_delta_deg = math.degrees(msg.tilt_rad)

        if msg.mode == PanTiltCommand.RELATIVE:
            target_pan_deg = base_pan_deg + pan_delta_deg
            target_tilt_deg = base_tilt_deg + tilt_delta_deg
        else:
            target_pan_deg = pan_delta_deg
            target_tilt_deg = tilt_delta_deg

        target_pan_deg = min(max(target_pan_deg, self._pan_min_deg), self._pan_max_deg)
        target_tilt_deg = min(
            max(target_tilt_deg, self._tilt_min_deg),
            self._tilt_max_deg,
        )
        speed_raw = (
            self._default_speed_raw if msg.speed_raw <= 0 else int(msg.speed_raw)
        )
        accel_raw = (
            self._default_accel_raw if msg.accel_raw <= 0 else int(msg.accel_raw)
        )

        target_raw_pan_deg = self._desired_to_raw_pan(target_pan_deg)
        target_raw_tilt_deg = self._desired_to_raw_tilt(target_tilt_deg)
        self._send_motion_command(
            target_raw_pan_deg,
            target_raw_tilt_deg,
            speed_raw,
            accel_raw,
        )

        with self._state_lock:
            self._command_raw_pan_deg = target_raw_pan_deg
            self._command_raw_tilt_deg = target_raw_tilt_deg

    def _handle_set_torque(
        self,
        request: SetTorque.Request,
        response: SetTorque.Response,
    ):
        try:
            self._send_payload({'T': 210, 'cmd': 1 if request.enable else 0})
        except serial.SerialException as exc:
            response.success = False
            response.message = str(exc)
            return response

        response.success = True
        response.message = 'Torque command sent.'
        return response

    def _handle_set_zero(
        self,
        request: SetZero.Request,
        response: SetZero.Response,
    ):
        logger = self.get_logger()
        logger.info(f'[set_zero] service called with axis={request.axis}')
        axis_ids = []
        if request.axis == SetZero.Request.BOTH:
            axis_ids = [1, 2]
        elif request.axis == SetZero.Request.TILT:
            axis_ids = [1]
        elif request.axis == SetZero.Request.PAN:
            axis_ids = [2]
        else:
            logger.error(f'[set_zero] unsupported axis value: {request.axis}')
            response.success = False
            response.message = f'Unsupported axis value: {request.axis}'
            return response

        if not self._serial_connected:
            logger.error('[set_zero] serial device not connected; cannot send T:502')
            response.success = False
            response.message = 'Serial device not connected.'
            return response

        try:
            for axis_id in axis_ids:
                payload = {'T': 502, 'id': axis_id}
                logger.info(f'[set_zero] writing serial: {payload}')
                self._send_payload(payload)
        except serial.SerialException as exc:
            logger.error(f'[set_zero] serial write failed: {exc}')
            response.success = False
            response.message = str(exc)
            return response

        logger.info(f'[set_zero] T:502 sent for axis_ids={axis_ids}')
        response.success = True
        response.message = f'Zero command sent for axis_ids={axis_ids}.'
        return response

    def _handle_remap_servo_ids(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ):
        """Send `{'T':501,'raw':1,'new':2}` — middle step of the zero-state
        wizard. Operator must physically disconnect the second motor before
        triggering this; the command renumbers the remaining (still-attached)
        servo from raw_id=1 to new_id=2 so the subsequent T:502 pass can
        address each motor individually.
        """
        logger = self.get_logger()
        logger.info('[remap_servo_ids] Trigger service called')
        if not self._serial_connected:
            logger.error('[remap_servo_ids] serial device not connected; cannot send T:501')
            response.success = False
            response.message = 'Serial device not connected.'
            return response
        payload = {'T': 501, 'raw': 1, 'new': 2}
        try:
            logger.info(f'[remap_servo_ids] writing serial: {payload}')
            self._send_payload(payload)
        except serial.SerialException as exc:
            logger.error(f'[remap_servo_ids] serial write failed: {exc}')
            response.success = False
            response.message = str(exc)
            return response
        logger.info('[remap_servo_ids] T:501 sent')
        response.success = True
        response.message = 'T:501 raw=1 new=2 sent.'
        return response

    def _send_motion_command(
        self,
        pan_deg: float,
        tilt_deg: float,
        speed_raw: int,
        accel_raw: int,
    ):
        self._send_payload(
            {
                'T': 133,
                'X': round(pan_deg, 4),
                'Y': round(tilt_deg, 4),
                'SPD': speed_raw,
                'ACC': accel_raw,
            },
        )

    def _send_payload(self, payload):
        line = json.dumps(payload, separators=(',', ':')) + '\n'
        with self._serial_lock:
            self._serial.write(line.encode('utf-8'))

    def _read_serial_loop(self):
        while not self._stop_event.is_set():
            try:
                raw_line = self._serial.readline()
            except serial.SerialException as exc:
                self._serial_connected = False
                if not self._stop_event.is_set():
                    self.get_logger().error(f'Serial read failed: {exc}')
                return

            if not raw_line:
                continue

            try:
                line = raw_line.decode('utf-8').strip()
                payload = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            if payload.get('T') != 1001:
                continue

            try:
                raw_pan_deg = float(payload['X'])
                raw_tilt_deg = float(payload['Y'])
            except (KeyError, TypeError, ValueError):
                continue

            with self._state_lock:
                self._feedback_raw_pan_deg = raw_pan_deg
                self._feedback_raw_tilt_deg = raw_tilt_deg
                self._command_raw_pan_deg = raw_pan_deg
                self._command_raw_tilt_deg = raw_tilt_deg
                self._last_feedback_monotonic = time.monotonic()
            self._first_feedback.set()

    def _publish_state(self):
        msg = PanTiltState()
        msg.header.stamp = self.get_clock().now().to_msg()

        with self._state_lock:
            msg.pan_rad = math.radians(
                self._raw_to_desired_pan(self._feedback_raw_pan_deg),
            )
            msg.tilt_rad = math.radians(
                self._raw_to_desired_tilt(self._feedback_raw_tilt_deg),
            )
            last_feedback = self._last_feedback_monotonic

        msg.connected = bool(self._serial_connected and self._serial.is_open)
        msg.feedback_ok = (
            last_feedback is not None
            and (time.monotonic() - last_feedback) <= self._feedback_stale_timeout_sec
        )
        self._state_pub.publish(msg)

    def _raw_to_desired_pan(self, raw_deg: float) -> float:
        sign = -1.0 if self._invert_pan else 1.0
        return sign * raw_deg + self._pan_trim_deg

    def _raw_to_desired_tilt(self, raw_deg: float) -> float:
        sign = -1.0 if self._invert_tilt else 1.0
        return sign * raw_deg + self._tilt_trim_deg

    def _desired_to_raw_pan(self, desired_deg: float) -> float:
        sign = -1.0 if self._invert_pan else 1.0
        return sign * (desired_deg - self._pan_trim_deg)

    def _desired_to_raw_tilt(self, desired_deg: float) -> float:
        sign = -1.0 if self._invert_tilt else 1.0
        return sign * (desired_deg - self._tilt_trim_deg)

    def destroy_node(self):
        self._stop_event.set()
        if hasattr(self, '_reader_thread'):
            self._reader_thread.join(timeout=1.0)
        if hasattr(self, '_serial') and self._serial.is_open:
            self._serial.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = PanTiltControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
