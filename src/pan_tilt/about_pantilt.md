# pan_tilt_base_v0.9

> Note
> This file is firmware and bench-side setup documentation only.
> Runtime ROS 2 integration now lives in [README.md](./README.md).
> The clean pan-tilt refactor removed the old `pan_tilt/ctrl` runtime,
> `/pan_tilt_ctrl*` topics, and `specs.json` as a runtime dependency.
> Also note that helper files referenced below such as
> `pan_tilt_base_v0.9.ino` and `pan_tilt_hf_test.py` are not checked into this
> repo; treat those references as historical/external tooling notes.

Firmware and test notes for the Waveshare 2-axis pan-tilt camera module using
[`pan_tilt_base_v0.9.ino`](./pan_tilt_base_v0.9.ino).

This README focuses on:

- how to upload and use the firmware from Arduino IDE
- how to send serial JSON commands to the pan-tilt
- how to initialize the two motor IDs
- how to set the zero-state degree for both axes

## Safety First

- Keep the pan-tilt clear of cables, hands, and hard stops during setup.
- Power off before unplugging or reconnecting either servo bus cable.
- During ID initialization, disconnect one motor before changing the other one.
  Both motors can ship with the same default ID, so changing IDs with both
  attached can affect the wrong motor.
- Use small motions first before trying high-frequency control.

## Hardware Assumptions

This firmware expects the gimbal motor IDs below:

- pan / X-axis motor: `2`
- tilt / Y-axis motor: `1`

These values are defined in `ugv_config.h` as:

- `GIMBAL_PAN_ID  2`
- `GIMBAL_TILT_ID 1`

The firmware also clamps gimbal commands to these angle ranges:

- `X`: `-180` to `180`
- `Y`: `-30` to `90`

## Assembly Reference

For the mechanical assembly and vendor-side configuration guide, see:

- https://www.waveshare.net/wiki/2-Axis_Pan-Tilt_Camera_Module_组装和配置教程

## Arduino IDE Setup

1. Open [`pan_tilt_base_v0.9.ino`](./pan_tilt_base_v0.9.ino) in Arduino IDE.
2. Select the correct ESP32 board and serial port.
3. Upload the firmware.
4. Open Serial Monitor.
5. Set Serial Monitor to:
   - baud rate: `115200`
   - line ending: `Newline`

Important:

- The firmware serial parser waits for `\n`, so `Newline` matters.
- Only one program can own the serial port at a time. Close Serial Monitor
  before using `pan_tilt_hf_test.py`, and close Python tools before returning to
  Serial Monitor.

## Basic Serial Usage

After boot, you can control the pan-tilt by sending JSON commands in Serial
Monitor.

## Automatic X/Y Feedback

The latest firmware boots in gimbal mode by default. After the pan-tilt is
plugged in, powered on, and finished booting, it automatically sends `T:1001`
feedback to the computer through USB serial.

The `T:1001` gimbal feedback no longer sends the module's own roll, pitch, yaw,
or voltage fields. It only sends the current pan/tilt values as `X` and `Y`:

```json
{"T":1001,"X":0,"Y":0}
```

Field meaning:

- `T`: feedback message type, always `1001`
- `X`: current pan / X-axis angle
- `Y`: current tilt / Y-axis angle

You should see this stream in Arduino IDE Serial Monitor after uploading
[`pan_tilt_base_v0.9.ino`](./pan_tilt_base_v0.9/pan_tilt_base_v0.9.ino) and
opening the correct serial port at `115200` baud.

Useful motion command:

```json
{"T":133,"X":0,"Y":0,"SPD":0,"ACC":0}
```

This sends a simple pan/tilt goal:

- `X`: pan angle target
- `Y`: tilt angle target
- `SPD`: speed parameter
- `ACC`: acceleration parameter

Example moves:

```json
{"T":133,"X":0,"Y":0,"SPD":0,"ACC":0}
{"T":133,"X":30,"Y":15,"SPD":120,"ACC":20}
{"T":133,"X":-45,"Y":10,"SPD":120,"ACC":20}
```

## Motor ID Initialization

Use this procedure when both motors still have the factory default ID and you
need to separate them into:

- pan / X-axis motor -> ID `2`
- tilt / Y-axis motor -> ID `1`

### Step 0: Upload firmware

Prepare Arduino IDE and upload [`pan_tilt_base_v0.9.ino`](./pan_tilt_base_v0.9.ino).
Use Arduino IDE Serial Monitor as the communication tool during initialization.

### Step 1: Isolate one motor

1. Remove external power.
2. Turn off the power button.
3. Keep USB-C connected to the pan-tilt controller.
4. Unplug the bus/transmission wire of the second motor so that only one motor
   remains on the servo bus.

This prevents both default-ID motors from responding to the same ID-change
command.

### Step 2: Power on

1. Plug in power.
2. Turn on the power button.
3. Wait for the controller to finish booting.

### Step 3: Set the X-axis motor ID to `2`

Send:

```json
{"T":501,"raw":1,"new":2}
```

Meaning:

- `raw`: current servo ID
- `new`: new servo ID to store

Because factory default ID is usually `1`, this changes the connected motor from
ID `1` to ID `2`.

Recommended assumption:

- Do this first for the pan / X-axis motor, since the firmware expects pan to be
  ID `2`.

### Step 4: Reconnect the second motor

1. Reconnect the other motor.

At this point the two motors should now be addressable separately as:

- pan / X-axis motor: `2`
- tilt / Y-axis motor: `1`

Optional torque-release command:

```json
{"T":210,"cmd":0}
```

Note:

- `T210` is torque control, not ID setup.
- `{"T":210,"cmd":0}` releases torque.
- `{"T":210,"cmd":1}` enables torque.

This can be useful if you need to manually align the mechanism before setting the
zero position.




In practice, this is how you define the physical pose that later corresponds to:

```json
{"T":133,"X":0,"Y":0,"SPD":0,"ACC":0}
```

### Step 5: Check the current default zero

After both motors are connected, test the current saved zero:

```json
{"T":133,"X":0,"Y":0,"SPD":0,"ACC":0}
```

Observe whether the current zero position matches the pose you want.

### Step 6: Move to the desired default pose

Send a target command that moves the pan-tilt to the physical pose you want to
be treated as zero:

```json
{"T":133,"X":??,"Y":??,"SPD":0,"ACC":0}
```

Replace `??` with the temporary angles that place the mechanism at your desired
home pose.

Example:

```json
{"T":133,"X":15,"Y":-5,"SPD":0,"ACC":0}
```

When the pan-tilt reaches the desired physical orientation, keep it there for the
next step.

### Step 7: Store the current position as zero for both motors

Send:

```json
{"T":502,"id":1} to set the tilt / Y-axis motor zero.
{"T":502,"id":2} to set the pan / X-axis motor zero.
```

This calibrates the current motor positions as the saved middle positions.

### Step 8: Verify the new zero

Send again:

```json
{"T":133,"X":0,"Y":0,"SPD":0,"ACC":0}
```

If the calibration succeeded, the pan-tilt should return to the physical pose
you wanted to define as zero.

## Quick Command Reference

```json
{"T":133,"X":0,"Y":0,"SPD":0,"ACC":0}
{"T":210,"cmd":0}
{"T":210,"cmd":1}
{"T":501,"raw":1,"new":2}
{"T":502,"id":1}
{"T":502,"id":2}
```

## Recommended Bring-Up Sequence

For normal usage after IDs and zero-state are already configured:

1. Power on the pan-tilt.
2. Open Serial Monitor at `115200` with `Newline`.
3. Confirm automatic `{"T":1001,"X":...,"Y":...}` feedback is streaming.
4. Send `{"T":133,"X":0,"Y":0,"SPD":0,"ACC":0}`.
5. Try small test moves before high-frequency control.

## Optional Python Test Script

Once Arduino IDE control is working, you can close Serial Monitor and use:

- [`pan_tilt_hf_test.py`](./pan_tilt_hf_test.py)

Example:

```bash
/opt/miniconda3/bin/python3 pan_tilt_hf_test.py --port /dev/cu.usbserial-3140 --vigorous --duration 10 --request-xy-hz 0 --startup-delay 4
```

## Troubleshooting

- No response to serial commands:
  - confirm baud rate is `115200`
  - confirm line ending is `Newline`
  - confirm only one serial client is connected
- Both motors move during ID setup:
  - both motors are probably still sharing the same default ID
  - power off and disconnect one motor before sending `T501`
- `X=0, Y=0` points to the wrong physical pose:
  - repeat the zero-state procedure with `T502`
- Feedback is missing:
  - confirm the latest firmware has been uploaded
  - confirm Serial Monitor baud rate is `115200`
  - confirm the pan-tilt has finished booting
  - optionally send `{"T":4,"cmd":2}` to force gimbal mode
  - optionally send `{"T":131,"cmd":1}` to re-enable feedback flow
