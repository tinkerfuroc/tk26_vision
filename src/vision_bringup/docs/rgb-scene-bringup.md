# RGB scene detection bringup

`vision_bringup.launch.py` starts `tk_vision_specialized/behaviour_detection`
and `tk_vision_specialized/litter_detection` by default, independently of task
flags. Existing generalist and door services remain in the default core.

| Launch argument | Default | Executable |
|---|---|---|
| `enable_behaviour_detection` | `true` | `behaviour_detection` |
| `enable_litter_detection` | `true` | `litter_detection` |

Both use `/camera/color/image_raw` and inherit the perception launch's DDS
environment. Start the Orbbec driver first. These services need RGB only;
they do not require depth or TF. Requests to `/behaviour_detection/detect`
and `/litter_detection/detect` trigger inference and vision-log writes.
Starting the nodes does not trigger VLM calls. Provider keys are loaded from
the workspace `.env`; missing keys are reported when a detection is requested.

To disable both:

```bash
ros2 launch vision_bringup vision_bringup.launch.py \
  enable_behaviour_detection:=false enable_litter_detection:=false
```

Do not run the specialized standalone launch or individual executables at the
same time as bringup unless the corresponding flags above are disabled.
The existing `tk_vision_specialized` dependency in `vision_bringup/package.xml`
already covers these executables. The new interface definitions and specialized
package must be built on the ROS host before running the updated bringup.

See [node documentation](../../tk_vision_specialized/RGB_SCENE_DETECTION.md)
for labels, response fields, model parameters and logging details.
