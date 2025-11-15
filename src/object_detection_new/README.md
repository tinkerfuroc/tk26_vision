Service name offered is `object_detection_yolo`.

Servive request `prompt` should match EXACTLY with trained yolo class.

Put model in `models/` directory (under `object_detection`, siblings with `config` folder)

Start node using 
```
ros2 run object_detection_new yolo_seg_node --ros-args -p model_path:="<pt_name>" -p visualization:=true -p sort_mode:="highest"
```

After starting, wait for `YOLO Segmentation Node initialized successfully` before starting task