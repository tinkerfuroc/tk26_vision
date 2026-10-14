# RGB 行为与地面垃圾检测

两个独立节点均位于 `tk_vision_specialized`，使用 Orbbec RGB + VLM，不加载 YOLO/SAM。
`behaviour_detection` 另外订阅已配准深度、RGB CameraInfo 和 TF，估计人物三维位置；
`litter_detection` 仍只订阅 RGB。节点常驻，但只在收到服务请求时推理，
不会后台自动轮询或持续产生 API 费用。

## 节点和接口

| 可执行程序及节点名 | 服务 | 功能 |
|---|---|---|
| `behaviour_detection` | `/behaviour_detection/detect` | 观察人物姿态及举手示意 |
| `litter_detection` | `/litter_detection/detect` | 判断地板/地面是否存在疑似废弃物 |

两个服务均使用 `tinker_vision_msgs_26/srv/DetectRgbScene`，请求有可选的
`target_frame` 字符串，空字符串使用节点默认值 `map`；`{}` 仍可调用。
垃圾检测忽略该字段。接口定义已扩展，ROS 主机需要重新构建接口包及所有调用者。
每次调用固定一张图进行推理；推理过程中订阅回调仍能更新下一次调用的图像。
同一节点的检测请求串行执行，两个节点可独立执行。

响应字段：

- `header`：输入图像的采集时间戳及 frame_id，不是 VLM 完成时刻。
- `status`：0 成功（包含正常空结果）、1 无新鲜图像/解码失败、2 VLM 失败、3 日志失败。
- `error_msg`：失败原因。仅 `status == 0` 时可把 `detected == false` 当作正常未检出。
- `detected`：是否至少发现一个目标行为/疑似垃圾。
- `observations[]`：`SceneObservation`，包括 `labels[]`、可见证据/不确定性说明
  `description`、像素 `RegionOfInterest bbox`，以及 `position_valid`、
  `geometry_msgs/PointStamped position`、`position_error`。
- `rgb_image`、`annotated_image`：本次输入图及标框图，均保留采集 header。
- `log_json`：本次日志 JSON 的绝对路径。

ROI 使用左上角偏移和 width/height，右/下边界不包含在框内；日志 bbox
使用 `[x1,y1,x2,y2]`，同样为右/下边界 exclusive。框从 VLM 的 0..1000
归一化 xyxy 转换，格式错误、逆序框和越界框被视为解析失败并重试，
不会悄悄当作“没有目标”。不输出身份 ID 或未经校准的置信度。

## 人物定位

所有行为人物（不仅挥手者）都会尝试定位，每一行的坐标与该行 bbox/labels 对应。
调用者必须检查 `status == 0` 且该行 `position_valid == true` 后才能使用位置；
定位失败不改变行为检测的成功状态，仍保留 bbox/标签并填写 `position_error`。
无效位置的默认零值不是人物坐标。垃圾检测始终 `position_valid=false`。

在 VLM 调用前固定 RGB、与 RGB 时间戳最近且差值不超过 `sync_slop_s` 的深度、
RGB 内参，以及采集时刻到目标坐标系的 TF。推理期间的新帧不会替换这些输入。
深度缓冲最多 20 帧；接收超时、时间戳为零、不匹配的帧不能用于定位。
CameraInfo 作为静态标定缓存，其分辨率和 optical frame 必须与 RGB 一致。

**驱动必须启用 RGB 深度配准**（Orbbec `depth_registration:=true`）。深度应为
RGB 像素坐标对应的光轴距离，与 RGB 分辨率相同；仅尺寸相同不能证明已经配准，
节点不能从消息元数据自动证明该条件。兼容 `16UC1`/`mono16` 毫米和 `32FC1` 米。
算法取人物框中央宽高各 50% 的区域，过滤非有限值及范围外深度，再用深度中位数
附近 `max(0.15 m, 10%深度)` 的点反投影，输出 XYZ 中位数。有效点不足则失败。
没有 mask，背景占主导或遮挡时仍可能定位到背景；结果为近似位置，需实机评测。

`position.header.frame_id` 是请求目标坐标系，`position.header.stamp` 始终为 RGB
采集时间。默认 `map`；可请求 `base_link` 或 RGB optical frame（后者不需要 TF）。
TF 缺失时不会把相机坐标冒充 map 坐标。位置是人物三维估计点，不是地面停车位，
不包含人物朝向；本次没有修改行为树，也没有发送导航命令。

行为节点独有参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `depth_topic` | `/camera/depth/image_raw` | 必须已配准至 RGB |
| `camera_info_topic` | `/camera/color/camera_info` | RGB 内参 |
| `target_frame` | `map` | 请求为空时的输出坐标系 |
| `sync_slop_s` | 0.1 | RGB 与深度最大时间差（秒） |
| `min_depth_m` / `max_depth_m` | 0.2 / 8.0 | 有效深度范围（米） |
| `min_depth_pixels` | 25 | 中央区域及深度过滤后最少点数 |
| `tf_timeout_s` | 0.5 | 采集时刻 TF 查询等待上限 |

## 行为标签

| 标签 | 解释 |
|---|---|
| `lying_resting_or_sleeping` | 躺在床/沙发等高于地面的休息面上，疑似休息或睡觉 |
| `sitting_on_bed` | 坐在床上 |
| `sitting_on_sofa` | 坐在沙发上 |
| `sitting_on_chair` | 坐在椅子上 |
| `lying_on_floor_possible_fall` | 已躺倒/倒伏在地面，可能摔倒 |
| `standing_waving` | 站立举手/挥手示意 |
| `sitting_waving` | 坐姿举手/挥手示意 |

同一个人物只使用一个人物框，可以带多个标签，例如 `sitting_on_chair` 与
`sitting_waving`。普通站立且未举手的人不属于此次检测目标。
单图无法确认睡眠、摔倒发生过程或实际摆手运动，提示词明确约束这些结论：
报告可见姿态及疑似状态，静态抬手也计入示意。描述字段用于解释可见证据。

垃圾标签只有 `floor_litter`，描述中说明纸团、包装、瓶罐、食物残渣等类型及
地面关系。提示词排除桌面/架子上、手持、垃圾桶内的物品以及正常家具和用品。
“废弃”和“在地面上”均为 RGB 场景判断，没有几何深度验证，准确率尚待实际场景评测。

## 模型配置

启动前从当前工作目录向上加载 `.env`，沿用现有环境变量：

- Qwen：`DASHSCOPE_API_KEY`，兼容历史拼写 `DASHCOPE_API_KEY`。
- Gemini/OpenRouter：`OPENROUTER_API_KEY`，可选 `OPENROUTER_BASE_URL`。
- 模型默认值来自 `vision_util.vlm_models`，支持 `VISION_QWEN_MODEL`、
  `VISION_VLM_FLASH_MODEL`/`FLASH_MODEL`。

| ROS 参数 | 默认值 | 含义 |
|---|---|---|
| `rgb_topic` | `/camera/color/image_raw` | Orbbec 彩色 topic，可适配驱动 namespace |
| `max_image_age_s` | 2.0 | 调用开始时检查采集时间和本地接收时间，拒绝陈旧帧 |
| `vlm_provider` | `qwen` | 首选 provider |
| `vlm_fallback_provider` | `gemini` | 错误回退，可设空字符串禁用 |
| `vlm_model_qwen` | 环境变量或 `qwen3-vl-plus` | Qwen 模型 ID |
| `vlm_model_gemini` | 环境变量或 `google/gemini-2.5-flash` | OpenRouter 模型 ID |
| `vlm_timeout_s` | 30.0 | 重试和供应商回退共享的时间预算 |
| `vlm_attempts` | 2 | 每个有密钥的 provider 最多尝试次数，1..5 |
| `vision_log_folder` | `vision_log` | 日志根目录，相对启动工作目录 |

参数在启动时读取，不支持运行中热更新。无密钥的 provider 会跳过；全部不可用时
返回 VLM 错误。正常空数组是最终答案，不因此继续调用其他模型。
HTTP timeout 使用剩余预算且关闭 SDK 内置重试，但网络库的超时不等同于严格的
进程中断期限。服务没有 Action 取消协议，适合按需调用；请求方应预留网络和写盘时间。

必须使用与图像 header 一致的 ROS 时钟；仿真环境需要相应设置 `use_sim_time`。
缺失/零时间戳、超过年龄限制或明显在未来的图像会被拒绝。

## 日志内容

沿用 `VisionLogger` 的 `TINKER_VISION_SESSION_TS → 最近会话 → 新会话` 目录规则：

```text
vision_log/<session>/
  behaviour_detection_behaviour_<request-uuid>_orig_<time>.jpg
  behaviour_detection_behaviour_<request-uuid>_overlay_<time>.jpg
  behaviour_detection_behaviour_<request-uuid>_req_<time>.json
  litter_detection_litter_<request-uuid>_orig_<time>.jpg
  litter_detection_litter_<request-uuid>_overlay_<time>.jpg
  litter_detection_litter_<request-uuid>_req_<time>.json
```

JSON 包含 `capture_header.stamp.sec/nanosec`、`completed_at_utc`、任务类型、状态、
错误、模型/provider、`detected`、标签、description 和 bbox。
`localization` 保存方法、目标 frame、深度 header 和与 observations 同序的
`positions[]`（valid、error、frame_id、stamp、xyz）；定位无效时 xyz 为 null。
每次请求使用 UUID
避免同毫秒重名。日志中的 overlay 使用响应同一份标框图（磁盘为 JPEG 有损编码）。
正常未检出和 VLM 失败也保存图片和 JSON；无可用图时只能记录 JSON，
`capture_header=null`，不伪造图片或采集时间。写盘失败单独返回状态 3。
这些节点按要求始终记录日志，没有关闭日志的参数。

## 将来在 ROS 主机上的使用方式

以下为使用说明；远端隔离验证结果见文末。正式使用需构建接口包及节点包，
并加载正确 workspace；Python 使用已有 `tk_vision_specialized/requirements.txt`
中的 openai、python-dotenv、OpenCV 和 NumPy 依赖。垃圾检测只要求彩色流；
人物定位还要求上述深度、内参和对应 TF。没有这些输入时行为识别仍可运行。

```bash
ros2 run tk_vision_specialized behaviour_detection
ros2 run tk_vision_specialized litter_detection

# 或一起启动；不自动启动相机
ros2 launch tk_vision_specialized rgb_scene_detection.launch.py

ros2 service call /behaviour_detection/detect tinker_vision_msgs_26/srv/DetectRgbScene '{}'
ros2 service call /behaviour_detection/detect tinker_vision_msgs_26/srv/DetectRgbScene "{target_frame: base_link}"
ros2 service call /litter_detection/detect tinker_vision_msgs_26/srv/DetectRgbScene '{}'
```

`vision_bringup/vision_bringup.launch.py` 已默认启动这两个节点，独立于 HRI/GPSR 等任务开关。
可以分别设置 `enable_behaviour_detection:=false`、`enable_litter_detection:=false` 关闭。
使用主 bringup 时，不要再同时运行上述独立节点或 `rgb_scene_detection.launch.py`，
以免重复启动相同服务。默认启动仅使服务可用，仍需客户端请求才执行推理。

## 本地验证范围

纯 Python 测试在 `test_scene_vlm.py`、`test_scene_logging.py` 和
`test_scene_geometry.py`，54 项通过，覆盖标签、框坐标、深度反投影、
深度异常/内参异常、按采集时间匹配深度、陈旧帧拒绝、正常空结果、
供应商回退、时间预算和定位日志。本地不执行 ROS；远端 ROS 验证见下一节。
网络请求、JPEG 编码及 OpenCV 绘制用替身，不请求外部 API；日志测试调用真实
`VisionLogger`，核对文件及 JSON 契约，不声称验证了图像编码质量。
本机无 ROS；未运行 ROS、colcon、相机或实机模型效果验证。

## tkmr ROS 验证（2026-10-11）

使用主机已有 ROS Humble 和视觉虚拟环境，在独立构建目录验证，未安装依赖或
修改系统环境、原工作区 install。4 个相关包构建成功，54 项单元测试通过。
使用合成 RGB/深度/CameraInfo、真实 ROS 服务通信和 TF、模拟 VLM 返回，验证：

- uint16 毫米和 float32 米深度、含旋转和平移的 map 坐标变换。
- 请求覆盖目标坐标系、返回采集时间戳、真实 JPEG 和定位 JSON 落盘。
- 深度缺失/不同步、内参缺失/不匹配、无效深度及 TF 缺失时保留行为结果、位置无效。
- VLM 等待期间发布新深度和新 TF，仍使用推理前固定的深度及变换。
- 垃圾节点没有深度/TF 订阅；bringup 默认启动两节点并传递定位参数，正常退出。

未验证真实 Orbbec 场景或真实 VLM 识别/定位精度；未发送导航目标。
