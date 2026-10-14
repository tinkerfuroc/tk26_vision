# Orbbec 场景检测验证页面

页面提供实时 RGB（最高 5 FPS）、人物行为/地面垃圾两个检测按钮，分别保留标注图、标签、描述、bbox、采集时间戳、日志路径、实际使用的模型及耗时。行为结果额外显示位置或定位失败原因。

## 构建与启动（ROS 主机）

```bash
cd /home/tinker/tk25_ws
./src/tk26_vision/scripts/build.sh --packages-select tinker_vision_msgs_26 tk_vision_specialized
source install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/src/tk26_vision/config/fastdds_shm.xml
```

相机未运行时，单独终端执行 `./src/tk26_vision/scripts/launch_orbbec_shm.sh`。

如果 bringup 已启动两个检测节点，只启动页面：

```bash
ros2 launch tk_vision_specialized scene_detection_web.launch.py
```

如果检测节点尚未启动，可一起启动（默认使用已配置的 GLM API，无 Qwen fallback）：

```bash
ros2 launch tk_vision_specialized scene_detection_web.launch.py start_detectors:=true
```

浏览器访问 `http://<服务器IP>:8088`。端口可用 `port:=8090` 改写；仅本机访问可设 `host:=127.0.0.1`。该工具用于可信局域网验证，不包含用户认证。页面不接触 API key。

参数还有 `rgb_topic`、`depth_topic`、`camera_info_topic`、`target_frame`（默认 map）、`vision_log_folder`。没有 map TF 时可以指定实际 RGB 光学坐标系验证相机坐标定位；缺少深度/TF 不影响 RGB 行为分类。页面不启动导航、不调用行为树。

勿重复启动同名检测服务。`start_detectors:=false` 时模型由已有检测节点决定，页面展示服务返回的实际模型。

## 耗时与错误

- `vlm_elapsed_s`：节点内 `infer_scene` 墙钟耗时，包含图像编码、网络、模型推理、重试/fallback 和解析；不是服务商提供的纯模型计算时间。未调用 VLM 时为 -1，页面显示“未调用”。同时写入 vision_log JSON 的 model 字段。
- `service_elapsed_s`：Web 后端等待 ROS 服务的总耗时，包括定位、VLM 和日志保存，不含浏览器传输图片时间。
- 预览超过 2 秒无新帧即停止显示旧图。检测依据服务接收到的最新新鲜图像，标注图与返回时间戳对应，不一定等于点击瞬间的预览帧。
- 两种检测可以独立运行；同一种检测期间禁止重复点击。HTTP 等待超时后保留服务占用状态，直到 ROS 请求真正完成。
- 非零服务 status 显示错误，不能当作“没有目标”。日志保存失败仍展示已得到的识别结果及错误。

本次扩展了 `DetectRgbScene` 响应字段，升级时须重新构建接口包及 specialized 包，并重启相关检测节点与页面。

## 验证

在 ROS 主机现有虚拟环境中执行 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q src/tk_vision_specialized/test/test_scene_web.py`（从 repo 根目录运行，先 source 工作空间）。覆盖 HTTP 页面、无图/旧图、跨源拒绝、错误结果与计时序列化、超时后防重复请求。

2026-10-12 在 tkmr 完成接口与节点构建，Web、scene VLM、logging、geometry 合计 57 项测试通过。通过 HTTP API 实测真实 Orbbec 图像 + GLM：行为检测返回 sitting_on_chair（VLM 18.62 s，服务 19.21 s）；垃圾检测返回疑似地面纸箱（VLM 21.91 s，服务 21.98 s）。两次均 status=0，JPEG 和 vision_log JSON 可读。这是链路验证，不是准确率评估。现场缺少 camera_color_optical_frame → map TF，人物定位结果为无效并附原因。内置浏览器拦截了服务器地址，未完成浏览器视觉验收。
