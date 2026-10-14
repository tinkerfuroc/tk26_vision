"""Opt-in validation UI and optional GLM-backed detection services; no camera startup."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as C
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {'host': '0.0.0.0', 'port': '8088', 'start_detectors': 'false',
                'rgb_topic': '/camera/color/image_raw', 'target_frame': 'map',
                'depth_topic': '/camera/depth/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'vlm_provider': 'gemini', 'vlm_model_gemini': 'glm-5.3-flash',
                'vision_log_folder': 'vision_log'}
    actions = [DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()]
    actions.append(Node(package='tk_vision_specialized', executable='scene_detection_web',
                        output='screen', parameters=[{'host': C('host'),
                            'port': ParameterValue(C('port'), value_type=int),
                            'rgb_topic': C('rgb_topic'), 'target_frame': C('target_frame')}]))
    for task in ('behaviour', 'litter'):
        params = {k: C(k) for k in ('rgb_topic', 'vision_log_folder',
                                    'vlm_provider', 'vlm_model_gemini')}
        params['vlm_fallback_provider'] = ''
        if task == 'behaviour':
            params.update({k: C(k) for k in ('depth_topic', 'camera_info_topic', 'target_frame')})
        actions.append(Node(package='tk_vision_specialized', executable=task + '_detection',
                            output='screen', parameters=[params],
                            condition=IfCondition(C('start_detectors'))))
    return LaunchDescription(actions)
