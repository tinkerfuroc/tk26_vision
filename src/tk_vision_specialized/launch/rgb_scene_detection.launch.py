"""Launch the two on-demand RGB VLM nodes; cameras are started separately."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [DeclareLaunchArgument('enable_behaviour', default_value='true'),
            DeclareLaunchArgument('enable_litter', default_value='true'),
            DeclareLaunchArgument('rgb_topic', default_value='/camera/color/image_raw'),
            DeclareLaunchArgument('vision_log_folder', default_value='vision_log')]
    parameters = [{'rgb_topic': LaunchConfiguration('rgb_topic'),
                   'vision_log_folder': LaunchConfiguration('vision_log_folder')}]
    return LaunchDescription(args + [
        Node(package='tk_vision_specialized', executable='behaviour_detection',
             parameters=parameters, output='screen',
             condition=IfCondition(LaunchConfiguration('enable_behaviour'))),
        Node(package='tk_vision_specialized', executable='litter_detection',
             parameters=parameters, output='screen',
             condition=IfCondition(LaunchConfiguration('enable_litter'))),
    ])
