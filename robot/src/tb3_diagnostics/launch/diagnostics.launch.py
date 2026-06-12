"""Launch the per-host diagnostics node.

Same node runs on every ROS host; it auto-detects available probes
(chrony / WiFi / RAPL). Override `wifi_interface` or `research_project_path`
via launch args if a host differs from the defaults.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    wifi_interface = LaunchConfiguration('wifi_interface')

    return LaunchDescription([
        DeclareLaunchArgument('wifi_interface', default_value='wlan0'),
        Node(
            package='tb3_diagnostics',
            executable='diagnostics_node',
            name='diagnostics_node',
            parameters=[{'wifi_interface': wifi_interface}],
            output='screen',
        ),
    ])
