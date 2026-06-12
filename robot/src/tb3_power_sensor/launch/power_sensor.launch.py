"""Launch the INA219 power-sensor node (robot SBC only)."""

import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('tb3_power_sensor'), 'config', 'sensor_map.yaml')

    return LaunchDescription([
        Node(
            package='tb3_power_sensor',
            executable='power_sensor_node',
            name='power_sensor_node',
            parameters=[config],
            output='screen',
        ),
    ])
