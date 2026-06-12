"""Launch file for the grid navigation node."""

import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('tb3_nav'), 'config', 'default_params.yaml')

    return LaunchDescription([
        Node(
            package='tb3_nav',
            executable='grid_nav_node',
            name='grid_nav_node',
            parameters=[config],
            output='screen',
        ),
    ])
