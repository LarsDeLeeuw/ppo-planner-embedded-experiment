"""Launch file for the A* planner node (energy variant, default)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('astar_planner'), 'config', 'default_params.yaml')

    return LaunchDescription([
        Node(
            package='astar_planner',
            executable='astar_planner_node',
            name='astar_planner_node',
            parameters=[config],
            output='screen',
        ),
    ])
