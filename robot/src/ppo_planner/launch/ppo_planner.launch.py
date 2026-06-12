"""Launch file for the PPO planner node."""

import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('ppo_planner'), 'config', 'default_params.yaml')

    return LaunchDescription([
        Node(
            package='ppo_planner',
            executable='ppo_planner_node',
            name='ppo_planner_node',
            parameters=[config],
            output='screen',
        ),
    ])
