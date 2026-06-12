"""Launch file for the A* planner node configured as the shortest-path variant.

Same executable as astar_planner.launch.py; this file overrides the node name
and `variant` parameter so the service lives at /astar_shortest_node/predict_action
and the search ignores the energy map.
"""

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
            name='astar_shortest_node',
            parameters=[config, {'variant': 'shortest'}],
            output='screen',
        ),
    ])
