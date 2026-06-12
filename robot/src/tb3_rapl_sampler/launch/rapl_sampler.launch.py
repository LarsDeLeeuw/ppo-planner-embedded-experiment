"""Launch the RAPL sampler (desktop VM, centralized mode only)."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tb3_rapl_sampler',
            executable='rapl_sampler_node',
            name='rapl_sampler_node',
            output='screen',
        ),
    ])
