"""Always-on robot-SBC sensor layer (Pi 4B).

Launches the INA219 power-sensor node (which also publishes SBC thermal) and
the per-host diagnostics node. The Pi runs THIS in BOTH experiment modes,
started once per session and left up across runs and mode switches:

  - Centralized mode: this is essentially all the Pi runs (plus OpenCR
    bringup); the compute stack lives on the desktop VM.
  - Decentralized mode: this runs alongside bringup.launch.py on the Pi.
    Because diagnostics_node is here, launch bringup.launch.py with
    use_diagnostics:=false on the Pi to avoid a duplicate node.

To consolidate the Pi-side always-on layer into a single launch, this file
also includes the stock ``turtlebot3_bringup/robot.launch.py`` (OpenCR base
driver + LDS) by default. The include is guarded by the ``use_turtlebot3_base``
launch argument (default ``true``); set ``use_turtlebot3_base:=false`` for
benchtop testing without an OpenCR connected. The ``TURTLEBOT3_MODEL`` and
``LDS_MODEL`` environment variables are set inside the launch so callers do
not have to export them manually.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sensor_config = os.path.join(
        get_package_share_directory('tb3_power_sensor'), 'config', 'sensor_map.yaml')

    wifi_interface = LaunchConfiguration('wifi_interface')
    use_turtlebot3_base = LaunchConfiguration('use_turtlebot3_base')

    turtlebot3_base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_bringup'),
                'launch',
                'robot.launch.py',
            ]),
        ]),
        condition=IfCondition(use_turtlebot3_base),
    )

    return LaunchDescription([
        DeclareLaunchArgument('wifi_interface', default_value='wlan0'),
        DeclareLaunchArgument(
            'use_turtlebot3_base',
            default_value='true',
            description=(
                'Include the stock turtlebot3_bringup robot.launch.py '
                '(OpenCR base driver + LDS). Set to false for benchtop '
                'testing without OpenCR connected.'
            ),
        ),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        SetEnvironmentVariable(
            'LDS_MODEL', os.environ.get('LDS_MODEL', 'LDS-01')),
        turtlebot3_base_launch,
        Node(
            package='tb3_power_sensor',
            executable='power_sensor_node',
            name='power_sensor_node',
            parameters=[sensor_config],
            output='screen',
        ),
        Node(
            package='tb3_diagnostics',
            executable='diagnostics_node',
            name='diagnostics_node',
            parameters=[{'wifi_interface': wifi_interface}],
            output='screen',
        ),
    ])
