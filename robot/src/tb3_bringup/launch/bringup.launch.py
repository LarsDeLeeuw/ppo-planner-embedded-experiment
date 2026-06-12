"""Launch all TurtleBot3 research nodes.

Launches, by default: grid_nav_node + ppo_planner_node + bridge_node.

Arguments:
  planner             Which planner to run: 'ppo' | 'astar' | 'astar_shortest' | 'none'.
                      Default: ppo
                        - 'ppo'            -> PPO planner
                          (service: /ppo_planner_node/predict_action)
                        - 'astar'          -> A* energy-harvesting variant
                          (service: /astar_planner_node/predict_action)
                        - 'astar_shortest' -> A* shortest-path variant
                          (service: /astar_shortest_node/predict_action)
                      The bridge's `predict_service` parameter is set
                      automatically from this arg (see
                      PLANNER_TO_PREDICT_SERVICE in this file). No
                      experiment.yaml edit needed when switching planners.
  bridge_predict_service
                      Optional explicit override for the bridge's
                      predict_service. Empty default => derived from
                      `planner`. Default: ''
  experiment_config   Path to an overrides YAML loaded on top of each
                      package's default_params.yaml.                    Default: <share>/tb3_bringup/config/experiment.yaml
  use_bridge          Whether to start the TCP bridge.                  Default: true
  use_diagnostics     Start a per-host diagnostics_node (chrony/WiFi/RAPL/
                      git/experiment.yaml). Set true on the host running the
                      compute stack that ISN'T already running
                      robot_sensors.launch.py — i.e. the desktop VM in
                      centralized mode. On the Pi the always-on
                      robot_sensors.launch.py already provides it, so leave
                      this false there to avoid a duplicate node.   Default: false
  use_rapl            Start the RAPL CPU-energy sampler. Desktop VM only
                      (centralized mode).                            Default: false
  log_level           Log level for every node here.                    Default: info

Examples:
  # Decentralized mode, on the Pi (robot_sensors.launch.py runs separately):
  ros2 launch tb3_bringup bringup.launch.py planner:=ppo use_bridge:=true
  # Centralized mode, on the desktop VM:
  ros2 launch tb3_bringup bringup.launch.py planner:=ppo use_bridge:=true \
      use_diagnostics:=true use_rapl:=true
  ros2 launch tb3_bringup bringup.launch.py planner:=none use_bridge:=false

See src/tb3_bringup/README.md for the full parameter reference.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Map planner launch arg -> bridge.predict_service. Centralized here so the
# launch arg drives the bridge wiring (experiment.yaml no longer needs to).
# Keep in sync with the per-planner Node names below.
PLANNER_TO_PREDICT_SERVICE = {
    'ppo':            '/ppo_planner_node/predict_action',
    'astar':          '/astar_planner_node/predict_action',
    'astar_shortest': '/astar_shortest_node/predict_action',
    'none':           '',
}


def generate_launch_description():
    # -- Resolve per-package default parameter files ------------------------
    nav_default = os.path.join(
        get_package_share_directory('tb3_nav'),
        'config', 'default_params.yaml')
    ppo_default = os.path.join(
        get_package_share_directory('ppo_planner'),
        'config', 'default_params.yaml')
    astar_default = os.path.join(
        get_package_share_directory('astar_planner'),
        'config', 'default_params.yaml')
    bridge_default = os.path.join(
        get_package_share_directory('ros2_bridge'),
        'config', 'default_params.yaml')

    # Default experiment-overrides YAML (scaffold shipped with this package).
    default_experiment = os.path.join(
        get_package_share_directory('tb3_bringup'),
        'config', 'experiment.yaml')

    # -- Launch arguments ---------------------------------------------------
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='ppo',
        description="Which planner to run: 'ppo' | 'astar' | 'astar_shortest' | 'none'",
    )
    experiment_arg = DeclareLaunchArgument(
        'experiment_config',
        default_value=default_experiment,
        description='Path to overrides YAML applied on top of each package default',
    )
    use_bridge_arg = DeclareLaunchArgument(
        'use_bridge',
        default_value='true',
        description='Whether to start the TCP bridge node',
    )
    use_diagnostics_arg = DeclareLaunchArgument(
        'use_diagnostics',
        default_value='false',
        description='Start a diagnostics_node here (true on the desktop in centralized mode)',
    )
    use_rapl_arg = DeclareLaunchArgument(
        'use_rapl',
        default_value='false',
        description='Start the RAPL CPU-energy sampler here (desktop VM, centralized mode)',
    )
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for every node launched here (debug|info|warn|error)',
    )
    # Empty default => the OpaqueFunction below derives it from `planner`. Set
    # this explicitly on the CLI to override (rare — useful for ad-hoc bridge
    # rewiring to a custom service name).
    bridge_predict_service_arg = DeclareLaunchArgument(
        'bridge_predict_service',
        default_value='',
        description=(
            "Override the bridge's predict_service. Empty (default) => derived "
            "from `planner` via PLANNER_TO_PREDICT_SERVICE in this launch file."
        ),
    )

    experiment_config = LaunchConfiguration('experiment_config')
    log_level = LaunchConfiguration('log_level')

    # Common --log-level injection for every Node below.
    log_args = ['--ros-args', '--log-level', log_level]

    # -- Nodes --------------------------------------------------------------
    # Package defaults are loaded first; experiment overrides are loaded
    # second and win on conflict. Command-line -p args win over both.
    nav_node = Node(
        package='tb3_nav',
        executable='grid_nav_node',
        name='grid_nav_node',
        parameters=[nav_default, experiment_config],
        arguments=log_args,
        output='screen',
    )

    ppo_node = Node(
        package='ppo_planner',
        executable='ppo_planner_node',
        name='ppo_planner_node',
        parameters=[ppo_default, experiment_config],
        arguments=log_args,
        output='screen',
        condition=LaunchConfigurationEquals('planner', 'ppo'),
    )

    # A* planner — energy-harvesting variant (default A* mode).
    astar_node = Node(
        package='astar_planner',
        executable='astar_planner_node',
        name='astar_planner_node',
        parameters=[astar_default, experiment_config],
        arguments=log_args,
        output='screen',
        condition=LaunchConfigurationEquals('planner', 'astar'),
    )

    # A* planner — shortest-path variant (same executable, different node name
    # and variant override so the service lives at /astar_shortest_node/...).
    astar_shortest_node = Node(
        package='astar_planner',
        executable='astar_planner_node',
        name='astar_shortest_node',
        parameters=[astar_default, {'variant': 'shortest'}, experiment_config],
        arguments=log_args,
        output='screen',
        condition=LaunchConfigurationEquals('planner', 'astar_shortest'),
    )

    def _make_bridge_node(context):
        """Build the bridge Node with predict_service resolved from `planner`.

        Override priority (later wins, per ROS 2 param resolution):
          bridge package default_params.yaml
            -> experiment.yaml (user overrides)
            -> inline {'predict_service': <resolved>}   <-- always set here
        Setting `bridge_predict_service:=...` on the CLI bypasses the planner
        map; otherwise we look up PLANNER_TO_PREDICT_SERVICE[planner].
        """
        planner_val = LaunchConfiguration('planner').perform(context)
        override = LaunchConfiguration('bridge_predict_service').perform(context)

        if override:
            predict_service = override
        else:
            if planner_val not in PLANNER_TO_PREDICT_SERVICE:
                raise RuntimeError(
                    f"Unknown planner '{planner_val}'. Expected one of "
                    f"{sorted(PLANNER_TO_PREDICT_SERVICE)}."
                )
            predict_service = PLANNER_TO_PREDICT_SERVICE[planner_val]

        # planner:=none => no planner running; don't pin the bridge to a stale
        # service name. Fall back to whatever experiment.yaml / package default
        # provide (which may itself be empty — fine for bridge-only smoke tests).
        params = [bridge_default, experiment_config]
        if predict_service:
            params.append({'predict_service': predict_service})

        return [Node(
            package='ros2_bridge',
            executable='bridge_node',
            name='bridge_node',
            parameters=params,
            arguments=log_args,
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_bridge')),
        )]

    bridge_node = OpaqueFunction(function=_make_bridge_node)

    # Per-host diagnostics (only when this host isn't already running
    # robot_sensors.launch.py — i.e. the desktop VM in centralized mode).
    diagnostics_node = Node(
        package='tb3_diagnostics',
        executable='diagnostics_node',
        name='diagnostics_node',
        arguments=log_args,
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_diagnostics')),
    )

    # RAPL CPU-energy sampler (desktop VM, centralized mode).
    rapl_node = Node(
        package='tb3_rapl_sampler',
        executable='rapl_sampler_node',
        name='rapl_sampler_node',
        arguments=log_args,
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rapl')),
    )

    return LaunchDescription([
        planner_arg,
        experiment_arg,
        use_bridge_arg,
        use_diagnostics_arg,
        use_rapl_arg,
        log_level_arg,
        bridge_predict_service_arg,
        nav_node,
        ppo_node,
        astar_node,
        astar_shortest_node,
        bridge_node,
        diagnostics_node,
        rapl_node,
    ])
