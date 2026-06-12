# tb3_bringup

One-stop launch package for the TurtleBot3 research stack. Starts `grid_nav_node`, an optional planner, and an optional TCP bridge based on launch arguments.

Think of this package as the **control surface**: all the knobs you'd want to flip between experiment runs live here. Individual-package `default_params.yaml` files hold the authoritative baselines; this package lets you override them per experiment without editing the packages themselves.

## Quick start

```bash
ros2 launch tb3_bringup bringup.launch.py
```

Launches, by default: `grid_nav_node` + `ppo_planner_node` + `bridge_node`, all reading from their package defaults with the overrides in [config/experiment.yaml](config/experiment.yaml) applied on top.

## Launch arguments

| Arg | Default | Values | Meaning |
|---|---|---|---|
| `planner` | `ppo` | `ppo` \| `astar` \| `astar_shortest` \| `none` | Which planner service to run. `astar` runs the energy-harvesting A\* variant (service at `/astar_planner_node/predict_action`); `astar_shortest` runs the shortest-path variant (service at `/astar_shortest_node/predict_action`). When switching to either A\* variant, also override `bridge_node.predict_service` in the experiment YAML to match. `none` skips the planner entirely (useful when testing nav alone). |
| `experiment_config` | `<share>/tb3_bringup/config/experiment.yaml` | file path | YAML file with per-experiment overrides applied on top of package defaults. Point this at any path to use a different overrides file. |
| `use_bridge` | `true` | `true` \| `false` | Whether to start the TCP bridge. Set `false` for nav-only testing without an external tracker. |
| `log_level` | `info` | `debug` \| `info` \| `warn` \| `error` | Log level applied to every node launched here. |

Examples:
```bash
# Nav-only, no bridge, no planner (for bench-testing odometry/IMU):
ros2 launch tb3_bringup bringup.launch.py planner:=none use_bridge:=false

# Different experiment overrides file:
ros2 launch tb3_bringup bringup.launch.py experiment_config:=/tmp/high_energy_run.yaml

# Verbose logs from every node:
ros2 launch tb3_bringup bringup.launch.py log_level:=debug
```

## Configuration hierarchy

Each node is fed parameters from these sources, **later wins on conflict**:

```
 1. package default_params.yaml   ← authoritative baseline, committed to git
 2. experiment_config YAML        ← per-experiment overrides, lives in this package
 3. command-line --ros-args -p x:=y  ← one-off tweaks
```

The goal: you should rarely need to edit package defaults directly. If you want to try a different `cell_size` for one experiment, add it to `experiment.yaml` or pass it on the command line — don't edit `tb3_nav/config/default_params.yaml`.

## Where the defaults live

| Node | File |
|---|---|
| `grid_nav_node` | [../tb3_nav/config/default_params.yaml](../tb3_nav/config/default_params.yaml) |
| `ppo_planner_node` | [../ppo_planner/config/default_params.yaml](../ppo_planner/config/default_params.yaml) |
| `astar_planner_node` / `astar_shortest_node` | [../astar_planner/config/default_params.yaml](../astar_planner/config/default_params.yaml) (shared via `/**:` wildcard) |
| `bridge_node` | [../ros2_bridge/config/default_params.yaml](../ros2_bridge/config/default_params.yaml) |

Each file is commented per-parameter. This README duplicates the parameter list below for a single-page reference.

## Parameters reference

### grid_nav_node (navigation)

| Param | Default | Meaning |
|---|---|---|
| `cell_size` | `0.33` | meters per grid cell |
| `max_move_distance` | `3.0` | safety limit on a single goal (m); aborts goals that would require driving further |
| `max_linear_speed` | `0.15` | m/s cap during driving |
| `min_linear_speed` | `0.05` | m/s floor (motor deadzone) |
| `max_angular_speed` | `0.8` | rad/s cap |
| `min_angular_speed` | `0.15` | rad/s floor |
| `kp_linear` | `1.0` | P-gain: distance → linear speed |
| `kp_angular` | `2.0` | P-gain: heading error → angular speed |
| `kp_heading_correction` | `0.2` | P-gain: heading drift correction during drive |
| `heading_tolerance` | `0.02` | rad (~1.1°); when rotation is "done" |
| `distance_tolerance` | `0.02` | m; when drive is "done" |
| `settling_time` | `0.3` | s; pause between rotate and drive to dissipate momentum |
| `move_timeout` | `30.0` | s; abort if a single move takes longer |
| `control_rate` | `20.0` | Hz; control loop frequency |
| `use_imu_heading` | `true` | use IMU for heading; falls back to odom if `false` |
| `linear_calibration` | `1.0` | per-surface drive scale = `actual_physical_distance / odom_reported_distance`. Set <1.0 when wheels slip (carpet). Use `ros2 run tb3_nav calibrate` to measure — see [docs/calibration.md](../../../docs/calibration.md). |
| `angular_calibration` | `1.0` | per-surface rotation scale. Only applied when `use_imu_heading: false` (the IMU is a gyro and is not affected by wheel slip). |
| `initial_grid_x` / `initial_grid_y` | `0` / `0` | starting cell coordinates |
| `initial_heading` | `0.0` | starting heading (rad); **must match the robot's physical heading at startup** |

### ppo_planner_node

| Param | Default | Meaning |
|---|---|---|
| `model_path` | `AIPPOm10EH_continued.onnx` | ONNX-exported PPO policy. Relative paths resolved against `<share>/ppo_planner/models/`; absolute paths used as-is. Runtime needs `numpy` + `onnxruntime` only (no torch). |
| `grid_size_x` / `grid_size_y` | `10` / `10` | must match the dimensions of maps sent in `PredictAction` requests |
| `num_robots` | `10` | training-time observation-space shape; do not change unless retraining |
| `deterministic` | `false` | `false` (default) samples from the policy distribution — how PPO navigates; `true` picks the max-probability action (argmax; tends to get stuck). |
| `seed` | `0` | RNG seed for stochastic sampling. `>= 0` reproduces across launches; negative draws fresh entropy. Ignored when `deterministic: true`. |

### astar_planner_node / astar_shortest_node

| Param | Default | Meaning |
|---|---|---|
| `variant` | `"energy"` (launched node chooses) | `"energy"` prefers energy-rich cells; `"shortest"` minimises geometric path length |
| `energy_weight` | `1.0` | Energy-harvest weight. Only used when `variant == "energy"` |
| `max_expansions` | `0` | Safety cap on A\* node expansions. `0` disables; positive values abort with `success=false` once exceeded |

### bridge_node

| Param | Default | Meaning |
|---|---|---|
| `tcp_port` | `9090` | TCP listen port |
| `poll_hz` | `20.0` | how often the bridge drains its TCP inbox (Hz) |
| `pose_service` | `/grid_nav_node/set_grid_pose` | `SetGridPose` service name |
| `goal_action` | `move_to_grid` | `MoveToGrid` action name |
| `grid_pose_topic` | `/grid_nav_node/grid_pose` | `GridPose` topic to forward to the TCP client |
| `predict_service` | `/ppo_planner_node/predict_action` | `PredictAction` service name. Change to `/astar_planner_node/predict_action` (via experiment YAML) when running with `planner:=astar`. |

## Writing an experiment YAML

`config/experiment.yaml` is the default target of `experiment_config`. It ships as a commented scaffold — uncomment the keys you want to override.

To keep it under version control but avoid merging every experiment into the file, either:
- edit it locally and don't commit, or
- copy it to `/tmp/myrun.yaml`, edit there, and pass `experiment_config:=/tmp/myrun.yaml`.

Example contents for a low-speed, tight-tolerance run:

```yaml
grid_nav_node:
  ros__parameters:
    max_linear_speed: 0.08
    heading_tolerance: 0.01
    distance_tolerance: 0.01

ppo_planner_node:
  ros__parameters: {}

bridge_node:
  ros__parameters: {}
```

## Adding a new planner

See [../astar_planner/README.md](../astar_planner/README.md) for the planner-service pattern A* uses — the template for any future planner.
