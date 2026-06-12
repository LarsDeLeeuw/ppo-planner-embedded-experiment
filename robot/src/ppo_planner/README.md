# ppo_planner

ROS2 node that wraps an **ONNX-exported** PPO policy as a native ROS2 service. Loads the model once at startup, then serves `PredictAction` requests synchronously.

Runtime dependencies are `numpy` + `onnxruntime` only — **no torch / stable-baselines3 / gymnasium**. That is what makes the policy deployable on the robot's on-board computer: `numpy` + `onnxruntime` is ~10-20 MB where the training stack is ~2 GB. The training stack is never imported at runtime; the committed `models/*.onnx` graph is the only artifact the node needs.

## Interface

| Interface | Type | Name | Description |
|---|---|---|---|
| `PredictAction` | Service | `~/predict_action` | Get next-step action for a given grid state |

### PredictAction request

| Field | Type | Description |
|---|---|---|
| `obstacle_map` | `GridMap` | 2D grid, `0.0` = free, `1.0` = occupied |
| `energy_map` | `GridMap` | 2D grid, floats in [0.0, 1.0] |
| `robot_x`, `robot_y` | `int32` | Robot's current grid cell |
| `goal_x`, `goal_y` | `int32` | Goal grid cell |

### PredictAction response

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether inference succeeded |
| `message` | `string` | Error description (empty on success) |
| `action` | `uint8` | Discrete action 0..7 (only valid if `success=true`) |
| `direction_x`, `direction_y` | `int8` | Direction vector (-1, 0, or 1) |

### Action-to-direction mapping

`(direction_x, direction_y)` — `direction_x` is the X (East-West) delta, `direction_y` is the Y (North-South) delta. Matches the GridMap layout pinned in [tb3_interfaces/msg/GridMap.msg](../tb3_interfaces/msg/GridMap.msg) (rows index X, cols index Y).

| Action | Direction | Cardinal |
|--------|-----------|----------|
| 0 | (-1,  0) | West |
| 1 | ( 1,  0) | East |
| 2 | ( 0, -1) | South |
| 3 | ( 0,  1) | North |
| 4 | (-1, -1) | SW |
| 5 | (-1,  1) | NW |
| 6 | ( 1, -1) | SE |
| 7 | ( 1,  1) | NE |

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `AIPPOm10EH_continued.onnx` | Path to the ONNX-exported PPO policy. If relative, resolved against `<package_share>/models/` |
| `grid_size_x` | int | `10` | Grid width. Must match the maps sent in requests |
| `grid_size_y` | int | `10` | Grid height. Must match the maps sent in requests |
| `num_robots` | int | `10` | Number of robots the model was trained with (observation-space shape) |
| `deterministic` | bool | `false` | If `false` (default), sample from the policy distribution — `softmax(logits)` in numpy. If `true`, always pick the highest-probability action (argmax). |
| `seed` | int | `0` | RNG seed for stochastic sampling. `>= 0` reproduces across launches; negative draws fresh entropy each launch. Ignored when `deterministic: true`. |

## Why stochastic by default

PPO was trained with stochastic action sampling. Under pure argmax (`deterministic: true`) the policy greedily steps toward the goal and **walks straight into walls and oscillates next to the goal without landing on it** — so it frequently fails on maps that are perfectly solvable. Stochastic sampling is how the policy actually navigates, so it is the default here. Set `deterministic: true` only to inspect the single greedy trajectory.

The ONNX graph emits both a deterministic `action` (argmax, baked in) and the raw `logits` `(B, 10, 8)`. With `deterministic: false` the node reads `logits` and samples Robot 0 from `softmax(logits[0, 0])` with `numpy.random.Generator.choice` — equivalent to `torch.distributions.Categorical(logits=...).sample()`, but without torch.

## Model details

- Trained with Stable-Baselines3 PPO (10-robot scenario), then exported to ONNX (the `.onnx` is committed; the training stack is not needed at runtime).
- Observation: local 10x10 window centered on the robot.
- Single-robot adaptation: the other 9 robots are stacked on the querying robot's cell, making them invisible to the policy.
- Works mechanically on any grid size (padding for small grids, windowing for large), though out-of-distribution sizes may drift in behavior. The physical experiment fixes the grid at 10x10, so the policy sees the whole map in one window.

## Usage

### Build
```bash
cd ~/ros2_ws
colcon build --packages-select ppo_planner
source install/setup.bash
```

### Run
```bash
ros2 launch ppo_planner ppo_planner.launch.py
```

Or with custom parameters:
```bash
ros2 run ppo_planner ppo_planner_node --ros-args \
  -p model_path:=/absolute/path/to/model.onnx \
  -p deterministic:=true
```

## Dependencies

- `rclpy`
- `tb3_interfaces` (provides `PredictAction`, `GridMap`)
- `tb3_planner_common` (provides the shared `DIRECTION` action table)
- `numpy >= 1.21`
- `onnxruntime >= 1.17, < 2.0`

`onnxruntime` is not packaged for `rosdep`; install it with pip on whichever
host runs this node (the desktop VM in centralized mode, or the robot's
on-board computer in decentralized mode):

```bash
python3 -m pip install "onnxruntime>=1.17,<2.0"
```

There is no torch / stable-baselines3 / gymnasium dependency.
