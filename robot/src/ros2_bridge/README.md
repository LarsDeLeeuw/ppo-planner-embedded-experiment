# ros2_bridge

TCP-to-ROS2 bridge for external grid navigation clients. Receives pose corrections, navigation goals, and PPO predict requests over a TCP socket and forwards them to `grid_nav_node` and `ppo_planner_node` via ROS2 services and actions.

## Architecture

```
[External client]       TCP/JSON        [bridge_node]         ROS2         [grid_nav_node]
(e.g. qr-tracker)  <--------------->   (this package)  -------------->   (navigation)
                                              |
                                              |           ROS2         [ppo_planner_node]
                                              +---------------------->   (PPO inference)
```

The bridge node is a thin adapter — it does not perform coordinate transforms or validation. Clients are responsible for sending coordinates in the ROS2 convention described below.

## Wire Protocol

Newline-delimited JSON (`\n`-terminated) over a single TCP connection.

### Coordinate Convention (ROS2)

All coordinates on the wire use the ROS2 grid_nav_node convention:

| Property | Convention |
|---|---|
| Grid origin | `(0, 0)` = bottom-left cell |
| X axis | Increases right (East) |
| Y axis | Increases up (North) |
| Heading 0 | Facing +X (East) |
| Heading direction | Counter-clockwise positive (radians) |

### Client → Server

**Pose correction** — update the robot's believed position:
```json
{"type": "pose", "x": 1.5, "y": 2.5, "heading": 1.57}
```
- `x`, `y`: grid coordinates (sub-cell precision)
- `heading`: radians, CCW from +X

**Navigation goal** — drive to a grid cell:
```json
{"type": "goal", "target_x": 2, "target_y": 1}
```
- `target_x`, `target_y`: integer cell coordinates
- Sending a new goal preempts any active goal

**Cancel goal** — abort the active navigation:
```json
{"type": "cancel_goal"}
```

**PPO predict** — request next-step action from the learned policy:
```json
{"type": "predict", "obstacle_map": [[0,1,0,...], ...], "energy_map": [[0.5,...], ...], "robot_pos": [2, 3], "goal_pos": [8, 7]}
```
- `obstacle_map`: 2D list (rows x cols), `0` = free, `1` = occupied
- `energy_map`: 2D list (rows x cols), floats in [0.0, 1.0]
- `robot_pos`: `[x, y]` current grid cell
- `goal_pos`: `[x, y]` goal grid cell

**Ping** — connectivity check:
```json
{"type": "ping"}
```

### Server → Client

**Goal feedback** — periodic progress updates (~20 Hz):
```json
{"type": "goal_feedback", "phase": "rotating", "distance": 0.3, "heading_error": 0.1}
```
- `phase`: one of `"idle"`, `"rotating"`, `"settling"`, `"driving"`, `"finalizing"`, `"done"`, `"aborted"` (lowercase name of the internal state machine)
- `distance`: remaining distance in grid units
- `heading_error`: remaining heading error in radians

**Goal result** — sent when the goal completes or fails:
```json
{"type": "goal_result", "success": true, "message": "Reached target"}
```

**Nav pose** — robot's self-estimated position (from grid_nav_node):
```json
{"type": "nav_pose", "x": 1.8, "y": 1.2, "heading": 0.5}
```

**Predict result** — PPO action response:
```json
{"type": "predict_result", "action": 3, "direction": [0, 1]}
```
- `action`: discrete action 0..7
- `direction`: `[dx, dy]` direction vector (each -1, 0, or 1)

**Error** — sent on service failures or malformed requests:
```json
{"type": "error", "message": "PredictAction service not ready"}
```

**Pong** — response to ping:
```json
{"type": "pong"}
```

### Forward Compatibility

Unknown `type` values are logged and ignored on both sides. New message types can be added without breaking existing clients.

## ROS2 Interfaces

| Interface | Type | Default Name | Description |
|---|---|---|---|
| `SetGridPose` | Service (client) | `/grid_nav_node/set_grid_pose` | Correct robot position |
| `MoveToGrid` | Action (client) | `move_to_grid` | Navigate to a grid cell |
| `GridPose` | Topic (subscription) | `/grid_nav_node/grid_pose` | Robot self-estimate |
| `PredictAction` | Service (client) | `/ppo_planner_node/predict_action` | PPO policy inference |

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tcp_port` | int | `9090` | TCP server listen port |
| `pose_service` | string | `/grid_nav_node/set_grid_pose` | SetGridPose service name |
| `goal_action` | string | `move_to_grid` | MoveToGrid action name |
| `grid_pose_topic` | string | `/grid_nav_node/grid_pose` | GridPose topic name |
| `predict_service` | string | `/ppo_planner_node/predict_action` | PredictAction service name |
| `poll_hz` | float | `20.0` | TCP inbox poll rate |

## Usage

### Build
```bash
cd ~/ros2_ws
colcon build --packages-select ros2_bridge
source install/setup.bash
```

### Run
```bash
ros2 run ros2_bridge bridge_node
```

With custom parameters:
```bash
ros2 run ros2_bridge bridge_node --ros-args \
  -p tcp_port:=9090 \
  -p pose_service:=/grid_nav_node/set_grid_pose \
  -p goal_action:=move_to_grid \
  -p predict_service:=/ppo_planner_node/predict_action
```

### Test with netcat
```bash
# Send a pose correction
echo '{"type":"pose","x":1.5,"y":2.5,"heading":1.57}' | nc robot_ip 9090

# Send a navigation goal
echo '{"type":"goal","target_x":2,"target_y":1}' | nc robot_ip 9090

# Request PPO prediction
echo '{"type":"predict","obstacle_map":[[0,0],[0,0]],"energy_map":[[0.5,0.5],[0.5,0.5]],"robot_pos":[0,0],"goal_pos":[1,1]}' | nc robot_ip 9090
```

## Dependencies

- `rclpy`
- `tb3_interfaces` (provides `MoveToGrid`, `SetGridPose`, `GridPose`, `PredictAction`, `GridMap`)
