# Grid Navigation Integration Guide

How external programs communicate with the `grid_nav_node`.

## Interfaces

| Interface | Type | Name | When to use |
|---|---|---|---|
| `MoveToGrid` | Action | `move_to_grid` | Send the robot to a grid cell |
| `SetGridPose` | Service | `~/set_grid_pose` | Correct the robot's believed position |
| `GridPose` | Topic | `~/grid_pose` | Monitor where the robot thinks it is |

## Recommended integration flow

```
1. [Optional] Call SetGridPose to correct drift from external localization
2. Send MoveToGrid goal with target cell
3. Monitor feedback (phase, distance remaining, heading error)
4. Receive result (success/failure, final pose)
5. Repeat from 1
```

## What the external program needs

**Minimum viable integration** -- just send goals and wait for results:

```python
from rclpy.action import ActionClient
from tb3_interfaces.action import MoveToGrid

client = ActionClient(node, MoveToGrid, 'move_to_grid')
client.wait_for_server()

goal = MoveToGrid.Goal(target_x=2, target_y=1)
future = client.send_goal_async(goal)
# ... handle result via callback or await
```

**With external localization** (e.g. overhead camera) -- correct pose before each move:

```python
from tb3_interfaces.srv import SetGridPose
from tb3_interfaces.msg import GridPose

pose_client = node.create_client(SetGridPose, '/grid_nav_node/set_grid_pose')

# Correct position from camera observation
req = SetGridPose.Request()
req.pose = GridPose(x=1.0, y=0.0, heading=1.57)
pose_client.call_async(req)

# Then send the navigation goal
goal = MoveToGrid.Goal(target_x=2, target_y=1)
client.send_goal_async(goal)
```

**Passive monitoring** -- subscribe to the robot's self-estimate without sending commands:

```python
from tb3_interfaces.msg import GridPose

node.create_subscription(GridPose, '/grid_nav_node/grid_pose', callback, 10)
```

## Behavior the external program should expect

- **One goal at a time.** Sending a new goal cancels the active one (preempt-and-replace). The old goal's result will report `success=False, message="Preempted"`.
- **Goals are absolute grid coordinates.** If the external program wants relative movement ("go 1 cell north"), it should read `~/grid_pose`, compute `target = current + offset`, and send the absolute target.
- **No bounds checking.** The nav node does not know the grid dimensions. It will attempt to drive to any `(target_x, target_y)`. The external program is responsible for validating goals. The only safety net is `max_move_distance` (default 3m) which aborts goals that require driving further than that.
- **Feedback is published at 20Hz** (configurable). The `phase` field tells you what the robot is doing: `"rotating"` -> `"driving"` -> `"done"`.
- **Failure modes**: timeout (robot stuck), max distance exceeded (bad goal), canceled, preempted. All reported via `result.success=False` with a descriptive `result.message`.

## Coordinate contract

The external program and the nav node must agree on:

| Property | Convention |
|---|---|
| Grid origin | `(0, 0)` = robot's starting cell |
| X axis | Increases to the right (East) |
| Y axis | Increases upward (North) |
| Heading 0 | Facing +X (East) |
| Heading direction | Counter-clockwise positive (radians) |
| Cell size | Must match `cell_size` parameter (default 0.33m) |

## Namespace awareness

All topic/service/action names are relative. If the nav node is launched in a namespace (e.g. `namespace:=robot1`), the full names become `/robot1/move_to_grid`, `/robot1/grid_nav_node/set_grid_pose`, etc. The external program should use the same namespace or use the fully-qualified names.
