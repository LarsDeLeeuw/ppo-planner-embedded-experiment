# Grid conventions

All grid-based interfaces in this repo (`PredictAction`, `MoveToGrid`, `GridPose`, `GridMap`) follow **one** convention. Producers and consumers must respect it end-to-end or maps and directions come out 90° rotated.

## Axes

- Origin `(0, 0)` is at the **bottom-left** of the world.
- **X = East** (`+X` = moving East).
- **Y = North** (`+Y` = moving North).
- Heading: ROS convention — `0 rad = +X` (East), CCW positive.

## GridMap layout (this is where most bugs come from)

A `GridMap` (obstacle map, energy map) is row-major. The cell at grid `(x, y)` lives at:

```
flat_index = x * cols + y
2D form:    grid[x][y]
```

- **First index = X** (East-West). `rows` = X-extent.
- **Second index = Y** (North-South). `cols` = Y-extent.

This is **rotated 90° from the image convention** (where rows index Y, top-down). If you build the map as `grid[y][x]`, the planner reads a transposed world: it'll either flag the robot's free cell as an obstacle, or return directions that drive the robot perpendicular to where you expect.

## Sending a predict request

```python
# `obstacle_map` is a 2D list indexed [x][y]
send({
    "type":         "predict",
    "obstacle_map": obstacle_map,   # 0.0 = free, 1.0 = occupied
    "energy_map":   energy_map,     # floats in [0, 1]
    "robot_pos":    [robot_x, robot_y],   # ints, X=East, Y=North
    "goal_pos":     [goal_x,  goal_y],
})
```

The bridge does no transforms — what you send is what the planner sees.

## Reading the response

```json
{"action": 7, "direction": [direction_x, direction_y]}
```

- `direction_x`: X delta in `{-1, 0, +1}`.
- `direction_y`: Y delta in `{-1, 0, +1}`.
- Compose the next cell: `(robot_x + direction_x, robot_y + direction_y)`.

Action ↔ direction:

| action | (dx, dy) | cardinal |
|---|---|---|
| 0 | (-1,  0) | West  |
| 1 | ( 1,  0) | East  |
| 2 | ( 0, -1) | South |
| 3 | ( 0,  1) | North |
| 4 | (-1, -1) | SW    |
| 5 | (-1,  1) | NW    |
| 6 | ( 1, -1) | SE    |
| 7 | ( 1,  1) | NE    |

## Sending a movement goal

After composing the next cell:

```python
send({"type": "goal", "target_x": next_x, "target_y": next_y})
```

The nav node drives to grid `(target_x, target_y)` using the same X=East, Y=North convention. Floats are accepted (sub-cell precision).

## Reading the robot's pose

The bridge republishes the nav node's self-estimate over TCP:

```json
{"type": "nav_pose", "x": <X grid>, "y": <Y grid>, "heading": <rad>}
```

`x` is the X (East) grid coordinate, `y` is the Y (North) grid coordinate. Both can be sub-cell. `heading = 0` means facing East.

## Quick self-check

You want to mark one obstacle at world cell `(x=2, y=1)` on a 4×4 grid. The right Python list:

```python
grid = [[0]*4 for _ in range(4)]
grid[2][1] = 1
```

If your code instinctively produces `grid[1][2] = 1` for "obstacle at (2, 1)", you're on image convention — transpose before sending (`np.array(grid).T.tolist()`).
