# Rollout — Scenario Input Spec

The format `scripts/rollout/rollout.py` expects: **one JSON file** with all four
inputs (obstacle map, energy map, robot start, goal). A working example is in
[`examples/example_scenario.json`](examples/example_scenario.json).

The convention is the **repo `[x][y]` frame** (see
[docs/coordinate-conventions.md](../../../docs/coordinate-conventions.md)) — the exact
layout the orchestrator sends planners over the bridge, so what you visualize
here is what the planners actually receive on the robot.

## The bundle

```jsonc
{
  "name": "my_map_01",        // optional, COSMETIC. Output files are named after the
                              // scenario FILENAME (or --name), not this field.

  "obstacle_map": [           // REQUIRED. 2D array indexed [x][y].
    [0, 0, 1, 0, ...],        //   obstacle_map[x][y]: 0 = free, non-zero = obstacle.
    ...                       //   First index = x (East), second = y (North).
  ],                          //   Every column (inner list) must be the same length.

  "energy_map": [             // REQUIRED. 2D array, SAME shape as obstacle_map.
    [0.15, 0.22, ...],        //   floats in [0, 1] (higher = brighter = better harvesting).
    ...                       //   Out-of-range values are clipped (with a warning).
  ],

  "robot": [1, 1],            // REQUIRED. Integer start cell [x, y].
  "goal":  [8, 1],            // REQUIRED. Integer goal cell  [x, y].

  "max_steps": 200            // optional; rollout step cap. Default = nx*ny + 10.
}
```

## Conventions — read carefully

### Coordinate frame: repo `[x][y]`, origin bottom-left
- Maps are indexed `array[x][y]`. **First index = x** (East, `+x` → East).
  **Second index = y** (North, `+y` → North). Origin `(0, 0)` = bottom-left.
- `robot` and `goal` are `[x, y]` integer cells in the same frame.
- The rendered PNGs use this frame: **x increases to the right, y upward**.
- This is rotated 90° from image convention. To place an obstacle at world cell
  `(x=2, y=1)`:

  ```python
  grid = [[0]*ny for _ in range(nx)]   # grid[x][y]
  grid[2][1] = 1
  ```

  If you instinctively wrote `grid[1][2] = 1`, you are on image convention —
  transpose before saving (`np.array(grid).T.tolist()`).

### Obstacle values
- `0` = free / traversable; any non-zero value = obstacle (thresholded `> 0`).

### Energy values
- Floats already normalized to `[0, 1]` (`0` = dark, `1` = brightest). Same
  shape as `obstacle_map`. Out-of-range values are clipped.

### Map size
- Any rectangular size (`nx × ny`). The physical experiment fixes it at
  **10 × 10** — that is also where PPO's local 10×10 window equals the whole map.

## Action → step (repo action table)

| action | (dx, dy) | cardinal |
|:--:|:--:|:--|
| 0 | (−1, 0) | West |
| 1 | (+1, 0) | East |
| 2 | (0, −1) | South |
| 3 | (0, +1) | North |
| 4 | (−1, −1) | SW |
| 5 | (−1, +1) | NW |
| 6 | (+1, −1) | SE |
| 7 | (+1, +1) | NE |

Next cell = `(x + dx, y + dy)`.

## ⚠️ PPO: stochastic vs argmax

PPO was trained with **stochastic sampling**. Under argmax (`--ppo-deterministic`)
the policy greedily steps toward the goal, **walks into walls**, and oscillates
near the goal — so it often reports `blocked`/`loop` on solvable maps. The tool
therefore runs PPO **stochastically by default** (`--trials` runs overlaid, with
a success rate). Use `--ppo-deterministic` only to inspect the single greedy
trajectory + failure diagnostics. PPO is also purely **local** (10×10 window,
goal clipped to its edge) — on grids larger than 10 it reacts to goal direction,
not true distance.
