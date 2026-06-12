# astar_planner

ROS2 node wrapping an energy-aware A\* search as a native `PredictAction` service. Drop-in alternative to `ppo_planner` — identical service contract, chosen at launch time via the `tb3_bringup` `planner` argument.

## Interface

| Interface | Type | Name | Description |
|---|---|---|---|
| `PredictAction` | Service | `~/predict_action` | Get next-step action for a given grid state |

Same request/response fields as [ppo_planner](../ppo_planner/README.md). The same `action` integers mean the same directions in both planners (see [tb3_planner_common/directions.py](../tb3_planner_common/tb3_planner_common/directions.py)).

## Variants

Selected via the `variant` ROS parameter:

| Variant | Cost function | Optimality |
|---|---|---|
| `energy` (default) | `Σ (c_step − energy_weight · EH)` (paper Eq. 34) | A\*-optimal for this cost: uses the paper's `(1 − energy_weight)`-scaled octile heuristic (Eq. 35). Admissible/consistent because energy is contractually normalised to `[0, 1]` (clipped to that range) and `energy_weight (= α) ∈ [0, 1]`, so every edge cost `≥ 1 − α ≥ 0`. Keep `energy_weight ≤ 1`; larger values make the heuristic inadmissible and the search greedy again. |
| `shortest` | `distance` only | Provably optimal (admissible octile heuristic). |

The bringup launch exposes both under distinct node names:

- `planner:=astar` → `/astar_planner_node/predict_action` (variant: `energy`)
- `planner:=astar_shortest` → `/astar_shortest_node/predict_action` (variant: `shortest`)

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `variant` | string | `"energy"` | `"energy"` or `"shortest"` |
| `energy_weight` | float | `0.5` | Energy-harvest weight (`alpha`, paper Eq. 34). Keep in `[0, 1]` to preserve the admissible heuristic. Used only when `variant == "energy"` |
| `max_expansions` | int | `0` | Safety cap on A\* node expansions. `0` disables; positive values abort with `success=false` once exceeded |

The shipped `config/default_params.yaml` uses a `/**:` wildcard namespace so the same file binds to whichever node name the launch file chooses.

## Usage

### Build
```bash
cd ~/ros2_ws
colcon build --packages-select tb3_planner_common astar_planner
source install/setup.bash
```

### Run standalone
```bash
# energy variant (default)
ros2 launch astar_planner astar_planner.launch.py

# shortest-path variant
ros2 launch astar_planner astar_shortest.launch.py
```

Or as an executable with overrides:
```bash
ros2 run astar_planner astar_planner_node --ros-args \
  -p variant:=energy -p energy_weight:=2.0 -p max_expansions:=10000
```

### Run via bringup
```bash
ros2 launch tb3_bringup bringup.launch.py planner:=astar            # energy
ros2 launch tb3_bringup bringup.launch.py planner:=astar_shortest   # shortest
```
The bridge's `predict_service` parameter (set in `experiment.yaml`) must point at the active service path.

## Testing

Algorithm unit tests are pure-Python — no ROS sourcing needed:
```bash
cd src/astar_planner
python -m pytest test/
```

## Dependencies

- `rclpy`
- `tb3_interfaces` (provides `PredictAction`, `GridMap`)
- `tb3_planner_common` (shared `gridmap_to_numpy` + `DIRECTION` table)
- `numpy >= 1.24`
