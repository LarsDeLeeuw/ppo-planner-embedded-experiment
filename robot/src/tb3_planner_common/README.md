# tb3_planner_common

Planner-neutral Python utilities shared by grid-navigation planner packages in this workspace.

## Contents

| Module | Symbol | Purpose |
|---|---|---|
| `tb3_planner_common.gridmap` | `gridmap_to_numpy(rows, cols, data)` | Convert a flat row-major `GridMap` payload to a 2D numpy array. Takes primitives, not the ROS message, so this package has no ROS runtime dependency. |
| `tb3_planner_common.directions` | `DIRECTION`, `ACTION_BY_DIRECTION` | Canonical 8-action ↔ `(dx, dy)` mapping. Single source of truth for the discrete action space. |

## Consumers

- [ppo_planner](../ppo_planner/)
- [astar_planner](../astar_planner/)

New planner packages should depend on `tb3_planner_common` (via `<exec_depend>` in `package.xml`) rather than duplicating these helpers.

## Dependencies

- `numpy >= 1.24`
