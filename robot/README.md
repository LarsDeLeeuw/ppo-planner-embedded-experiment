# robot

The **ROS 2 (Humble) workspace** that runs on the TurtleBot3 Burger (and, in centralized mode, on a
desktop). It contains the planners, the closed-loop grid navigation, the sensor + diagnostics nodes,
and the TCP bridge that the overhead [`tracker/`](../tracker/) talks to.

Treat **`robot/` as the workspace root** — the scripts here expect `src/` and `scripts/` directly
beneath them.

## Packages (`src/`)

| Package | Role |
|---|---|
| `tb3_interfaces` | **Canonical** message/service/action definitions (`PredictAction`, `MoveToGrid`, `GridPose`, power/thermal/diagnostics msgs, …). Also consumed by `tracker/` and `analysis/` for bag I/O. |
| `tb3_planner_common` | Shared, ROS-free helpers (8-direction table, GridMap→numpy). |
| `ppo_planner` | Learned policy planner — runs the PhD student's PPO model via ONNX (`PredictAction` service). |
| `astar_planner` | Energy-aware / shortest-path A* baseline (same `PredictAction` contract — drop-in swappable). |
| `tb3_nav` | Closed-loop grid navigation action server (`MoveToGrid`): rotate → settle → drive, IMU/odom fusion. |
| `ros2_bridge` | Thin TCP/JSON gateway (port 9090) between the tracker/orchestrator and the ROS graph. |
| `tb3_power_sensor` | Reads 3× INA219 (solar / SBC / motor) + SBC thermal over I2C; always-on on the robot. |
| `tb3_diagnostics` | Per-host ambient telemetry (chrony offset, WiFi RSSI, RAPL availability, git SHA, experiment.yaml). |
| `tb3_rapl_sampler` | Intel RAPL CPU-energy counters (desktop / centralized mode only). |
| `tb3_bringup` | Unified launcher — selects the planner and toggles bridge / diagnostics / RAPL; merges `config/experiment.yaml`. |

The two planners expose the **same `PredictAction` service**, so the experiment swaps PPO ↔ A* by
changing one launch argument.

## Build

```bash
cd robot
./scripts/build_all.sh            # symlinks the packages into the colcon workspace and builds
source ~/turtlebot3_ws/install/setup.bash
```

## Run

```bash
# Decentralized — the robot runs everything (planner + nav + bridge + sensors)
ros2 launch tb3_bringup bringup.launch.py planner:=ppo

# swap the planner
ros2 launch tb3_bringup bringup.launch.py planner:=astar          # energy-aware A*
ros2 launch tb3_bringup bringup.launch.py planner:=astar_shortest # shortest-path A*
```

The always-on sensor layer (power + diagnostics) launches separately on the robot:

```bash
ros2 launch tb3_bringup robot_sensors.launch.py
```

**Centralized mode** runs the compute stack (planner, bridge, RAPL, diagnostics) on a desktop while
the robot publishes only sensors — enable the desktop-side extras:

```bash
ros2 launch tb3_bringup bringup.launch.py planner:=ppo use_diagnostics:=true use_rapl:=true
```

`config/experiment.yaml` in `tb3_bringup` is the single place to override per-experiment parameters;
see [`src/tb3_bringup/README.md`](src/tb3_bringup/README.md).

## Deploy to the robot

```bash
./scripts/ssh_deploy.sh          # rsyncs the workspace to the robot and rebuilds there
```

Deploy targets and paths are **environment-overridable** and currently default to placeholders:

| Variable | Default | Meaning |
|---|---|---|
| `ROBOT_HOST` | `ubuntu@turtlebot3.local` | SSH target for the robot SBC |
| `REMOTE_DIR` | `~/ppo-tb3-research-project/robot` | where this workspace lives on the robot |

> If your robot deploy layout differs (different hostname, clone path, or you deploy only `robot/`),
> set these before running the scripts — e.g. `ROBOT_HOST=ubuntu@192.0.2.5 ./scripts/ssh_deploy.sh`.

## Offline planner comparison (no robot needed)

`scripts/rollout/` runs the pure-Python planner layers against a JSON scenario (obstacle map + energy
map + start/goal) and renders route comparisons — handy for sanity-checking planners without
hardware. See [`scripts/rollout/README.md`](scripts/rollout/README.md).

## Smoke tests

`scripts/smoke/00_preflight.sh` … `05_record_run.sh` validate the stack incrementally (ROS overlay,
sensor topics, compute stack, bridge drive, bag recording). Start with `00_preflight.sh`.
