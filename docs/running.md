# Running an experiment end to end

This walks through a full run, from a clean checkout to analysed results. It assumes the physical
testbed from [hardware.md](hardware.md). Commands use safe placeholder hosts/paths — put your real
values in the local config files noted below (never commit those).

> Architecture and the message contract live in [architecture.md](architecture.md),
> [bridge-protocol.md](bridge-protocol.md), and [coordinate-conventions.md](coordinate-conventions.md).

## 0. One-time setup

**Machines.** A Windows laptop (tracker + orchestrator + overhead camera), the TurtleBot3 (Pi 4B),
and — for centralized mode — a desktop running ROS 2 Humble.

**SSH.** Give the laptop passphrase-free SSH to the robot (and desktop). The scripts default to
`ubuntu@turtlebot3.local`; override per invocation or via `~/.ssh/config` aliases. The orchestrator
refers to hosts by alias (`robot`, `vm`).

**Markers & camera.** Print the ArUco markers and checkerboard from [`tracker/markers/`](../tracker/markers/);
place IDs 1–4 at the grid corners and ID 0 on the robot. Set your camera (device index or IP-camera
URL), bridge host, and SSH targets in **`tracker/local.yml`** (git-ignored):

```yaml
# tracker/local.yml  — machine-local, never committed
camera:
  index: "http://<camera-ip>:8081/video"   # or an int device index
bridge:
  host: <robot-or-desktop-ip>               # whichever runs the bridge this session
orchestrator:
  ssh:
    pi:      { target: robot }
    desktop: { target: vm }
```

**Camera intrinsics (optional but recommended).**

```bash
cd tracker && pip install -r requirements.txt
python calibrate.py                                   # capture checkerboard views -> calibration.npz
python main.py --set camera.calibration_file=calibration.npz
```

## 1. Build the robot workspace

On whichever host will run the compute stack (the robot for decentralized, the desktop for
centralized):

```bash
cd robot
./scripts/build_all.sh                       # symlinks packages into the colcon ws and builds
source ~/turtlebot3_ws/install/setup.bash
```

Deploy from the laptop/desktop to the robot with `./scripts/ssh_deploy.sh`
(`ROBOT_HOST` / `REMOTE_DIR` are env-overridable — see [robot/README.md](../robot/README.md)).

## 2. Start the always-on sensor layer (robot)

```bash
ros2 launch tb3_bringup robot_sensors.launch.py     # 3× INA219 power + thermal + diagnostics
```

Sanity-check sensor topics with `scripts/smoke/02_check_sensor_topics.sh`.

## 3. Launch the compute stack (planner + nav + bridge)

Pick a **mode** (where this runs) and a **planner**.

```bash
# Decentralized — on the robot:
ros2 launch tb3_bringup bringup.launch.py planner:=ppo            # or astar / astar_shortest

# Centralized — on the desktop (robot only runs the sensor layer from step 2):
ros2 launch tb3_bringup bringup.launch.py planner:=ppo use_diagnostics:=true use_rapl:=true
```

PPO takes ~60 s to load its model on first launch; A\* starts instantly. Per-experiment overrides
(grid cell size, calibration, A\* weighting) live in
[`robot/src/tb3_bringup/config/experiment.yaml`](../robot/src/tb3_bringup/config/experiment.yaml).

Quick benchtop check without the tracker (drives the bridge directly):

```bash
python robot/scripts/smoke/04_drive_bridge.py --robot-ip <bridge-host> --no-goal
```

## 4. Drive it from the tracker (single manual run)

On the laptop, with the overhead camera in view of all four corner markers:

```bash
cd tracker
python main.py
```

Lock the grid (`r`), mark obstacles and a goal (`m` + click, `g` for the energy map), then toggle the
autonomous planner loop (`p`). Full key bindings are in [tracker/README.md](../tracker/README.md).

## 5. Run a campaign (orchestrator)

The orchestrator automates many runs: warmup, timed run, bag start/stop over SSH, pull-back, and
metadata — optionally kicking off the analysis summary per run.

```bash
cd tracker
python main.py --campaign orchestrator/one_trial_plan.yaml --auto-advance      # one trial
python main.py --campaign orchestrator/example_campaign_plan.yaml              # full campaign
python main.py --campaign orchestrator/example_campaign_plan.yaml --ssh-mock   # dry run, no robot
```

Runs land under `experiments/runs/<run_id>/` — each with `robot.mcap` (+ `desktop.mcap` in
centralized mode), `orchestrator.mcap`, `predict_log.jsonl`, `metadata.yaml`, and
`maps/obstacle.npy`. Campaign formats: see `tracker/orchestrator/example_campaign_plan.yaml`.

## 6. Analyse

No ROS install needed — bags carry their schemas.

```bash
cd analysis
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Windows
python summarize_run.py <run_dir>                 # summary.json + summary.png per run
python analyze_session.py <session_root>          # cross-run figures
python analyze_day.py <root> --map <map_id>       # per-map day report (tables + figures)
```

Before any hardware is available you can exercise the whole pipeline on fabricated data:

```bash
python make_synthetic_run.py campaign _synthetic   # 4 cells × 6 maps of self-consistent fake runs
```

See [analysis/README.md](../analysis/README.md) for every tool and the importable modules.

## Where the real values live (and stay)

| Value | Put it in | Committed? |
|---|---|---|
| Camera index/URL, bridge host, SSH aliases | `tracker/local.yml` | no (git-ignored) |
| Robot SSH host / deploy dir | `ROBOT_HOST` / `REMOTE_DIR` env vars | no |
| Per-experiment params (cell size, calibration, A\* weight) | `robot/src/tb3_bringup/config/experiment.yaml` | yes (defaults only) |
| Per-surface drive calibration | from [calibration.md](calibration.md) → `experiment.yaml` | yes |
