# tracker

Overhead-camera **ArUco localization** and the **experiment orchestrator** for the TurtleBot3
energy-harvesting path-planning testbed.

An overhead camera sees four ArUco corner markers (IDs 1–4) on the floor plus a marker (ID 0) on
the robot. From the four corners the tracker reconstructs the floor plane, projects the robot's pose
onto it to get a grid cell + heading, and streams pose corrections and navigation goals over TCP to
the ROS2 `ros2_bridge` running on the robot (see [`robot/`](../robot/)). Pressing `g` captures a
top-down view and turns per-cell luminance into an **energy map** for the planners.

The **orchestrator** (`orchestrator/`) layers experiment automation on top: it walks a campaign of
runs, starts/stops ROS bag recording over SSH, drives warmup + timed runs, pulls the bags back, and
writes per-run metadata — then optionally kicks off the [`analysis/`](../analysis/) summary.

> This was originally a drone-tracking proof of concept. The legacy TimescaleDB telemetry sink and
> the React/Three.js visualizer have been removed; only the localization + orchestration code
> remains.

## Layout

| Path | What it is |
|---|---|
| `main.py` | Live capture loop: detect → calibrate grid → localize → AR overlay → (optional) bridge + auto-drive. |
| `config/` | YAML config: `defaults.yml` (canonical), `schema.py` (validated types), `loader.py` (merge logic). |
| `orchestrator/` | Campaign runner, SSH bag recording, per-run metadata. Opt-in. |
| `calibrate.py` | Interactive checkerboard camera calibration → `.npz`. |
| `generate_marker.py`, `generate_checkerboard.py` | Generate the printable ArUco / checkerboard images. |
| `markers/` | Ready-to-print ArUco markers (0–4) and the calibration checkerboard. |

## Install

```bash
cd tracker
python -m venv .venv
.venv/Scripts/activate     # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The tracker reads `config/defaults.yml`, then an experiment file (`--config ...`), then
`tracker/local.yml` if present, then `--set key=value` overrides — later layers win.

```bash
# point at a specific camera and bridge host without editing defaults
python main.py --set camera.index=1 --set bridge.host=192.0.2.10

# validate the resolved config and exit
python main.py --validate-config

# print the fully merged config
python main.py --print-config
```

Machine-specific values (camera index, real bridge host/IP, SSH targets) belong in
**`tracker/local.yml`**, which is git-ignored. `config/defaults.yml` ships only safe placeholders
(e.g. `bridge.host: TURTLEBOT3-VM`).

### Camera calibration (one-time)

```bash
python calibrate.py        # capture checkerboard views → writes a .npz
python main.py --set camera.calibration_file=calibration.npz
```

### Key bindings (in the live window)

| Key | Action |
|---|---|
| `Esc` | quit |
| `r` | re-calibrate the grid from the 4 corner markers |
| `WASD` / `QEZC` | move the target cell (orthogonal / diagonal) |
| `g` | capture a top-down frame and build the energy map |
| `m` | toggle mark mode (click a cell to toggle it) |
| `[` `]` `F` `X` | (mark mode) rotate / flip / clear the scene layer |
| `K` / `L` | save / load a scene (obstacle + goal layout) |
| `p` | toggle the autonomous planner loop (needs bridge + a goal cell) |
| `v` / `V` | toggle the minimap / clear its trail |
| `+` `-` `0` | scale the UI up / down / reset |

## Orchestrator (experiment automation)

Off by default. Enable with `orchestrator.enabled: true` (or pass `--campaign <plan.yaml>`), then
provide an SSH-reachable robot and the bag-record script on the robot side. See
`orchestrator/example_campaign_plan.yaml` for the campaign format and
[`config/defaults.yml`](config/defaults.yml) (the `orchestrator:` block) for the knobs.

```bash
python main.py --campaign orchestrator/one_trial_plan.yaml --auto-advance
python main.py --campaign orchestrator/example_campaign_plan.yaml --ssh-mock   # no robot needed
```

Runs land under `experiments/runs/<run_id>/` (bags + `predict_log.jsonl` + `metadata.yaml`), which
the [`analysis/`](../analysis/) pipeline turns into figures and summaries.

## Notes / known follow-ups

- The orchestrator's default robot-side bag-record path (`config/defaults.yml` → `orchestrator.ssh.*.bag_script`)
  points at where the ROS workspace is deployed on the robot; override it in `local.yml` to match your deploy.
- `orchestrator_bag.py` writes `orchestrator.mcap` using the message schemas in
  [`robot/src/tb3_interfaces`](../robot/src/tb3_interfaces); it locates them by walking up the repo tree.
