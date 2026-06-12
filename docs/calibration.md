# Calibrating grid_nav for a new surface

The TurtleBot's drive accuracy depends on the surface. Carpet causes the
wheels to slip; smooth tile or wood is closer to the simulator's
zero-slip assumption. Run this procedure once per surface and copy the
result into [src/tb3_bringup/config/experiment.yaml](../robot/src/tb3_bringup/config/experiment.yaml).

## What you need

- The robot, fully charged.
- A roll of masking tape and a tape measure (~5 m, mm markings).
- A printed 90-degree template, **or** a square tile / two perpendicular floor edges you can align against.
- An open patch of the experiment surface, at least 1.5 x 1.5 m.

## What it does

The script runs three short tests:

| # | Maneuver | Measurement you take | What it tells us |
|---|----------|----------------------|------------------|
| T1 | Drive 1 m straight, no rotation. Repeat 3x. | Forward distance vs the start mark | Linear slip ratio |
| T2 | Rotate ~90 deg in place (then 5 cm drive). Repeat 2x. | Actual rotation against the 90 deg template | Whether IMU rotation matches physical rotation |
| T3 | Rotate 90 deg, then drive 1 m. Once. | Final position (dx, dy) and final heading | Whether rotation introduces position drift |

The script also captures raw IMU yaw, raw odom (x, y, yaw) and the nav
node's world-frame estimate at every phase transition during each move.
That data is printed at the end so any rotation-coupled drift can be
diagnosed.

## How to run

In one terminal (on the robot or wherever you normally launch):

```bash
ros2 launch tb3_bringup bringup.launch.py use_bridge:=false planner:=none
```

In another terminal:

```bash
ros2 run tb3_nav calibrate
```

The script will:

1. Wait for the action server, the SetGridPose service, and at least one IMU and odom message.
2. Read `cell_size` and `use_imu_heading` from the running nav node.
3. Walk you through T1, T2, T3, prompting for measurements between moves.
4. Print a verdict and a YAML snippet.

Hit `Ctrl+C` at any prompt to abort cleanly (the script cancels any active goal).

## Reading the report

Three sections to check, in order:

### 1. Linear (T1)

```
T1.1: commanded 1.000 m, measured 0.870 m, ratio 0.870
T1.2: commanded 1.000 m, measured 0.880 m, ratio 0.880
T1.3: commanded 1.000 m, measured 0.875 m, ratio 0.875
T1 mean ratio = 0.875  (spread 0.010)
```

A spread > 0.05 means measurement noise is dominating; re-run with more
care on the start/end marks. Otherwise the mean ratio is your
`linear_calibration`.

### 2. Rotation (T2)

```
T2.1: physical  +88.5 deg | IMU  +90.0 deg | odom  +95.0 deg
T2.2: physical  +89.0 deg | IMU  +90.0 deg | odom  +96.0 deg
```

Compare the three columns:

- **physical and IMU agree (within ~5 deg)**: the IMU is faithful. With
  `use_imu_heading: true` (the default), `angular_calibration` should
  stay at 1.0.
- **IMU and odom disagree**: expected on carpet — wheel encoders slip
  during in-place pivot, but the IMU does not. Just confirms IMU is the
  right sensor for heading.
- **physical and IMU disagree**: the IMU yaw is biased. Either the
  sensor needs re-calibration, or the user's reference template was
  wrong. Re-check the template before doing anything else.

### 3. Cross-check (T3)

The verdict line will say one of:

- *"motion direction matches commanded heading"*: rotation does not
  drag the robot off course; calibration from T1 should be enough.
- *"motion direction off by N deg"*: the robot rotates correctly per
  IMU but ends up driving in a different direction. This means rotation
  introduces position drift (or there's an odom/world frame mismatch).
  In that case, do **not** trust the linear_calibration value yet -
  raise `settling_time` to ~0.6 s and re-run the procedure.

### YAML snippet

The final block is ready to paste:

```yaml
grid_nav_node:
  ros__parameters:
    linear_calibration: 0.875
    # angular_calibration: 1.0  # IMU mode in use; not applied
```

Paste this into [src/tb3_bringup/config/experiment.yaml](../robot/src/tb3_bringup/config/experiment.yaml),
under the `grid_nav_node` block, replacing any prior values.

## When to re-run

Any time the surface changes — tile to carpet, dry to wet, fresh
batteries with different wheel pressure, etc. The whole procedure takes
~5 minutes once you have the floor markings.

## Hop-scale calibration — for the experiment's actual operating point

T1 above measures `linear_calibration` over a 1.0 m sustained drive, where
~93% of the trajectory is at saturated `max_linear_speed=0.15 m/s`. The
experiment never drives 1 m at a time — every commanded move is one grid
cell (0.20 m cardinal or 0.283 m diagonal) starting and ending at
standstill. The slip dynamics in that operating regime are different:
steady-state wheel slip is small (no sustained cruise), and brake-coast
at the `min_linear_speed` floor dominates the per-hop error.

For experiment use, calibrate against 0.20 m hops instead:

```bash
# In one terminal:
ros2 launch tb3_bringup bringup.launch.py use_bridge:=false planner:=none
# In another (only if the running max_move_distance < hop_distance):
ros2 param set /grid_nav_node max_move_distance 1.5
# Then:
ros2 run tb3_nav calibrate_hop --per-hop
```

Defaults: 5 hops × 0.20 m, with `--per-hop` prompting between hops so
you get a random-error sigma. The tool accounts for whatever
`linear_calibration` is currently active and prints an absolute
recommendation. Same paste-into-experiment.yaml workflow as T1.

Field testing (lab surface, 2026-06-06): T1 produced 0.79, hop-scale
produced 0.9283 with σ=4.5 mm — different operating regimes, both
valid measurements, hop-scale is the one to use for experiment runs.
