# Hardware & testbed

The physical setup this repo drives and instruments.

## Robot

| Item | Spec |
|---|---|
| Platform | TurtleBot3 **Burger** (`TURTLEBOT3_MODEL=burger`), ROS 2 Humble |
| Onboard computer | Raspberry Pi 4B, I2C on `/dev/i2c-1` |
| Base MCU | OpenCR (motors, IMU, servo); motor PWM ≈ 16 kHz |
| Max kinematics | ≈ 0.22 m/s linear, ≈ 2.84 rad/s angular (limits set conservatively in nav config for accuracy) |

The IMU is the heading authority for navigation (wheel odometry slips during in-place pivots,
especially on carpet); see [calibration.md](calibration.md).

## Energy & power sensing — 3× INA219

Three INA219 current/power sensors share the Pi's I2C bus, each publishing
`tb3_interfaces/PowerSample` at 100 Hz. Together they capture both **harvested** (solar) and
**consumed** (compute + motor) energy — the physical analog of the paper's simulated energy field.

| Rail (topic) | I2C addr | Placement | Sign convention |
|---|---|---|---|
| `solar` (`/power/solar`) | 0x45 | solar panel ↔ battery | + = charging (harvest) |
| `sbc` (`/power/sbc`) | 0x44 | battery ↔ Pi 4B | + = power into the Pi |
| `opencr` (`/power/opencr`) | 0x41 | battery ↔ OpenCR | + = power into motors/IMU/servo |

Configured in [`robot/src/tb3_power_sensor/config/sensor_map.yaml`](../robot/src/tb3_power_sensor/config/sensor_map.yaml):
all shunts 0.1 Ω, PGA ÷8 (±320 mV ≈ ±3.2 A), 32 V bus range, **8-sample ADC averaging**, I2C at
400 kHz. The 8-sample averaging acts as an anti-aliasing low-pass against the ~16 kHz motor PWM;
overflow samples are flagged per reading and excluded from energy integration. Each sensor has a
small per-sensor gain calibration applied at the node (consumers must not re-apply it).

## Thermal

The Pi's `thermal_zone0` temperature and `vcgencmd get_throttled` flags are published at 1 Hz
(`tb3_interfaces/ThermalStat`) so thermal throttling during a run is visible in the data.

## Compute-energy — Intel RAPL (centralized mode only)

When the compute stack runs on the desktop (centralized mode), `tb3_rapl_sampler` reads Intel RAPL
package + DRAM energy counters from `/sys/class/powercap/intel-rapl*` at 1 Hz
(`tb3_interfaces/RaplEnergy`). This measures the CPU energy of the planning stack — invisible to the
robot's INA219 rails — enabling the centralized-vs-decentralized compute-energy comparison. Validity
depends on the hypervisor exposing RAPL (`passthrough_ok`).

## Overhead localization

| Item | Notes |
|---|---|
| Camera | An overhead camera viewing the whole grid. A USB device index or an IP-camera URL both work; set it in `tracker/local.yml`. |
| Markers | 4 ArUco corner markers (IDs 1–4) define the grid plane + 1 marker (ID 0) on the robot. Print them from [`tracker/markers/`](../tracker/markers/). |
| Calibration | Print the checkerboard in `tracker/markers/`, run `python calibrate.py` for camera intrinsics. Physical marker sizes are set in the tracker config (corner 15.2 cm, robot 8.0 cm by default). |

The tracker reconstructs the floor plane from the 4 corners and projects the robot marker's pose onto
it to get a grid cell + heading, which it streams to the robot as pose corrections.

## Grid surface calibration

Drive accuracy depends on the floor (carpet slips; tile/wood is closer to the simulator's no-slip
assumption). Run the per-surface calibration once and paste the result into the bringup config — see
[calibration.md](calibration.md).

## Clock sync

The robot and desktop run `chrony` (warn at >5 ms offset, reject at >20 ms). The laptop sits outside
the chrony hierarchy; cross-host data is joined by a per-call `sequence` counter rather than by
subtracting clocks, so no cross-host clock arithmetic is ever required.
