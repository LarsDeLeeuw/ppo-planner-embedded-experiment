# Documentation

| Doc | What it covers |
|---|---|
| [architecture.md](architecture.md) | How tracker + robot + analysis fit together; the 2×2 experiment design; the closed control loop. |
| [running.md](running.md) | End-to-end run guide: setup → build → run → campaign → analyse. |
| [hardware.md](hardware.md) | The physical testbed: robot, INA219 power sensing, RAPL, overhead camera + markers. |
| [calibration.md](calibration.md) | Per-surface drive calibration procedure for `grid_nav`. |
| [bridge-protocol.md](bridge-protocol.md) | Full TCP/JSON wire contract for talking to the robot. |
| [coordinate-conventions.md](coordinate-conventions.md) | Grid / heading / action conventions (get these wrong and maps rotate 90°). |

`media/` holds images/gifs used by the READMEs. `workflows/` is internal project-tracking and is not
part of the user-facing documentation.
