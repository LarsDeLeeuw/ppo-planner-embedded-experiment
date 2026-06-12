# Planner rollout / comparison tool

Off-robot (workstation) tool that simulates how each of the experiment's three
planners would drive a single robot across a map, then renders the routes and
reports the comparison metrics from the energy-harvesting paper replication. Use
it to **sanity-check a map before running the physical experiment**: does the
energy-aware planner detour into the bright cells? does PPO reach the goal? how
much longer is its route than the shortest path?

It imports the same pure-Python planner layers the ROS nodes use
([`astar_planner.astar_search`](../../src/astar_planner/astar_planner/astar_search.py),
[`ppo_planner.ppo_inference`](../../src/ppo_planner/ppo_planner/ppo_inference.py)),
so a route here is exactly what the deployed service produces for the same
inputs. **No ROS / colcon build is required** — the tool adds the `src/` package
dirs to `sys.path` itself.

## Authoring maps

[`map-designer.html`](map-designer.html) is a companion browser tool for drawing
maps — obstacles, start/goal, and light sources — with a live preview of the
three planners' optimal paths and the energy field. Its **"Export as rollout
scenario"** button writes a JSON bundle in exactly the [format below](SCENARIO_FORMAT.md),
already in the repo `[x][y]` frame. Open it in any browser (no server needed),
design a map, download the scenario, and pass it to `--scenario`.

## Planners

| `--planners` key | What it is |
|---|---|
| `astar_shortest` | classic 8-connected A*, minimises geometric path length (the optimal-route baseline / `L*`) |
| `astar_energy` | energy-aware A*; trades a little distance for harvested energy (`--energy-weight`) |
| `ppo` | the trained PPO policy (ONNX backend, numpy + onnxruntime); stochastic by default |

## Install (one-time, workstation only)

```bash
python3 -m venv ~/.venvs/rollout
~/.venvs/rollout/bin/pip install "numpy" "onnxruntime>=1.17,<2.0" "matplotlib"
```

(`onnxruntime` is only needed when `ppo` is in `--planners`.)

## Run

```bash
PY=~/.venvs/rollout/bin/python

# all three planners on the bundled example:
$PY scripts/rollout/rollout.py --scenario scripts/rollout/examples/example_scenario.json

# just the two A* variants, heavier energy weight:
$PY scripts/rollout/rollout.py --scenario my_map.json \
    --planners astar_shortest,astar_energy --energy-weight 2.0

# PPO only, more stochastic trials, fixed seed:
$PY scripts/rollout/rollout.py --scenario my_map.json --planners ppo --trials 16 --seed 0

# PPO greedy (argmax) to inspect the single trajectory + failure diagnostics:
$PY scripts/rollout/rollout.py --scenario my_map.json --planners ppo --ppo-deterministic
```

| Flag | Effect |
|---|---|
| `--planners` | comma list from `astar_shortest,astar_energy,ppo` (default: all three) |
| `--energy-weight` | energy weight for `astar_energy` (default `1.0`, matches the experiment default) |
| `--trials` | stochastic-PPO runs to overlay (default `8`; ignored for A* / argmax PPO) |
| `--seed` | RNG seed for stochastic PPO (default `0`) |
| `--ppo-deterministic` | run PPO with argmax instead of sampling |
| `--ppo-model` | override the PPO `.onnx` (default: `src/ppo_planner/models/AIPPOm10EH_continued.onnx`) |
| `--max-steps` | override the rollout step cap |
| `--out-dir` / `--name` | output directory / filename-stem override |

## Outputs (next to the scenario, or `--out-dir`)

| File | What it is |
|---|---|
| `<name>_comparison.png` | all selected planners side by side on the shared map |
| `<name>_comparison.json` | every metric for every planner (machine-readable) |
| `<name>_<planner>_route.png` | per-planner route (PPO overlays all stochastic trials; A*/argmax PPO show failure diagnostics) |
| `<name>_<planner>_path.csv` | per-step log of the representative route |

A metrics table is also printed to stdout.

## Metrics (what the comparison reports)

These mirror the metrics the energy-harvesting paper uses (see the research
context + citation in [docs/architecture.md](../../../docs/architecture.md)):

- **status / success** — reached goal? (for PPO: `n_reached / n_trials` over the stochastic trials)
- **steps**, **path_cells** — route length in moves / cells
- **path_length_geom** — Euclidean route length (`1` per cardinal step, `√2` per diagonal)
- **path_length_ratio** (`L/L*`) — route length ÷ shortest-path length; `≥ 1`, closer to `1` is more direct (the paper's path-length efficiency)
- **energy_harvested** (`EH`) — Σ `energy_map` over each occupied cell along the route (start + every landed cell; revisits counted each time)
- **energy_per_length** (`EH/len`) — harvested energy per unit route length (the paper's energy-harvesting efficiency)

See [SCENARIO_FORMAT.md](SCENARIO_FORMAT.md) for the input file spec and the
`[x][y]` coordinate convention.
