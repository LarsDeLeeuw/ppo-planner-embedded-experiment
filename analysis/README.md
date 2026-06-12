# analysis

Standalone post-processing for the energy-harvesting experiment. Reads run
folders produced by the orchestrator and emits the report figures and summary
metrics. **No ROS2 install and no tracker import** — vanilla Python venv;
bags are decoded with `mcap-ros2-support` using schemas embedded in the `.mcap`.


## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows
# .venv/bin/python  on Linux/Mac
```

## Input: a run folder

```
runs/<run_id>/
  robot.mcap            # always
  desktop.mcap          # centralized mode only
  orchestrator.mcap     # PredictTcpTiming + local ExperimentEvent copy
  predict_log.jsonl     # extended AutoSessionLog (per-call energy maps, sequence)
  metadata.yaml
  maps/obstacle.npy     # static obstacle snapshot (energy maps are dynamic, in JSONL)
```

## Tools

| Command | Output |
|---|---|
| `python summarize_run.py runs/<run_id>` | `summary.json` + 4-panel `summary.png` in the run folder |
| `python analyze_session.py <root> [--out DIR] [--refresh]` | 5 required + 4 extra figures under `<root>/figures` |
| `python analyze_campaign.py <root> [--out DIR] [--refresh]` | same figures + `paired_comparisons.csv` (paired Wilcoxon, Mann-Whitney fallback) |
| `python analyze_day.py <root> --map <map_id>` | one-map day report: crash-aware `run_inventory.csv`, `runs_table.csv`, `metrics_by_group.csv`, `stats_tests.csv` (Mann-Whitney + Cliff's delta) + 13 figures incl. genuine-vs-goal-assisted success, energy decomposition, time/net-energy trade-off, mode overhead, inference time, turns-vs-energy |
| `python plot_trajectories.py <root> --map <map_id>` | mode×planner grid of executed trajectories over the obstacle map (genuine / assisted / failed colour-coded) |
| `python plot_energy_dynamics.py <root> --map <map_id>` | cumulative net-energy curves per run + harvest-power-vs-progress profiles (power series cached under the report dir) |
| `python compare_maps.py day3=<report>/runs_table.csv map2=... --out DIR` | cross-map comparison: success, headline-metric means, trade-off centroids, inference scaling across maps |

The day-report tools share `day_aggregate.py` (metadata merge, *genuine
success* = success without `goal_assisted`, derived efficiency columns,
crash-aware inventory) and default their output to
`<root>/figures/report-<map_id>/`, so the per-map reports feed straight into
`compare_maps.py`.

`<root>` is any directory; run folders are discovered recursively, so it works
on one session's `runs/` or a whole campaign tree. `summary.json` is regenerated
on demand (use `--refresh` to force).

## Generating test data (no hardware needed)

`make_synthetic_run.py` fabricates fully self-consistent run folders — the only
way to exercise the pipeline before the real experiment, and the validation
harness for tracker's `OrchestratorBag` writer (same `rosbags` path).

```bash
python make_synthetic_run.py run      OUTDIR [--mode ...] [--planner ...] [--baseline] [--dry-run]
python make_synthetic_run.py session  OUTDIR     # 6 runs, one (mode,planner) cell
python make_synthetic_run.py campaign OUTDIR     # 4 cells x 6 maps = 24 runs
```

## Importable modules (for Jupyter / ad-hoc)

- `run_io.load_run(dir) -> Run` — DataFrames per topic, JSONL sidecar, obstacle map, run window.
- `metrics.compute_summary(run) -> (row, warnings)` — all per-run derivations.
- `aggregate.load_summaries(root)`, `split_and_filter(df)`, `wilson_ci(k, n)`.
- `analyze_session.build_figures(df_kept, df_baselines, out_dir)`.

## Conventions / guarantees

- All latency math is single-host deltas joined by `sequence` — no cross-host
  clock subtraction (the laptop is not chrony-synced).
- The run window is bounded by bridge-host-stamped `run_start`/`run_end` on
  `/experiment/events`, not by `metadata.yaml` wall-clock.
- Energy integration is trapezoidal, excludes `overflow` samples, and drops
  genuine gaps (>50 ms or sequence skips).
- `runs/` is treated read-only except for the generated `summary.*` / `figures/`.
