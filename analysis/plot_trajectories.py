"""plot_trajectories.py - Executed trajectories per (mode x planner) cell.

    python plot_trajectories.py <experiments_root> --map day3 [--out DIR] [--refresh]

One panel per (mode, planner) on the obstacle map, every kept run overlaid:
  green  = genuine success      orange = goal-assisted success
  red    = failure / timeout (kept but goal not reached)
Start cell (green square) and goal cell (magenta star) are marked; the octile
straight-line is drawn as reference. This shows *where* astar_energy detours
relative to astar_shortest and how repeatable each planner's route is.

Cheap: reads only pose_log.jsonl / metadata.yaml / maps/obstacle.npy
(summary.json is used for filtering and is generated only if missing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day_aggregate import MODE_ORDER, PLANNER_ORDER, load_day  # noqa: E402

COLOR_GENUINE = "#2ca02c"
COLOR_ASSISTED = "#ff7f0e"
COLOR_FAILED = "#d62728"


def _load_pose(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "pose_log.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not recs:
        return None
    return pd.DataFrame(recs).sort_values("t_mono_ns")


def _cells(s: str | None) -> tuple[int, int] | None:
    try:
        x, y = (int(v) for v in str(s).split("_"))
        return x, y
    except (TypeError, ValueError):
        return None


def build_figure(kept: pd.DataFrame, out_path: Path) -> None:
    modes = [m for m in MODE_ORDER if m in set(kept["mode"])]
    planners = [p for p in PLANNER_ORDER if p in set(kept["planner"])]
    fig, axes = plt.subplots(len(modes), len(planners),
                             figsize=(4.6 * len(planners), 4.6 * len(modes)),
                             squeeze=False)

    for i, mode in enumerate(modes):
        for j, planner in enumerate(planners):
            ax = axes[i][j]
            sub = kept[(kept["mode"] == mode) & (kept["planner"] == planner)]
            obs = None
            n_drawn = 0
            for _, r in sub.iterrows():
                run_dir = Path(r["run_dir"])
                if obs is None:
                    obs_path = run_dir / "maps" / "obstacle.npy"
                    if obs_path.exists():
                        obs = np.load(obs_path)
                        ax.imshow(obs, origin="lower", cmap="Greys", alpha=0.4,
                                  extent=[0, obs.shape[1], 0, obs.shape[0]], zorder=1)
                pl = _load_pose(run_dir)
                if pl is None or pl.empty:
                    continue
                if r["genuine_success"]:
                    color = COLOR_GENUINE
                elif r["assisted_success"]:
                    color = COLOR_ASSISTED
                else:
                    color = COLOR_FAILED
                ax.plot(pl["x"], pl["y"], "-", color=color, lw=1.1, alpha=0.55, zorder=2)
                n_drawn += 1
            sc = _cells(sub["start_cell"].iloc[0]) if len(sub) else None
            gc = _cells(sub["goal_cell"].iloc[0]) if len(sub) else None
            if sc and gc:
                ax.plot([sc[0] + 0.5, gc[0] + 0.5], [sc[1] + 0.5, gc[1] + 0.5],
                        ":", color="#555", lw=0.9, zorder=2)
            if sc:
                ax.plot(sc[0] + 0.5, sc[1] + 0.5, "s", color="green", ms=10, zorder=4)
            if gc:
                ax.plot(gc[0] + 0.5, gc[1] + 0.5, "*", color="magenta", ms=15, zorder=4)
            ax.set_title(f"{mode} / {planner} (n={n_drawn})", fontsize=10)
            ax.set_aspect("equal")
            if obs is not None:
                ax.set_xlim(0, obs.shape[1])
                ax.set_ylim(0, obs.shape[0])
            ax.tick_params(labelsize=7)

    handles = [plt.Line2D([], [], color=c, lw=2, label=l) for c, l in
               [(COLOR_GENUINE, "genuine success"),
                (COLOR_ASSISTED, "goal-assisted success"),
                (COLOR_FAILED, "failed")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False)
    fig.suptitle("Executed trajectories (pose_log, bridge frame)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--map", dest="map_id", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="report dir (default <root>/figures/report-<map>)")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    _, kept, _ = load_day(args.root, args.map_id, refresh=args.refresh)
    if kept.empty:
        print("no kept runs", file=sys.stderr)
        return 1
    out = args.out or (args.root / "figures" / f"report-{args.map_id or 'all'}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trajectories.png"
    build_figure(kept, path)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
