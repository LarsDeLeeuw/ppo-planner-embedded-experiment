"""plot_energy_dynamics.py - Within-run energy time-series per (mode x planner).

    python plot_energy_dynamics.py <experiments_root> --map day3 [--out DIR] [--refresh]

Decodes /power/{solar,sbc,opencr} from robot.mcap for every kept run (window
bounded by run_start/run_end like metrics.py) and renders:

  cumulative_net_energy.png   per-cell panels: each run's cumulative
                              (solar - sbc - opencr) mWh over run time —
                              shows *when* a planner wins or loses energy,
                              not just the end total
  harvest_power_profile.png   per-run solar traces per cell (0.5 s bin max,
                              log axis) — sub-second harvest transients; no
                              group averaging (a mean fabricates group
                              behaviour out of single-run events)

Decoded series are downsampled to a 0.5 s grid and cached as JSON under
<out>/cache/ so re-renders are instant; nothing is written into runs/.
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
from day_aggregate import (  # noqa: E402
    MODE_ORDER, PLANNER_COLOR, PLANNER_ORDER, load_day,
)
from run_io import best_time_col, load_run  # noqa: E402

SENSORS = ("solar", "sbc", "opencr")
GRID_DT_S = 0.5


def _extract_series(run_dir: Path) -> dict | None:
    """Power per sensor binned onto a 0.5 s grid inside the run window.

    Bin-AGGREGATED, not point-sampled: solar is zero except for sub-second
    spikes, so sampling instantaneous values at grid points silently drops
    most harvest events. Per bin we keep the mean (energy-preserving, used
    for the cumulative figure) and for solar also the max (event detection).
    """
    run = load_run(run_dir)
    if run.window_ns is None:
        return None
    t0, t1 = run.window_ns
    dur = (t1 - t0) / 1e9
    if dur <= GRID_DT_S:
        return None
    n_bins = int(np.ceil(dur / GRID_DT_S))
    out = {"duration_s": round(dur, 3),
           "t": [round((b + 0.5) * GRID_DT_S, 2) for b in range(n_bins)]}
    for s in SENSORS:
        df = run.topic(f"/power/{s}")
        if df is None or df.empty:
            return None
        col = best_time_col(df)
        d = df[(df[col] >= t0) & (df[col] <= t1)].sort_values(col)
        if "overflow" in d.columns:
            d = d[~d["overflow"].astype(bool)]
        if len(d) < 2:
            return None
        t = (d[col].to_numpy() - t0) / 1e9
        p = d["power_mw"].to_numpy()
        bins = np.clip((t / GRID_DT_S).astype(int), 0, n_bins - 1)
        mean = np.zeros(n_bins)
        counts = np.bincount(bins, minlength=n_bins)
        np.add.at(mean, bins, p)
        present = counts > 0
        mean[present] /= counts[present]
        out[s] = mean.round(3).tolist()
        if s == "solar":
            mx = np.zeros(n_bins)
            np.maximum.at(mx, bins, p)
            out["solar_max"] = mx.round(3).tolist()
    return out


def _series_for(kept: pd.DataFrame, cache_dir: Path) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    series: dict[str, dict] = {}
    for _, r in kept.iterrows():
        # v2: bin-aggregated (mean + solar max); v1 point-sampled caches are stale
        cache = cache_dir / f"{r['run_id']}_power_v2.json"
        if cache.exists():
            series[r["run_id"]] = json.loads(cache.read_text())
            continue
        print("  decoding", r["run_id"], file=sys.stderr)
        s = _extract_series(Path(r["run_dir"]))
        if s is None:
            print(f"  skip {r['run_id']}: incomplete power data", file=sys.stderr)
            continue
        cache.write_text(json.dumps(s))
        series[r["run_id"]] = s
    return series


def _net_cumulative_mwh(s: dict) -> np.ndarray:
    net_mw = (np.asarray(s["solar"]) - np.asarray(s["sbc"]) - np.asarray(s["opencr"]))
    return np.cumsum(net_mw) * GRID_DT_S / 3600.0


def fig_cumulative(kept, series, out_path: Path):
    modes = [m for m in MODE_ORDER if m in set(kept["mode"])]
    planners = [p for p in PLANNER_ORDER if p in set(kept["planner"])]
    fig, axes = plt.subplots(len(modes), len(planners),
                             figsize=(4.8 * len(planners), 3.8 * len(modes)),
                             squeeze=False, sharey=True)
    for i, mode in enumerate(modes):
        for j, planner in enumerate(planners):
            ax = axes[i][j]
            sub = kept[(kept["mode"] == mode) & (kept["planner"] == planner)]
            c = PLANNER_COLOR.get(planner, "#999")
            for _, r in sub.iterrows():
                s = series.get(r["run_id"])
                if s is None:
                    continue
                cum = _net_cumulative_mwh(s)
                alpha = 0.8 if r["success"] else 0.35
                ls = "-" if r["success"] else ":"
                ax.plot(s["t"], cum, ls, color=c, lw=1.2, alpha=alpha)
            ax.axhline(0, color="k", lw=0.7, ls="--")
            ax.set_title(f"{mode} / {planner}", fontsize=10)
            ax.grid(alpha=0.25)
            if i == len(modes) - 1:
                ax.set_xlabel("run time (s)")
            if j == 0:
                ax.set_ylabel("cumulative net energy (mWh)")
    fig.suptitle("Cumulative net energy (solar − SBC − OpenCR); dotted = failed run",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


EVENT_MW = 0.5       # above this we call it a harvest event, not ambient trickle
LOG_FLOOR_MW = 0.01  # log-axis clip so zero-power samples form a visible floor


def fig_harvest_profile(kept, series, out_path: Path):
    """Per-run solar traces (0.5 s bin MAX), one panel per (mode, planner).

    Deliberately NO cross-run mean: solar is zero except sub-second spikes,
    so a group-mean curve fabricates "group behaviour" out of single-run
    events. Bin max is used because point-sampling drops spikes that fall
    between grid instants. Panel titles count runs with any event instead.
    """
    modes = [m for m in MODE_ORDER if m in set(kept["mode"])]
    planners = [p for p in PLANNER_ORDER if p in set(kept["planner"])]
    fig, axes = plt.subplots(len(modes), len(planners),
                             figsize=(4.8 * len(planners), 3.8 * len(modes)),
                             squeeze=False, sharey=True)
    for i, mode in enumerate(modes):
        for j, planner in enumerate(planners):
            ax = axes[i][j]
            sub = kept[(kept["mode"] == mode) & (kept["planner"] == planner)]
            c = PLANNER_COLOR.get(planner, "#999")
            n_event, n = 0, 0
            for _, r in sub.iterrows():
                s = series.get(r["run_id"])
                if s is None:
                    continue
                n += 1
                solar = np.asarray(s.get("solar_max", s["solar"]))
                if (solar > EVENT_MW).any():
                    n_event += 1
                ax.plot(s["t"], np.clip(solar, LOG_FLOOR_MW, None), "-",
                        color=c, lw=0.9, alpha=0.55)
            ax.set_yscale("log")
            ax.axhline(EVENT_MW, color="#888", lw=0.7, ls=":")
            ax.set_title(f"{mode} / {planner} — {n_event}/{n} runs with "
                         f">{EVENT_MW} mW event", fontsize=9)
            ax.grid(alpha=0.25, which="both")
            if i == len(modes) - 1:
                ax.set_xlabel("run time (s)")
            if j == 0:
                ax.set_ylabel(f"solar power (mW, log; floor {LOG_FLOOR_MW})")
    fig.suptitle("Solar power per run (0.5 s bin max) — sub-second harvest "
                 "transients; no group averaging", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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

    series = _series_for(kept, out / "cache")
    print(f"power series for {len(series)}/{len(kept)} kept runs")
    fig_cumulative(kept, series, out / "cumulative_net_energy.png")
    fig_harvest_profile(kept, series, out / "harvest_power_profile.png")
    print("wrote", out / "cumulative_net_energy.png")
    print("wrote", out / "harvest_power_profile.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
