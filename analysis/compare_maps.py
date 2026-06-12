"""compare_maps.py - Compare experiment days (maps) from their runs_table.csv.

    python compare_maps.py day3=<report_dir>/runs_table.csv map2=<...>/runs_table.csv \
        --out <dir>

Each positional argument is label=path-to-runs_table.csv (the per-run table
analyze_day.py writes). Works with one map (degenerates to per-group bars) but
is built for two+: are the planner/mode effects consistent across maps, or
map-specific?

Outputs:
  map_comparison.csv          group x map means for the headline metrics
  success_by_map.png          genuine success rate per group, clustered by map
  metrics_by_map.png          mean +/- 95% CI per group, one panel per metric,
                              maps side by side
  tradeoff_by_map.png         time vs net energy centroids, one marker fill
                              style per map — does the trade-off move?
  inference_by_map.png        inference mean/p99 scaling across maps per
                              (mode, planner) — A* search grows with the map,
                              a fixed forward pass (PPO) does not
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import wilson_ci  # noqa: E402
from day_aggregate import (  # noqa: E402
    MODE_LS, MODE_MARKER, PLANNER_COLOR, ordered_group_keys,
)

# Outcome metrics use successful runs only: failed runs' partial paths/times/
# energies contaminate group means (a failed PPO run "walks" a shorter path than
# a successful shortest-A* run, which is meaningless as a comparison).
METRICS = [
    ("time_to_completion_s", "time to goal (s, successes)"),
    ("executed_path_length_m", "path length (m, successes)"),
    ("energy_harvested_mwh", "harvested (mWh, successes)"),
    ("energy_net_mwh", "net energy (mWh, successes)"),
]
MAP_ALPHA = [0.9, 0.55, 0.3]  # bar shading per map slot


def _load(arg: str) -> tuple[str, pd.DataFrame]:
    if "=" not in arg:
        sys.exit(f"expected label=path, got: {arg}")
    label, path = arg.split("=", 1)
    df = pd.read_csv(path)
    df["map_label"] = label
    return label, df


def _ci95(vals: np.ndarray) -> float:
    return 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0


def fig_success(dfs: dict[str, pd.DataFrame], groups: list[str], out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / len(dfs)
    for k, (label, df) in enumerate(dfs.items()):
        xs, ys, errs = [], [], []
        for i, g in enumerate(groups):
            sub = df[df["group"] == g]
            if not len(sub):
                continue
            n = len(sub)
            kk = int(sub["genuine_success"].sum())
            phat, lo, hi = wilson_ci(kk, n)
            x = i + (k - (len(dfs) - 1) / 2) * width
            xs.append(x); ys.append(phat)
            errs.append([max(0, phat - lo), max(0, hi - phat)])
            p = g.split("/", 1)[1]
            ax.bar(x, phat, width=width * 0.9, color=PLANNER_COLOR.get(p, "#999"),
                   alpha=MAP_ALPHA[k % len(MAP_ALPHA)])
            ax.annotate(f"{kk}/{n}", (x, phat), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7)
        if xs:
            ax.errorbar(xs, ys, yerr=np.array(errs).T, fmt="none", ecolor="k",
                        capsize=3, lw=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8, rotation=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("genuine success rate")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#888", alpha=MAP_ALPHA[k % len(MAP_ALPHA)],
                             label=label) for k, label in enumerate(dfs)]
    ax.legend(handles=handles, fontsize=8)
    ax.set_title("Genuine success rate per map (Wilson 95% CI)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "success_by_map.png", dpi=130)
    plt.close(fig)


def fig_metrics(dfs: dict[str, pd.DataFrame], groups: list[str], out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    width = 0.8 / len(dfs)
    for ax, (col, label) in zip(axes.flat, METRICS):
        for k, (mlabel, df) in enumerate(dfs.items()):
            ok = df[df["success"]]
            for i, g in enumerate(groups):
                vals = pd.to_numeric(ok[ok["group"] == g].get(col),
                                     errors="coerce").dropna().to_numpy()
                if not len(vals):
                    continue
                x = i + (k - (len(dfs) - 1) / 2) * width
                p = g.split("/", 1)[1]
                ax.bar(x, vals.mean(), width=width * 0.9,
                       color=PLANNER_COLOR.get(p, "#999"),
                       alpha=MAP_ALPHA[k % len(MAP_ALPHA)])
                ax.errorbar(x, vals.mean(), yerr=_ci95(vals), fmt="none",
                            ecolor="k", capsize=3, lw=1)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=7, rotation=12)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if "net" in col:
            ax.axhline(0, color="k", lw=0.8, ls="--")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#888", alpha=MAP_ALPHA[k % len(MAP_ALPHA)],
                             label=label) for k, label in enumerate(dfs)]
    fig.legend(handles=handles, loc="lower center", ncol=len(dfs), fontsize=9,
               frameon=False)
    fig.suptitle("Group means ± 95% CI per map (successful runs)", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out / "metrics_by_map.png", dpi=130)
    plt.close(fig)


def fig_inference(dfs: dict[str, pd.DataFrame], groups: list[str], out: Path):
    """Inference scaling across maps: A* cost grows with the map, a fixed
    forward pass does not. Lines connect the same (mode, planner) cell across
    maps; log scale because Pi/desktop and A*/PPO differ by up to ~10x."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    labels = list(dfs)
    xs = np.arange(len(labels))
    for ax, col, title in (
            (axes[0], "planner_inference_mean_us", "Inference (per-run mean)"),
            (axes[1], "planner_inference_p99_us", "Inference (per-run p99)")):
        for g in groups:
            m, p = g.split("/", 1)
            ys, errs = [], []
            for label in labels:
                vals = pd.to_numeric(dfs[label][dfs[label]["group"] == g].get(col),
                                     errors="coerce").dropna().to_numpy()
                ys.append(vals.mean() if len(vals) else np.nan)
                errs.append(_ci95(vals) if len(vals) else 0.0)
            ax.errorbar(xs, ys, yerr=errs, marker=MODE_MARKER.get(m, "o"),
                        ls=MODE_LS.get(m, "-"), color=PLANNER_COLOR.get(p, "#999"),
                        ms=8, mec="k", mew=0.6, capsize=3, lw=1.6, label=g)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.4, len(labels) - 0.6)
        ax.set_ylabel("µs (mean ± 95% CI)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle("Planner inference across maps — search scales, a forward pass doesn't",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "inference_by_map.png", dpi=130)
    plt.close(fig)


def fig_tradeoff(dfs: dict[str, pd.DataFrame], out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fills = ["full", "none", "left"]
    for k, (mlabel, df) in enumerate(dfs.items()):
        for g in ordered_group_keys(df):
            sub = df[(df["group"] == g) & df["success"]]
            m, p = g.split("/", 1)
            t = pd.to_numeric(sub["time_to_completion_s"], errors="coerce")
            e = pd.to_numeric(sub["energy_net_mwh"], errors="coerce")
            ok = t.notna() & e.notna()
            if not ok.any():
                continue
            ax.errorbar(t[ok].mean(), e[ok].mean(), xerr=_ci95(t[ok].to_numpy()),
                        yerr=_ci95(e[ok].to_numpy()),
                        fmt=MODE_MARKER.get(m, "o"), ms=11,
                        fillstyle=fills[k % len(fills)],
                        color=PLANNER_COLOR.get(p, "#999"), mec="k", mew=0.8,
                        capsize=3, label=f"{mlabel}: {g}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("time to goal (s), mean ± 95% CI")
    ax.set_ylabel("net energy (mWh), mean ± 95% CI")
    ax.set_title("Time / net-energy trade-off across maps (successful runs)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "tradeoff_by_map.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+", help="label=path/to/runs_table.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    dfs = dict(_load(a) for a in args.tables)
    args.out.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(dfs.values(), ignore_index=True)
    groups = ordered_group_keys(combined)

    rows = []
    for mlabel, df in dfs.items():
        for g in groups:
            sub = df[df["group"] == g]
            if not len(sub):
                continue
            ok = sub[sub["success"]]
            row = {"map": mlabel, "group": g, "n": len(sub),
                   "n_success": len(ok),
                   "genuine_success_rate": round(sub["genuine_success"].mean(), 3)}
            for col, _ in METRICS:  # outcome means over successes only
                vals = pd.to_numeric(ok.get(col), errors="coerce").dropna()
                row[f"{col}_mean"] = round(vals.mean(), 4) if len(vals) else None
            for col in ("planner_inference_mean_us", "tcp_rtt_mean_us"):
                vals = pd.to_numeric(sub.get(col), errors="coerce").dropna()
                row[f"{col}_mean"] = round(vals.mean(), 1) if len(vals) else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.out / "map_comparison.csv", index=False)

    fig_success(dfs, groups, args.out)
    fig_metrics(dfs, groups, args.out)
    fig_tradeoff(dfs, args.out)
    fig_inference(dfs, groups, args.out)
    print(f"wrote map_comparison.csv + 4 figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
