"""analyze_session.py - Cross-run figures for a directory of runs.

    python analyze_session.py <root> [--out <figures_dir>] [--refresh]

`<root>` is any folder containing run subfolders (a session's runs/, or a whole
campaign tree). Runs are discovered recursively, summary.json (re)generated as
needed, filtered per handover_analysis §5, grouped by (mode x planner), and
rendered into the 5 required + 4 extra report figures.

Importable: build_figures(df_kept, df_baselines, out_dir) for Jupyter use.
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
from aggregate import (  # noqa: E402
    load_summaries, ordered_groups, split_and_filter, wilson_ci,
)


def _box_by_group(ax, df, value_col, groups, title, ylabel, success_only=False):
    data, labels, present = [], [], []
    for g in groups:
        sub = df[df["group"] == g]
        if success_only:
            sub = sub[sub["outcome"] == "success"]
        vals = pd.to_numeric(sub.get(value_col), errors="coerce").dropna().tolist() \
            if value_col in sub.columns else []
        if vals:
            data.append(vals); labels.append(g); present.append(g)
    if not data:
        ax.text(0.5, 0.5, f"no data for {value_col}", ha="center", va="center",
                transform=ax.transAxes)
    else:
        # Set tick labels after the call (the boxplot `labels` kwarg was renamed
        # tick_labels in mpl 3.9 — avoid both by labelling the axis directly).
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        for i in range(1, len(data) + 1):  # scatter overlay
            vals = data[i - 1]
            ax.scatter(np.random.normal(i, 0.04, len(vals)), vals, s=12,
                       alpha=0.6, color="#333")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=8)


def _fig(out_dir: Path, name: str):
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * 4), 4.5))
    return fig, ax, out_dir / name


def build_figures(df: pd.DataFrame, baselines: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = ordered_groups(df)
    written: list[Path] = []

    def save(fig, path):
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
        written.append(path)

    # 1) success_rate.png — Wilson 95% CI
    fig, ax, path = _fig(out_dir, "success_rate.png")
    xs, rates, los, his = [], [], [], []
    for i, g in enumerate(groups):
        sub = df[df["group"] == g]
        n = len(sub); k = int((sub["outcome"] == "success").sum())
        phat, lo, hi = wilson_ci(k, n)
        xs.append(i); rates.append(phat); los.append(phat - lo); his.append(hi - phat)
        ax.annotate(f"{k}/{n}", (i, phat), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.bar(xs, rates, color="#4a90d9")
    # Clamp tiny floating-point negatives (e.g. when phat==1.0 the Wilson upper
    # bound can round just under 1.0); errorbar rejects negative yerr.
    yerr = [np.clip(los, 0, None), np.clip(his, 0, None)]
    ax.errorbar(xs, rates, yerr=yerr, fmt="none", ecolor="k", capsize=4)
    ax.set_xticks(xs); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel("success rate"); ax.set_title("Success rate (Wilson 95% CI)")
    save(fig, path)

    # 2) path_length.png (meters; cells/octile are secondary in summary.json)
    fig, ax, path = _fig(out_dir, "path_length.png")
    _box_by_group(ax, df, "executed_path_length_m", groups,
                  "Executed path length", "metres")
    save(fig, path)

    # 3) time_to_goal.png (successful runs only)
    fig, ax, path = _fig(out_dir, "time_to_goal.png")
    _box_by_group(ax, df, "time_to_completion_s", groups,
                  "Time to goal (successful runs)", "seconds", success_only=True)
    save(fig, path)

    # 4) energy_harvested.png
    fig, ax, path = _fig(out_dir, "energy_harvested.png")
    _box_by_group(ax, df, "energy_harvested_mwh", groups,
                  "Energy harvested (lighting-dependent; compare within block)", "mWh")
    save(fig, path)

    # 5) net_energy_balance.png
    fig, ax, path = _fig(out_dir, "net_energy_balance.png")
    _box_by_group(ax, df, "energy_net_mwh", groups,
                  "Net energy balance (harvested - consumed)", "mWh")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    save(fig, path)

    # 6) planner_inference_latency.png — violin per planner
    fig, ax, path = _fig(out_dir, "planner_inference_latency.png")
    planners = sorted(df["planner"].dropna().unique())
    vdata = [pd.to_numeric(df[df["planner"] == p]["planner_inference_mean_us"],
                           errors="coerce").dropna().tolist() for p in planners]
    vdata_nonempty = [(p, v) for p, v in zip(planners, vdata) if v]
    if vdata_nonempty:
        ax.violinplot([v for _, v in vdata_nonempty], showmeans=True)
        ax.set_xticks(range(1, len(vdata_nonempty) + 1))
        ax.set_xticklabels([p for p, _ in vdata_nonempty], fontsize=8)
    ax.set_ylabel("inference µs (per-run mean)"); ax.set_title("Planner inference latency")
    save(fig, path)

    # 7) tcp_rtt_decomposition.png — stacked bar per group
    fig, ax, path = _fig(out_dir, "tcp_rtt_decomposition.png")
    inf, brg, tcp = [], [], []
    for g in groups:
        sub = df[df["group"] == g]
        rtt = pd.to_numeric(sub.get("tcp_rtt_mean_us"), errors="coerce").mean()
        b = pd.to_numeric(sub.get("bridge_plumbing_mean_us"), errors="coerce").mean()
        t = pd.to_numeric(sub.get("tcp_only_mean_us"), errors="coerce").mean()
        i = (rtt - b - t) if pd.notna(rtt) else np.nan
        inf.append(i if pd.notna(i) else 0)
        brg.append(b if pd.notna(b) else 0)
        tcp.append(t if pd.notna(t) else 0)
    xs = range(len(groups))
    ax.bar(xs, tcp, label="tcp", color="#9467bd")
    ax.bar(xs, brg, bottom=tcp, label="bridge plumbing", color="#8c564b")
    ax.bar(xs, inf, bottom=np.array(tcp) + np.array(brg), label="inference", color="#d62728")
    ax.set_xticks(list(xs)); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("µs (single-host deltas)"); ax.set_title("Predict round-trip decomposition")
    ax.legend(fontsize=8)
    save(fig, path)

    # 8) control_loop_health.png
    fig, ax, path = _fig(out_dir, "control_loop_health.png")
    means, flagged = [], []
    for g in groups:
        sub = df[df["group"] == g]
        means.append(pd.to_numeric(sub.get("control_loop_mean_period_ms"), errors="coerce").mean())
        flagged.append(int((pd.to_numeric(sub.get("control_loop_overruns"),
                                          errors="coerce").fillna(0) > 0).sum()))
    xs = range(len(groups))
    bars = ax.bar(xs, [m if pd.notna(m) else 0 for m in means], color="#2ca02c")
    for i, f in enumerate(flagged):
        if f:
            ax.annotate(f"{f} run(s)\nw/ overruns", (i, means[i] or 0),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=7, color="red")
    ax.axhline(50.0, color="k", lw=0.8, ls="--", label="nominal 50 ms")
    ax.set_xticks(list(xs)); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("mean control loop period (ms)"); ax.set_title("Control loop health")
    ax.legend(fontsize=8)
    save(fig, path)

    # 9) desktop_energy_centralized.png (passthrough-ok runs only)
    fig, ax, path = _fig(out_dir, "desktop_energy_centralized.png")
    cen = df[(df["mode"] == "centralized") &
             pd.to_numeric(df.get("energy_consumed_desktop_mwh"), errors="coerce").notna()]
    if len(cen):
        _box_by_group(ax, cen, "energy_consumed_desktop_mwh", ordered_groups(cen),
                      "Desktop CPU energy (RAPL, centralized)", "mWh")
    else:
        ax.text(0.5, 0.5, "no centralized runs with RAPL data", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("Desktop CPU energy (RAPL, centralized)")
    save(fig, path)

    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="figures dir (default <root>/figures)")
    ap.add_argument("--refresh", action="store_true", help="regenerate every summary.json")
    args = ap.parse_args(argv)

    df = load_summaries(args.root, refresh=args.refresh)
    if df.empty:
        print("no runs found under", args.root, file=sys.stderr)
        return 1
    kept, baselines, log = split_and_filter(df)
    for line in log:
        print("  filter:", line, file=sys.stderr)
    print(f"{len(df)} runs discovered; {len(kept)} movement runs kept, "
          f"{len(baselines)} baseline(s), {len(df) - len(kept) - len(baselines)} filtered")
    # surface warnings present in kept runs
    for _, r in kept.iterrows():
        for wmsg in (r.get("warnings") or []):
            print(f"  warn[{r['run_id']}]: {wmsg}", file=sys.stderr)

    out_dir = args.out or (args.root / "figures")
    written = build_figures(kept, baselines, out_dir)
    print(f"wrote {len(written)} figures to {out_dir}:")
    for p in written:
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
