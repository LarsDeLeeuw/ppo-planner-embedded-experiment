"""analyze_day.py - One-map experiment-day report (6 mode x planner cells).

    python analyze_day.py <experiments_root> --map day3 [--out DIR] [--refresh]

Discovers every run under <experiments_root> (recursively), keeps the runs of
the requested map, applies the standard filter (dry_run / aborted / chrony),
and writes a self-contained report directory:

  run_inventory.csv      crash-aware per-cell accounting (tracker app crashes)
  runs_table.csv         one row per kept run incl. derived metrics
                         (this file is the input to compare_maps.py)
  metrics_by_group.csv   mean/std/median/min/max per (mode, planner)
  stats_tests.csv        Mann-Whitney U + Cliff's delta + Hodges-Lehmann shift
                         for planner pairs within mode and mode pairs within
                         planner (unpaired: one map => pairing key is constant)
  figures: success_rates, time_to_goal, path_length, energy_harvested,
           net_energy, energy_decomposition, tradeoff_time_net,
           tradeoff_path_harvest, efficiency, mode_overhead, latency_by_group,
           inference_time, turns_vs_energy

Success semantics: a goal_assisted finish (orchestrator drove the final step)
counts toward overall success but NOT toward genuine planner success; the
success figure shows both.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import wilson_ci  # noqa: E402
from day_aggregate import (  # noqa: E402
    MODE_HATCH, MODE_LS, MODE_MARKER, MODE_ORDER, PLANNER_COLOR, PLANNER_ORDER,
    load_day, ordered_group_keys, run_inventory,
)

REPORT_NOTE = "successful runs"  # outcome metrics exclude failed runs: a failed
# run's partial path/time/energy is not comparable (source-paper MCT/APL semantics)
BOX_METRICS = {
    "time_to_goal": ("time_to_completion_s", "seconds",
                     f"Time to goal ({REPORT_NOTE})", True),
    "path_length": ("executed_path_length_m", "metres",
                    f"Executed path length ({REPORT_NOTE})", True),
    "energy_harvested": ("energy_harvested_mwh", "mWh",
                         f"Energy harvested ({REPORT_NOTE}; compare within one lighting block)", True),
    "net_energy": ("energy_net_mwh", "mWh",
                   f"Net energy balance ({REPORT_NOTE}; harvested - consumed on robot)", True),
}
TEST_METRICS = [
    "time_to_completion_s", "executed_path_length_m", "energy_harvested_mwh",
    "energy_net_mwh", "consumed_robot_mwh", "harvest_power_mw",
    "path_efficiency", "avg_speed_mps", "turn_count", "n_steps",
]
RUNS_TABLE_COLS = [
    "run_id", "session_id", "mode", "planner", "group", "map_id", "goal_cell",
    "start_cell", "outcome", "success", "genuine_success", "assisted_success",
    "goal_assisted", "time_to_completion_s", "executed_path_length_m",
    "executed_path_length_cells", "octile_optimal_m", "path_efficiency",
    "avg_speed_mps", "energy_harvested_mwh", "energy_consumed_sbc_mwh",
    "energy_consumed_opencr_mwh", "consumed_robot_mwh", "energy_net_mwh",
    "energy_consumed_desktop_mwh", "harvest_power_mw", "net_power_mw",
    "harvest_per_m_mwh", "consumed_per_m_mwh", "n_steps", "turn_count",
    "turns_per_step", "diagonal_step_frac", "planner_inference_mean_us",
    "planner_inference_p99_us", "tcp_rtt_mean_us", "battery_voltage_start_v",
    "battery_voltage_end_v", "chrony_offset_max_ms", "n_warnings", "run_dir",
]


# -- plotting helpers ---------------------------------------------------------

def _planners(df) -> list[str]:
    present = df["planner"].dropna().unique()
    return [p for p in PLANNER_ORDER if p in present] + sorted(set(present) - set(PLANNER_ORDER))


def _modes(df) -> list[str]:
    present = df["mode"].dropna().unique()
    return [m for m in MODE_ORDER if m in present] + sorted(set(present) - set(MODE_ORDER))


def _group_positions(planners, modes):
    """x positions: modes side by side within each planner cluster."""
    pos = {}
    for i, p in enumerate(planners):
        for j, m in enumerate(modes):
            pos[(m, p)] = i + (j - (len(modes) - 1) / 2) * 0.32
    return pos


def _style_group_axis(ax, planners):
    ax.set_xticks(range(len(planners)))
    ax.set_xticklabels(planners, fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def _mode_legend(ax, modes, **kw):
    handles = [plt.Line2D([], [], color="#555", marker=MODE_MARKER[m], ls="none",
                          label=m) for m in modes]
    ax.legend(handles=handles, fontsize=8, **kw)


def grouped_box(ax, df, col, title, ylabel, success_only=False):
    """Box+jitter per (mode, planner); planner = colour, mode = position/marker."""
    planners, modes = _planners(df), _modes(df)
    pos = _group_positions(planners, modes)
    for p in planners:
        for m in modes:
            sub = df[(df["planner"] == p) & (df["mode"] == m)]
            if success_only:
                sub = sub[sub["success"]]
            vals = pd.to_numeric(sub.get(col), errors="coerce").dropna().to_numpy()
            if not len(vals):
                continue
            x = pos[(m, p)]
            bp = ax.boxplot([vals], positions=[x], widths=0.26, showmeans=True,
                            patch_artist=True)
            bp["boxes"][0].set(facecolor=PLANNER_COLOR.get(p, "#999"), alpha=0.35,
                               hatch=MODE_HATCH[m])
            bp["medians"][0].set(color="k")
            rng = np.random.default_rng(0)
            ax.scatter(x + rng.normal(0, 0.035, len(vals)), vals, s=14, zorder=3,
                       color=PLANNER_COLOR.get(p, "#999"), marker=MODE_MARKER[m],
                       edgecolors="k", linewidths=0.4, alpha=0.85)
            ax.annotate(f"n={len(vals)}", (x, np.min(vals)),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=7, color="#555")
    _style_group_axis(ax, planners)
    _mode_legend(ax, modes)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def _save(fig, path: Path, written: list[Path]):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)


# -- figures ------------------------------------------------------------------

def fig_success_rates(df, out, written):
    """Genuine (solid) + goal-assisted (hatched) success, Wilson CI on overall."""
    planners, modes = _planners(df), _modes(df)
    pos = _group_positions(planners, modes)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for p in planners:
        for m in modes:
            sub = df[(df["planner"] == p) & (df["mode"] == m)]
            n = len(sub)
            if not n:
                continue
            k_gen = int(sub["genuine_success"].sum())
            k_ass = int(sub["assisted_success"].sum())
            x = pos[(m, p)]
            c = PLANNER_COLOR.get(p, "#999")
            ax.bar(x, k_gen / n, width=0.28, color=c, alpha=0.85, hatch=MODE_HATCH[m])
            if k_ass:
                ax.bar(x, k_ass / n, bottom=k_gen / n, width=0.28, color=c,
                       alpha=0.35, hatch="..")
            phat, lo, hi = wilson_ci(k_gen + k_ass, n)
            ax.errorbar(x, phat, yerr=[[max(0, phat - lo)], [max(0, hi - phat)]],
                        fmt="none", ecolor="k", capsize=3, lw=1)
            label = f"{k_gen}/{n}" + (f" (+{k_ass})" if k_ass else "")
            ax.annotate(label, (x, min(1.04, hi + 0.02)), ha="center", fontsize=8)
    _style_group_axis(ax, planners)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("success rate")
    ax.set_title("Success rate — solid: genuine, dotted overlay: goal-assisted "
                 "(Wilson 95% CI on overall)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#bbb", hatch=MODE_HATCH[m], label=m)
               for m in modes]
    # below the axes — inside the axes it overlaps the PPO bars
    ax.legend(handles=handles, fontsize=8, ncol=len(modes), loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    _save(fig, out / "success_rates.png", written)


def fig_energy_decomposition(df, out, written):
    """Per group: harvested above zero, consumption stacked below, net diamond.
    Successful runs only — failed runs' partial energy budgets flatter the mean."""
    df = df[df["success"]]
    groups = ordered_group_keys(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(groups))
    for i, g in enumerate(groups):
        sub = df[df["group"] == g]
        m, p = g.split("/", 1)
        harv = pd.to_numeric(sub["energy_harvested_mwh"], errors="coerce").mean()
        sbc = pd.to_numeric(sub["energy_consumed_sbc_mwh"], errors="coerce").mean()
        ocr = pd.to_numeric(sub["energy_consumed_opencr_mwh"], errors="coerce").mean()
        net = pd.to_numeric(sub["energy_net_mwh"], errors="coerce").mean()
        c = PLANNER_COLOR.get(p, "#999")
        ax.bar(i, harv, width=0.6, color=c, alpha=0.8, hatch=MODE_HATCH[m])
        if pd.notna(harv):  # harvest is orders of magnitude below consumption
            ax.annotate(f"harvested\n+{harv:.4f}", (i, 0.15), ha="center",
                        fontsize=7, color=c)
        ax.bar(i, -sbc, width=0.6, color="#8c564b", alpha=0.75, hatch=MODE_HATCH[m],
               label="SBC (consumed)" if i == 0 else None)
        ax.bar(i, -ocr, bottom=-sbc, width=0.6, color="#7f7f7f", alpha=0.75,
               hatch=MODE_HATCH[m], label="OpenCR (consumed)" if i == 0 else None)
        ax.plot(i, net, "D", color="k", ms=7, zorder=4,
                label="net (mean)" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=8, rotation=12)
    ax.set_ylabel("mWh (group mean; harvested up, consumed down)")
    ax.set_title("Energy decomposition per run (successful runs) — "
                 "harvested (planner colour) vs consumed")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out / "energy_decomposition.png", written)


def _tradeoff_scatter(df, xcol, ycol, xlabel, ylabel, title, fname, out, written,
                      hline0=False):
    planners, modes = _planners(df), _modes(df)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for p in planners:
        for m in modes:
            sub = df[(df["planner"] == p) & (df["mode"] == m)]
            x = pd.to_numeric(sub.get(xcol), errors="coerce")
            y = pd.to_numeric(sub.get(ycol), errors="coerce")
            ok = x.notna() & y.notna()
            if not ok.any():
                continue
            c = PLANNER_COLOR.get(p, "#999")
            ax.scatter(x[ok], y[ok], s=34, color=c, marker=MODE_MARKER[m],
                       edgecolors="k", linewidths=0.4, alpha=0.8,
                       label=f"{m}/{p}")
            # group centroid + std crosshair
            ax.errorbar(x[ok].mean(), y[ok].mean(), xerr=x[ok].std(), yerr=y[ok].std(),
                        fmt=MODE_MARKER[m], color=c, ms=13, mec="k", mew=1.2,
                        ecolor=c, elinewidth=1.2, capsize=3, alpha=0.9, zorder=4)
    if hline0:
        ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, out / fname, written)


def fig_efficiency(df, out, written):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    grouped_box(axes[0], df, "path_efficiency",
                "Path efficiency (octile-optimal / executed, successes)",
                "ratio (1 = optimal)", success_only=True)
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
    grouped_box(axes[1], df, "harvest_power_mw",
                "Mean harvest power (successes)", "mW", success_only=True)
    grouped_box(axes[2], df, "consumed_per_m_mwh",
                "Robot energy cost of locomotion (successes)", "mWh per metre",
                success_only=True)
    _save(fig, out / "efficiency.png", written)


def fig_mode_overhead(df, out, written):
    """Centralized minus decentralized group means (successes), bootstrap 95% CI."""
    df = df[df["success"]]
    metrics = [("time_to_completion_s", "time to goal (s)"),
               ("consumed_robot_mwh", "robot consumed (mWh)"),
               ("energy_consumed_sbc_mwh", "SBC consumed (mWh)"),
               ("energy_net_mwh", "net energy (mWh)")]
    planners = _planners(df)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 4.4))
    for ax, (col, label) in zip(axes, metrics):
        for i, p in enumerate(planners):
            a = pd.to_numeric(df[(df["planner"] == p) & (df["mode"] == "centralized")]
                              .get(col), errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(df[(df["planner"] == p) & (df["mode"] == "decentralized")]
                              .get(col), errors="coerce").dropna().to_numpy()
            if not (len(a) and len(b)):
                continue
            diff = a.mean() - b.mean()
            boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
                    for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            c = PLANNER_COLOR.get(p, "#999")
            ax.bar(i, diff, width=0.55, color=c, alpha=0.8)
            ax.errorbar(i, diff, yerr=[[diff - lo], [hi - diff]], fmt="none",
                        ecolor="k", capsize=4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(planners)))
        ax.set_xticklabels(planners, fontsize=8, rotation=12)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Centralized − decentralized (successful runs; group-mean "
                 "difference, bootstrap 95% CI)", fontsize=11)
    _save(fig, out / "mode_overhead.png", written)


def fig_latency(df, out, written):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    grouped_box(axes[0], df, "planner_inference_mean_us",
                "Planner inference (per-run mean)", "µs")
    grouped_box(axes[1], df, "tcp_rtt_mean_us",
                "Predict TCP round-trip (per-run mean)", "µs")
    _save(fig, out / "latency_by_group.png", written)


def fig_turns_vs_energy(df, out, written):
    """The locomotion-cost mechanism: turns are pure cost (in-place rotation,
    zero progress), so when step counts are comparable the turn count decides
    consumed — and hence net — energy. Successful runs only: a failed run's
    partial step/turn tally is not comparable."""
    ok = df[df["success"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    grouped_box(axes[0], ok, "turn_count",
                "In-place heading changes per run (successes)", "turns")
    # steps context: annotate group-mean step count under each planner cluster
    planners = _planners(ok)
    for i, p in enumerate(planners):
        steps = pd.to_numeric(ok[ok["planner"] == p]["n_steps"],
                              errors="coerce").dropna()
        if len(steps):
            axes[0].annotate(f"~{steps.mean():.0f} steps", (i, axes[0].get_ylim()[0]),
                             textcoords="offset points", xytext=(0, -30),
                             ha="center", fontsize=8, color="#555")

    ax = axes[1]
    rng = np.random.default_rng(0)
    for p in planners:
        for m in _modes(ok):
            sub = ok[(ok["planner"] == p) & (ok["mode"] == m)]
            t = pd.to_numeric(sub["turn_count"], errors="coerce")
            e = pd.to_numeric(sub["consumed_robot_mwh"], errors="coerce")
            sel = t.notna() & e.notna()
            if not sel.any():
                continue
            ax.scatter(t[sel] + rng.normal(0, 0.08, int(sel.sum())), e[sel], s=34,
                       color=PLANNER_COLOR.get(p, "#999"), marker=MODE_MARKER[m],
                       edgecolors="k", linewidths=0.4, alpha=0.8, label=f"{m}/{p}")
    tt = pd.to_numeric(ok["turn_count"], errors="coerce")
    ee = pd.to_numeric(ok["consumed_robot_mwh"], errors="coerce")
    sel = tt.notna() & ee.notna()
    if sel.sum() > 2:
        r = np.corrcoef(tt[sel], ee[sel])[0, 1]
        k, b = np.polyfit(tt[sel], ee[sel], 1)
        xs = np.linspace(tt[sel].min(), tt[sel].max(), 20)
        ax.plot(xs, k * xs + b, "--", color="#555", lw=1,
                label=f"fit: {k:.2f} mWh/turn (r={r:.2f})")
    ax.set_xlabel("turn count")
    ax.set_ylabel("robot energy consumed (mWh)")
    ax.set_title("Consumed energy vs turn count (successes)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, out / "turns_vs_energy.png", written)


def fig_inference(df, out, written):
    """Inference cost in context: level, tail, and share of the round-trip.

    Log scale on the time panels — Pi vs desktop differs ~4x and A* vs PPO
    up to ~10x, which a linear axis flattens into unreadability.
    """
    df = df.copy()
    inf_us = pd.to_numeric(df["planner_inference_mean_us"], errors="coerce")
    rtt_us = pd.to_numeric(df["tcp_rtt_mean_us"], errors="coerce")
    df["inference_rtt_share_pct"] = (100.0 * inf_us / rtt_us).round(2)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    grouped_box(axes[0], df, "planner_inference_mean_us",
                "Inference (per-run mean)", "µs")
    axes[0].set_yscale("log")
    grouped_box(axes[1], df, "planner_inference_p99_us",
                "Inference (per-run p99)", "µs")
    axes[1].set_yscale("log")
    grouped_box(axes[2], df, "inference_rtt_share_pct",
                "Inference share of predict round-trip", "% of TCP RTT")
    _save(fig, out / "inference_time.png", written)


# -- tables -------------------------------------------------------------------

def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a>b) - P(a<b); |d|<0.147 negligible, <0.33 small, <0.474 medium."""
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def _hl_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Hodges-Lehmann estimator: median of all pairwise differences a - b."""
    return float(np.median(np.subtract.outer(a, b)))


def stats_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Unpaired two-sided Mann-Whitney U for every within-mode planner pair and
    within-planner mode pair, with effect sizes. One map / one goal => the
    paired Wilcoxon key (map_id, goal_cell) is constant, so unpaired is the
    honest test here; cross-map pairing lives in analyze_campaign."""
    rows = []

    success_only = set(TEST_METRICS)  # all outcome metrics: failed-run partial
    # paths/times/energies contaminate comparisons (inference metrics, tested
    # nowhere here, would be per-call hardware properties and exempt)

    def compare(label, da, db, name_a, name_b):
        for metric in TEST_METRICS:
            sa = da[da["success"]] if metric in success_only else da
            sb = db[db["success"]] if metric in success_only else db
            a = pd.to_numeric(sa.get(metric), errors="coerce").dropna().to_numpy()
            b = pd.to_numeric(sb.get(metric), errors="coerce").dropna().to_numpy()
            if len(a) < 3 or len(b) < 3:
                continue
            try:
                stat, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                continue
            rows.append({
                "comparison": label, "metric": metric,
                "A": name_a, "B": name_b, "n_A": len(a), "n_B": len(b),
                "mean_A": round(a.mean(), 4), "mean_B": round(b.mean(), 4),
                "hl_shift_A_minus_B": round(_hl_shift(a, b), 4),
                "cliffs_delta": round(_cliffs_delta(a, b), 3),
                "U": round(float(stat), 1), "p_value": round(float(pval), 5),
            })

    for mode in _modes(df):
        sub = df[df["mode"] == mode]
        for pa, pb in itertools.combinations(_planners(sub), 2):
            compare(f"{mode}: {pa} vs {pb}",
                    sub[sub["planner"] == pa], sub[sub["planner"] == pb], pa, pb)
    for planner in _planners(df):
        sub = df[df["planner"] == planner]
        if {"centralized", "decentralized"} <= set(sub["mode"]):
            compare(f"{planner}: centralized vs decentralized",
                    sub[sub["mode"] == "centralized"],
                    sub[sub["mode"] == "decentralized"],
                    "centralized", "decentralized")
    return pd.DataFrame(rows)


def metrics_by_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in ordered_group_keys(df):
        sub = df[df["group"] == g]
        for metric in TEST_METRICS + ["planner_inference_mean_us", "tcp_rtt_mean_us"]:
            vals = pd.to_numeric(sub.get(metric), errors="coerce").dropna()
            if not len(vals):
                continue
            rows.append({
                "group": g, "metric": metric, "n": len(vals),
                "mean": round(vals.mean(), 4), "std": round(vals.std(), 4),
                "median": round(vals.median(), 4),
                "min": round(vals.min(), 4), "max": round(vals.max(), 4),
            })
    return pd.DataFrame(rows)


# -- main ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="experiments root (sessions discovered recursively)")
    ap.add_argument("--map", dest="map_id", default=None,
                    help="keep only runs with this metadata map_id (e.g. day3)")
    ap.add_argument("--out", type=Path, default=None,
                    help="report dir (default <root>/figures/report-<map>)")
    ap.add_argument("--refresh", action="store_true", help="regenerate every summary.json")
    args = ap.parse_args(argv)

    df_all, kept, log = load_day(args.root, args.map_id, refresh=args.refresh)
    if df_all.empty:
        print("no runs found", "for map " + args.map_id if args.map_id else "",
              "under", args.root, file=sys.stderr)
        return 1
    for line in log:
        print("  filter:", line, file=sys.stderr)
    for _, r in kept.iterrows():
        for wmsg in (r.get("warnings") or []):
            print(f"  warn[{r['run_id']}]: {wmsg}", file=sys.stderr)

    out = args.out or (args.root / "figures" / f"report-{args.map_id or 'all'}")
    out.mkdir(parents=True, exist_ok=True)

    inv = run_inventory(df_all, kept)
    inv.to_csv(out / "run_inventory.csv", index=False)
    print("\nRun inventory (tracker crashes counted separately):")
    print(inv.to_string(index=False))

    cols = [c for c in RUNS_TABLE_COLS if c in kept.columns]
    kept[cols].to_csv(out / "runs_table.csv", index=False)
    metrics_by_group(kept).to_csv(out / "metrics_by_group.csv", index=False)
    tests = stats_tests(kept)
    tests.to_csv(out / "stats_tests.csv", index=False)

    written: list[Path] = []
    fig_success_rates(kept, out, written)
    for name, (col, ylabel, title, success_only) in BOX_METRICS.items():
        fig, ax = plt.subplots(figsize=(8, 4.6))
        grouped_box(ax, kept, col, title, ylabel, success_only=success_only)
        if name == "net_energy":
            ax.axhline(0, color="k", lw=0.8, ls="--")
        if name == "path_length":
            opt = pd.to_numeric(kept["octile_optimal_m"], errors="coerce").dropna()
            if len(opt):
                ax.axhline(opt.iloc[0], color="k", lw=0.8, ls="--",
                           label=f"octile optimal {opt.iloc[0]:.2f} m")
                ax.legend(fontsize=8)
        _save(fig, out / f"{name}.png", written)
    fig_energy_decomposition(kept, out, written)
    _tradeoff_scatter(kept, "time_to_completion_s", "energy_net_mwh",
                      "run duration (s)", "net energy (mWh)",
                      "Net energy vs duration (all kept runs incl. failures)",
                      "tradeoff_time_net.png", out, written, hline0=True)
    _tradeoff_scatter(kept[kept["success"]], "executed_path_length_m",
                      "energy_harvested_mwh",
                      "executed path length (m)", "energy harvested (mWh)",
                      "Trade-off: extra distance vs extra harvest (successes)",
                      "tradeoff_path_harvest.png", out, written)
    fig_efficiency(kept, out, written)
    fig_mode_overhead(kept, out, written)
    fig_latency(kept, out, written)
    fig_inference(kept, out, written)
    fig_turns_vs_energy(kept, out, written)

    print(f"\nwrote {len(written)} figures + 4 CSVs to {out}")
    if not tests.empty:
        sig = tests[tests["p_value"] < 0.05]
        print(f"\n{len(sig)}/{len(tests)} comparisons significant at p<0.05:")
        if len(sig):
            print(sig[["comparison", "metric", "hl_shift_A_minus_B",
                       "cliffs_delta", "p_value"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
