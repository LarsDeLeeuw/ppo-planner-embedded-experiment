"""analyze_campaign.py - Campaign-level figures + paired statistical comparisons.

    python analyze_campaign.py <campaign_root> [--out <figures_dir>] [--refresh]

Produces the same figures as analyze_session over the whole campaign, plus
paired comparisons (handover_analysis §5):
  - planner_A vs planner_B within a mode, paired by (map_id, goal_cell)
  - centralized vs decentralized within a planner, paired by (planner, map_id, goal_cell)
Writes paired_comparisons.csv. Each metric is aggregated to one value per
(condition, pairing-key) by mean before pairing; pairs present in both
conditions feed a Wilcoxon signed-rank test, with a Mann-Whitney U fallback
(on the raw per-run values) when fewer than 5 pairs exist.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import load_summaries, split_and_filter  # noqa: E402
from analyze_session import build_figures  # noqa: E402

# Metrics compared (lower-is-better noted only for the reader; we report signed diff).
COMPARE_METRICS = [
    "time_to_completion_s",
    "executed_path_length_m",
    "energy_net_mwh",
    "energy_harvested_mwh",
    "energy_consumed_opencr_mwh",
]
MIN_PAIRS = 5
PAIR_KEY = ["map_id", "goal_cell"]


def _bootstrap_median_ci(diffs: np.ndarray, n_boot: int = 2000,
                         seed: int = 0) -> tuple[float, float]:
    if len(diffs) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(diffs, size=len(diffs), replace=True))
            for _ in range(n_boot)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _paired(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str,
            key: list[str]) -> dict | None:
    """Pair two conditions on `key` (mean per key) and test the metric.

    Returns a result dict or None if there is no usable data for this metric.
    """
    a = df_a.dropna(subset=[metric]) if metric in df_a.columns else df_a.iloc[0:0]
    b = df_b.dropna(subset=[metric]) if metric in df_b.columns else df_b.iloc[0:0]
    if a.empty or b.empty:
        return None
    ga = a.groupby(key)[metric].mean()
    gb = b.groupby(key)[metric].mean()
    common = ga.index.intersection(gb.index)
    n_pairs = len(common)
    res = {"metric": metric, "n_pairs": n_pairs}
    if n_pairs >= 1:
        diffs = (ga.loc[common] - gb.loc[common]).to_numpy()
        res["median_diff_A_minus_B"] = round(float(np.median(diffs)), 4)
        lo, hi = _bootstrap_median_ci(diffs)
        res["ci95_lo"] = round(lo, 4)
        res["ci95_hi"] = round(hi, 4)
    if n_pairs >= MIN_PAIRS and np.any(diffs != 0):
        try:
            stat, p = stats.wilcoxon(diffs)
            res["test"] = "wilcoxon"
            res["stat"] = round(float(stat), 4)
            res["p_value"] = round(float(p), 5)
        except ValueError as e:
            res["test"] = f"wilcoxon-failed:{e}"
    else:
        # Fallback: unpaired Mann-Whitney U on raw per-run values.
        try:
            stat, p = stats.mannwhitneyu(a[metric].to_numpy(), b[metric].to_numpy(),
                                         alternative="two-sided")
            res["test"] = "mannwhitneyu(fallback)"
            res["stat"] = round(float(stat), 4)
            res["p_value"] = round(float(p), 5)
            res["note"] = f"fewer than {MIN_PAIRS} pairs; reduced inferential weight"
        except ValueError as e:
            res["test"] = f"mannwhitneyu-failed:{e}"
    return res


def paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    modes = sorted(df["mode"].dropna().unique())
    planners = sorted(df["planner"].dropna().unique())

    # planner vs planner within each mode
    for mode in modes:
        sub = df[df["mode"] == mode]
        for pa, pb in itertools.combinations(planners, 2):
            da, db = sub[sub["planner"] == pa], sub[sub["planner"] == pb]
            if da.empty or db.empty:
                continue
            for metric in COMPARE_METRICS:
                r = _paired(da, db, metric, PAIR_KEY)
                if r:
                    rows.append({"comparison": f"{mode}: {pa} vs {pb}",
                                 "condition_A": pa, "condition_B": pb, **r})

    # centralized vs decentralized within each planner
    for planner in planners:
        sub = df[df["planner"] == planner]
        if {"centralized", "decentralized"} <= set(sub["mode"].unique()):
            da = sub[sub["mode"] == "centralized"]
            db = sub[sub["mode"] == "decentralized"]
            for metric in COMPARE_METRICS:
                r = _paired(da, db, metric, ["planner", *PAIR_KEY])
                if r:
                    rows.append({"comparison": f"{planner}: centralized vs decentralized",
                                 "condition_A": "centralized", "condition_B": "decentralized",
                                 **r})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    df = load_summaries(args.root, refresh=args.refresh)
    if df.empty:
        print("no runs found under", args.root, file=sys.stderr)
        return 1
    kept, baselines, log = split_and_filter(df)
    for line in log:
        print("  filter:", line, file=sys.stderr)

    out_dir = args.out or (args.root / "figures")
    written = build_figures(kept, baselines, out_dir)
    print(f"wrote {len(written)} figures to {out_dir}")

    comparisons = paired_comparisons(kept)
    csv_path = out_dir / "paired_comparisons.csv"
    comparisons.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(comparisons)} comparison rows)")
    if not comparisons.empty:
        cols = [c for c in ["comparison", "metric", "n_pairs", "median_diff_A_minus_B",
                            "p_value", "test"] if c in comparisons.columns]
        print(comparisons[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
