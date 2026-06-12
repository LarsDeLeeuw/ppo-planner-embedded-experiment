"""aggregate.py - Discover runs, load/refresh summary.json rows, filter, group.

Shared by analyze_session and analyze_campaign. A "run dir" is any folder
containing metadata.yaml. summary.json is (re)generated on demand via
summarize_run so cross-run analysis never depends on the operator having run
the per-run step first.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_run import summarize  # noqa: E402

CHRONY_REJECT_MS = 20.0


def discover_runs(root: Path) -> list[Path]:
    """All run folders under `root` (recursively), identified by metadata.yaml."""
    return sorted({p.parent for p in Path(root).rglob("metadata.yaml")})


def load_summaries(root: Path, *, refresh: bool = False) -> pd.DataFrame:
    """Build a DataFrame of summary rows, generating summary.json as needed."""
    rows = []
    for rd in discover_runs(root):
        sj = rd / "summary.json"
        if refresh or not sj.exists():
            summarize(rd)
        rows.append(json.loads(sj.read_text()))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["group"] = df["mode"].astype(str) + "/" + df["planner"].astype(str)
    df["n_warnings"] = df["warnings"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return df


def split_and_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (movement_runs_kept, baselines, filter_log).

    Filter order per handover_analysis §5: drop dry_run, drop aborted, drop
    chrony>20ms. Baselines (is_baseline) are split out, not filtered.
    """
    log: list[str] = []
    if df.empty:
        return df, df, log
    baselines = df[df.get("is_baseline", False) == True].copy()  # noqa: E712
    work = df[df.get("is_baseline", False) != True].copy()       # noqa: E712

    def drop(mask, reason):
        nonlocal work
        n = int(mask.sum())
        if n:
            for rid in work.loc[mask, "run_id"]:
                log.append(f"dropped {rid}: {reason}")
            work = work[~mask]

    drop(work["dry_run"] == True, "dry_run")                      # noqa: E712
    drop(work["outcome"] == "aborted", "outcome=aborted")
    if "chrony_offset_max_ms" in work.columns:
        drop(work["chrony_offset_max_ms"] > CHRONY_REJECT_MS,
             f"chrony_offset_max_ms > {CHRONY_REJECT_MS}")
    return work, baselines, log


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (phat, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def ordered_groups(df: pd.DataFrame) -> list[str]:
    return sorted(df["group"].dropna().unique().tolist())
