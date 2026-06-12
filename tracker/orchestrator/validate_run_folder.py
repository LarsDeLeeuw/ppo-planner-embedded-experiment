"""orchestrator.validate_run_folder - Post-hoc self-test for a run folder.

Verifies that runs/<run_id>/ has every file the analysis pipeline expects and
that metadata.yaml conforms to the v3 schema. Usable as:

    python -m orchestrator.validate_run_folder runs/<run_id>            # CLI
    from orchestrator.validate_run_folder import validate; validate(p)   # import

Returns 0 exit code only when ALL checks pass. Warnings (non-fatal) print to
stderr. Designed to be run by the orchestrator after each run completes, AND
by the operator on hand-inspected folders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REQUIRED_FILES = ("predict_log.jsonl", "metadata.yaml", "orchestrator.mcap")
REQUIRED_MAP_FILES = ("maps/obstacle.npy",)
REQUIRED_META_FIELDS = (
    "run_id", "session_id", "mode", "planner", "map_id", "started_at",
    "ended_at", "goal_cell", "start_cell", "outcome", "dry_run", "bag_capped",
    "warmup_inferences_us", "git_commits",
)
VALID_OUTCOMES = {"success", "failure", "aborted"}
VALID_MODES = {"decentralized", "centralized"}


def validate(run_dir: Path) -> tuple[bool, list[str], list[str]]:
    """Returns (ok, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    if not run_dir.is_dir():
        return False, [f"not a directory: {run_dir}"], []

    # --- presence checks ----------------------------------------------------
    for f in REQUIRED_FILES:
        p = run_dir / f
        if not p.exists():
            errors.append(f"missing: {f}")
        elif p.stat().st_size == 0:
            errors.append(f"empty: {f}")
    for f in REQUIRED_MAP_FILES:
        if not (run_dir / f).exists():
            errors.append(f"missing: {f}")

    # --- metadata schema ----------------------------------------------------
    meta_path = run_dir / "metadata.yaml"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = yaml.safe_load(meta_path.read_text()) or {}
        except yaml.YAMLError as e:
            errors.append(f"metadata.yaml is not valid yaml: {e}")
        for k in REQUIRED_META_FIELDS:
            if k not in meta:
                errors.append(f"metadata.yaml missing field: {k}")
        outcome = meta.get("outcome")
        if outcome not in VALID_OUTCOMES:
            errors.append(f"metadata.outcome={outcome!r} not in {VALID_OUTCOMES}")
        mode = meta.get("mode")
        if mode not in VALID_MODES:
            errors.append(f"metadata.mode={mode!r} not in {VALID_MODES}")
        if not isinstance(meta.get("warmup_inferences_us"), list):
            errors.append("warmup_inferences_us must be a list")
        # goal_cell: None (idle baseline) or [col, row] of two ints
        gc = meta.get("goal_cell")
        if gc is not None and not (
                isinstance(gc, list) and len(gc) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in gc)):
            errors.append(f"goal_cell={gc!r} must be null or [col, row] (two ints)")
        sc = meta.get("start_cell")
        if sc is not None and not (
                isinstance(sc, list) and len(sc) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in sc)):
            errors.append(f"start_cell={sc!r} must be [col, row] (two ints)")

    # --- bag presence per mode ---------------------------------------------
    if meta.get("mode") == "centralized" and not (run_dir / "desktop.mcap").exists():
        errors.append("centralized run missing desktop.mcap")
    if not (run_dir / "robot.mcap").exists():
        warnings.append("robot.mcap missing (real runs require it; mock runs do not)")

    # --- predict_log shape --------------------------------------------------
    pl = run_dir / "predict_log.jsonl"
    events: list[str] = []
    if pl.exists() and pl.stat().st_size > 0:
        for ln in pl.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError as e:
                errors.append(f"predict_log.jsonl: malformed line: {e}")
                break
            if "event" not in rec or "mono_ns" not in rec:
                errors.append("predict_log.jsonl: record missing event/mono_ns")
                break
            events.append(rec["event"])
        if events and "run_start" not in events:
            warnings.append("predict_log.jsonl: no run_start event")
        if events and "run_end" not in events and meta.get("outcome") != "aborted":
            warnings.append("predict_log.jsonl: no run_end event")

    # For clean pre-run-start aborts (outcome=aborted AND no run_start event
    # was ever emitted), demote 'empty:' file errors to warnings — the bags
    # had nothing useful to record. Real aborts mid-run still error if their
    # files are empty.
    if (meta.get("outcome") == "aborted"
            and (not events or "run_start" not in events)):
        demote = {"empty: predict_log.jsonl", "empty: orchestrator.mcap"}
        keep, demoted = [], []
        for e in errors:
            (demoted if e in demote else keep).append(e)
        if demoted:
            errors = keep
            warnings.extend(d.replace("empty:", "empty (pre-run abort):") for d in demoted)

    return (len(errors) == 0), errors, warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    ok, errors, warnings = validate(args.run_dir)
    for w in warnings:
        print(f"  warn: {w}", file=sys.stderr)
    for e in errors:
        print(f"  ERROR: {e}", file=sys.stderr)
    print(("OK" if ok else "FAIL") + f"  {args.run_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
