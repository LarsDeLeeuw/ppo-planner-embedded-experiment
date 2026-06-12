"""run_io.py - Load a single run folder into pandas DataFrames.

Pure reader. Treats `runs/<run_id>/` as read-only. No ROS2 runtime required:
bags are decoded with mcap-ros2-support using schemas embedded in the .mcap.

Importable for ad-hoc Jupyter exploration:

    from run_io import load_run
    run = load_run("runs/20260527-150000-ppo-map3-decentralized")
    run.streams["robot"]["/power/sbc"]          # a DataFrame
    run.window_ns                               # (t0, t1) run window in ns
    run.predict_log                             # JSONL sidecar DataFrame

All per-message DataFrames carry:
  _log_time_ns, _publish_time_ns   bag record times (recording-host clock)
  _stamp_ns                        header.stamp in ns, if the message has a header
plus the message fields flattened with dotted keys (e.g. header.frame_id).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

BAG_NAMES = ("robot", "desktop", "orchestrator")
EVENTS_TOPIC = "/experiment/events"


# -- message flattening -------------------------------------------------------

def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a decoded ROS2 message into dotted-key scalars/lists.

    Nested messages (header -> stamp) recurse; arrays are kept as-is.
    """
    slots = getattr(obj, "__slots__", None)
    if not slots:
        return {prefix.rstrip("."): obj}
    out: dict[str, Any] = {}
    for s in slots:
        val = getattr(obj, s)
        key = f"{prefix}{s}"
        if getattr(val, "__slots__", None):
            out.update(_flatten(val, key + "."))
        else:
            out[key] = val
    return out


def _stamp_ns(flat: dict[str, Any]) -> int | None:
    if "header.stamp.sec" in flat and "header.stamp.nanosec" in flat:
        return int(flat["header.stamp.sec"]) * 1_000_000_000 + int(
            flat["header.stamp.nanosec"],
        )
    return None


def collect_topics(bag_path: Path) -> dict[str, pd.DataFrame]:
    """Decode an .mcap into {topic: DataFrame}, one row per message.

    Tolerates channels whose schema is unrecognised (encoding != ros2msg,
    empty schema data, etc.) — these get skipped with a stderr warning
    rather than killing the whole bag read. rosbag2 occasionally records
    action-derived feedback types this way and the old `read_ros2_messages`
    path raised DecoderNotFoundError on the first such message.
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    factory = DecoderFactory()
    rows: dict[str, list[dict]] = defaultdict(list)
    decoders: dict[int, Callable[[bytes], Any] | None] = {}
    skipped: dict[str, str] = {}

    with open(bag_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if channel.topic in skipped:
                continue
            sid = schema.id if schema is not None else -1
            if sid not in decoders:
                try:
                    decoders[sid] = factory.decoder_for(channel.message_encoding, schema)
                except Exception as e:
                    decoders[sid] = None
                    skipped[channel.topic] = f"decoder build failed: {e}"
                    continue
            dec = decoders[sid]
            if dec is None:
                sname = schema.name if schema is not None else "?"
                senc = schema.encoding if schema is not None else "?"
                skipped[channel.topic] = (
                    f"no decoder (message_encoding={channel.message_encoding}, "
                    f"schema={sname!r}, schema_encoding={senc!r})"
                )
                continue
            try:
                ros_msg = dec(message.data)
            except Exception as e:
                skipped[channel.topic] = f"decode error: {e}"
                continue
            flat = _flatten(ros_msg)
            row: dict[str, Any] = {
                "_log_time_ns": int(message.log_time),
                "_publish_time_ns": int(message.publish_time),
            }
            sn = _stamp_ns(flat)
            if sn is not None:
                row["_stamp_ns"] = sn
            row.update(flat)
            rows[channel.topic].append(row)

    if skipped:
        print(f"[run_io] {bag_path.name}: skipped {len(skipped)} undecodable topic(s):",
              file=sys.stderr)
        for topic, reason in sorted(skipped.items()):
            print(f"  - {topic}: {reason}", file=sys.stderr)

    return {t: pd.DataFrame(r) for t, r in rows.items()}


def best_time_col(df: pd.DataFrame) -> str:
    """Prefer header stamp; fall back to publish time for header-less msgs."""
    if "_stamp_ns" in df.columns and df["_stamp_ns"].notna().any():
        return "_stamp_ns"
    return "_publish_time_ns"


# -- run container ------------------------------------------------------------

@dataclass
class Run:
    run_dir: Path
    meta: dict[str, Any]
    streams: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    predict_log: pd.DataFrame | None = None
    pose_log: pd.DataFrame | None = None  # 30 Hz laptop tracker pose (bridge frame)
    obstacle_map: np.ndarray | None = None
    window_ns: tuple[int, int] | None = None
    window_source: str = ""  # which bag the run_start/run_end events came from

    @property
    def run_id(self) -> str:
        return self.meta.get("run_id", self.run_dir.name)

    def topic(self, name: str) -> pd.DataFrame | None:
        """First DataFrame for `name` across bags (most topics live in one bag)."""
        for bag in self.streams.values():
            if name in bag:
                return bag[name]
        return None

    def window_mask(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask of rows inside the run window using the best time col."""
        if self.window_ns is None or df is None or df.empty:
            return pd.Series([True] * (0 if df is None else len(df)))
        t0, t1 = self.window_ns
        col = best_time_col(df)
        return (df[col] >= t0) & (df[col] <= t1)


def _extract_window(streams: dict[str, dict[str, pd.DataFrame]],
                    expected_run_id: str | None = None) -> tuple[tuple[int, int] | None, str]:
    """Find run_start/run_end on /experiment/events.

    Prefers a ROS-host bag (robot/desktop) since those events carry the
    bridge-host header.stamp that bounds nav/power data. Falls back to the
    orchestrator copy. Returns ((t0, t1), source_bag) or (None, "").

    The events topic is latched, so a bag recorded mid-session replays every
    earlier run's events (original stamps) at subscription time. Events are
    therefore filtered to `expected_run_id` when possible; otherwise the
    window is the LAST run_start and the first run_end at/after it.
    """
    for bag_name in ("robot", "desktop", "orchestrator"):
        bag = streams.get(bag_name)
        if not bag or EVENTS_TOPIC not in bag:
            continue
        ev = bag[EVENTS_TOPIC]
        col = best_time_col(ev)
        if (expected_run_id and "run_id" in ev.columns
                and (ev["run_id"] == expected_run_id).any()):
            ev = ev[ev["run_id"] == expected_run_id]
        starts = ev.loc[ev["event_type"] == "run_start", col]
        ends = ev.loc[ev["event_type"] == "run_end", col]
        if starts.empty:
            continue
        t0 = int(starts.max())
        ends_after = ends[ends >= t0]
        if not ends_after.empty:
            return (t0, int(ends_after.min())), bag_name
        # No run_end (aborted/truncated) — bound by last message time.
        t1 = max(int(df[best_time_col(df)].max()) for df in bag.values()
                 if not df.empty)
        return (t0, t1), bag_name
    return None, ""


def load_run(run_dir: Path | str) -> Run:
    """Load everything in runs/<run_id>/ into a Run."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}

    streams: dict[str, dict[str, pd.DataFrame]] = {}
    for bag_name in BAG_NAMES:
        bag_path = run_dir / f"{bag_name}.mcap"
        if bag_path.exists():
            streams[bag_name] = collect_topics(bag_path)

    predict_log = None
    jsonl = run_dir / "predict_log.jsonl"
    if jsonl.exists():
        records = [json.loads(line) for line in jsonl.read_text().splitlines()
                   if line.strip()]
        if records:
            predict_log = pd.DataFrame(records)

    pose_log = None
    pose_path = run_dir / "pose_log.jsonl"
    if pose_path.exists():
        recs = [json.loads(line) for line in pose_path.read_text().splitlines()
                if line.strip()]
        if recs:
            pose_log = pd.DataFrame(recs)

    obstacle_map = None
    obs_path = run_dir / "maps" / "obstacle.npy"
    if obs_path.exists():
        obstacle_map = np.load(obs_path)

    window_ns, window_source = _extract_window(streams, meta.get("run_id"))

    return Run(
        run_dir=run_dir,
        meta=meta,
        streams=streams,
        predict_log=predict_log,
        pose_log=pose_log,
        obstacle_map=obstacle_map,
        window_ns=window_ns,
        window_source=window_source,
    )
