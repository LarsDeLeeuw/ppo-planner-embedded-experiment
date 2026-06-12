"""orchestrator_bag.py - Write orchestrator.mcap without a ROS2 runtime.

The orchestrator (this repo, running on a no-ROS Windows laptop) records two
ROS2-shaped streams per run so the analysis pipeline can read them with
mcap-ros2-support alongside the robot/desktop bags:

  /predict_tcp_timing   tb3_interfaces/msg/PredictTcpTiming   (per predict round-trip)
  /experiment/events    tb3_interfaces/msg/ExperimentEvent    (local copy of run markers)

Serialization is done by `rosbags` (pure-Python CDR + mcap), with the message
schemas read from the real .msg files in ../tb3_interfaces so there is a single
source of truth.  rosbags writes a rosbag2 *directory* (an inner .mcap plus a
metadata.yaml); on close we relocate the inner .mcap to a flat
`<run_dir>/orchestrator.mcap` to match the analysis folder layout.

Everything here is best-effort and isolated: a failure to write the bag must
never take down the live experiment.  Callers should treat OrchestratorBag as
an optional sink.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Topics + message types this writer emits.
_TOPIC_TIMING = "/predict_tcp_timing"
_TYPE_TIMING = "tb3_interfaces/msg/PredictTcpTiming"
_TOPIC_EVENTS = "/experiment/events"
_TYPE_EVENTS = "tb3_interfaces/msg/ExperimentEvent"

# .msg files needed (their Header/Time deps come from the Humble typestore).
_MSG_FILES = {
    _TYPE_TIMING: "PredictTcpTiming",
    _TYPE_EVENTS: "ExperimentEvent",
}


def _default_interfaces_dir() -> Path:
    """Locate tb3_interfaces/msg by walking up from this file.

    tb3_interfaces is the ROS2 message package; in this mono-repo its canonical
    copy lives at robot/src/tb3_interfaces/.  Walking up and probing a few
    candidate sub-paths keeps this robust to where the process is launched.
    """
    here = Path(__file__).resolve()
    candidates = (
        Path("robot") / "src" / "tb3_interfaces" / "msg",
        Path("src") / "tb3_interfaces" / "msg",
        Path("tb3_interfaces") / "msg",
    )
    for parent in here.parents:
        for rel in candidates:
            cand = parent / rel
            if cand.is_dir():
                return cand
    raise FileNotFoundError(
        "could not locate tb3_interfaces/msg (looked upward from "
        f"{here}); pass interfaces_dir explicitly",
    )


class OrchestratorBag:
    """Append-only writer for the laptop's orchestrator.mcap.

    Usage:
        bag = OrchestratorBag(run_dir / "orchestrator.mcap")
        bag.open()
        bag.write_predict_timing(seq, send_ns, recv_ns)
        bag.write_experiment_event("run_start", run_id, payload_json)
        bag.close()

    Thread-safety: rosbags' Writer is not re-entrant, so all writes funnel
    through a lock.  Timestamps on the bag records use the laptop's wall clock
    in nanoseconds (`time.time_ns()`) purely as the mcap log-time; the
    analysis-relevant timing lives inside the message fields (monotonic ns for
    PredictTcpTiming, bridge-host stamp for events once forwarded by the
    bridge).
    """

    def __init__(
        self,
        mcap_path: Path | str,
        *,
        interfaces_dir: Path | str | None = None,
        frame_id: str = "laptop",
    ) -> None:
        self._mcap_path = Path(mcap_path)
        self._frame_id = frame_id
        self._interfaces_dir = (
            Path(interfaces_dir) if interfaces_dir is not None
            else _default_interfaces_dir()
        )

        # Lazily imported so tracker still starts if rosbags is absent;
        # the bag just stays disabled in that case.
        self._ts = None
        self._writer = None
        self._bag_dir: Path | None = None
        self._conns: dict[str, object] = {}
        self._types: dict[str, type] = {}
        self._enabled = False

        import threading
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------------

    def open(self) -> bool:
        """Create the underlying rosbag2 mcap. Returns True on success.

        On any failure the bag is left disabled and all writes become no-ops;
        the experiment continues.
        """
        try:
            from rosbags.rosbag2 import StoragePlugin, Writer
            from rosbags.typesys import Stores, get_types_from_msg, get_typestore

            ts = get_typestore(Stores.ROS2_HUMBLE)
            for typename, stem in _MSG_FILES.items():
                if typename in ts.types:
                    continue
                msg_text = (self._interfaces_dir / f"{stem}.msg").read_text()
                ts.register(get_types_from_msg(msg_text, typename))
            self._ts = ts

            # rosbags writes a bag *directory*; use a temp sibling we relocate
            # the inner .mcap out of on close.
            self._mcap_path.parent.mkdir(parents=True, exist_ok=True)
            self._bag_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{self._mcap_path.stem}_", dir=self._mcap_path.parent,
                ),
            ) / "bag"

            self._writer = Writer(
                self._bag_dir, version=9, storage_plugin=StoragePlugin.MCAP,
            )
            self._writer.open()
            for topic, typename in (
                (_TOPIC_TIMING, _TYPE_TIMING),
                (_TOPIC_EVENTS, _TYPE_EVENTS),
            ):
                self._conns[topic] = self._writer.add_connection(
                    topic, typename, typestore=ts,
                )
                self._types[typename] = ts.types[typename]

            self._enabled = True
            logger.info("[orch-bag] writing %s", self._mcap_path)
            return True
        except Exception:
            logger.exception("[orch-bag] open failed; bag disabled")
            self._enabled = False
            return False

    def close(self) -> None:
        """Flush, close, and relocate the inner .mcap to the flat path."""
        with self._lock:
            if self._writer is None:
                return
            # Disable BEFORE closing so any concurrent write_* call from the
            # bridge recv thread (which holds onto a stale `_active.recorder`
            # reference for a few ms during teardown) short-circuits cleanly
            # at the `if not self._enabled` guard instead of dereferencing
            # _writer=None.
            self._enabled = False
            try:
                self._writer.close()
            except Exception:
                logger.exception("[orch-bag] close failed")
            self._writer = None

            if self._bag_dir is None:
                return
            try:
                inner = next(self._bag_dir.glob("*.mcap"))
                if self._mcap_path.exists():
                    self._mcap_path.unlink()
                shutil.move(str(inner), str(self._mcap_path))
            except StopIteration:
                logger.warning("[orch-bag] no inner .mcap produced")
            except Exception:
                logger.exception("[orch-bag] relocating inner mcap failed")
            finally:
                # Drop the temp bag directory (and its metadata.yaml).
                shutil.rmtree(self._bag_dir.parent, ignore_errors=True)
                self._bag_dir = None

    # -- writes ---------------------------------------------------------------

    def write_predict_timing(
        self, sequence: int, t_tcp_send_ns: int, t_tcp_recv_ns: int,
    ) -> None:
        if not self._enabled:
            return
        with self._lock:
            # Re-check inside the lock — close() may have run between the
            # outer guard and this acquire, and the writer may already be None.
            if self._writer is None:
                return
            try:
                Header, Time = self._header_types()
                msg = self._types[_TYPE_TIMING](
                    header=self._make_header(Header, Time),
                    sequence=int(sequence) & 0xFFFFFFFF,
                    t_tcp_send_ns=int(t_tcp_send_ns),
                    t_tcp_recv_ns=int(t_tcp_recv_ns),
                )
                self._writer.write(
                    self._conns[_TOPIC_TIMING],
                    time.time_ns(),
                    self._ts.serialize_cdr(msg, _TYPE_TIMING),
                )
            except Exception:
                logger.exception("[orch-bag] write_predict_timing failed")

    def write_experiment_event(
        self, event_type: str, run_id: str, payload_json: str = "",
    ) -> None:
        if not self._enabled:
            return
        with self._lock:
            # Re-check inside the lock — close() may have run between the
            # outer guard and this acquire, and the writer may already be None.
            if self._writer is None:
                return
            try:
                Header, Time = self._header_types()
                msg = self._types[_TYPE_EVENTS](
                    header=self._make_header(Header, Time),
                    event_type=str(event_type),
                    run_id=str(run_id),
                    payload_json=str(payload_json),
                )
                self._writer.write(
                    self._conns[_TOPIC_EVENTS],
                    time.time_ns(),
                    self._ts.serialize_cdr(msg, _TYPE_EVENTS),
                )
            except Exception:
                logger.exception("[orch-bag] write_experiment_event failed")

    # -- helpers --------------------------------------------------------------

    def _header_types(self) -> tuple[type, type]:
        return (
            self._ts.types["std_msgs/msg/Header"],
            self._ts.types["builtin_interfaces/msg/Time"],
        )

    def _make_header(self, Header: type, Time: type):
        now = time.time_ns()
        return Header(
            stamp=Time(sec=now // 1_000_000_000, nanosec=now % 1_000_000_000),
            frame_id=self._frame_id,
        )
