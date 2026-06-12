"""orchestrator.ssh_bag - SSH + scp wrapper around tb3_bag_record.sh.

Implements the orchestrator side of [handover_orchestrator §4](
../../temp-context/handover_orchestrator_2026-05-24.md), with the rsync ->
scp substitution from the minimap-visualizer handover §7.

Three operations per remote bag:
  - start(): invoke `tb3_bag_record.sh start ...` over ssh; the script daemonises
    `timeout --signal=SIGINT $max ros2 bag record -s mcap`, writes a pid file,
    and returns exit 0 on spawn (NOT on first message received).
  - stop():  invoke `tb3_bag_record.sh stop ...` over ssh; the script signals
    the pid and returns elapsed seconds. On non-zero exit we retry, then fall
    back to `pkill -SIGINT ros2 bag`.
  - pull():  scp the produced .mcap into the run dir locally. Retries 3 times.

When `ssh_mock=True` is set on the BagHost, all three operations are local
no-ops that just touch a sentinel file in remote_tmp so we can exercise the
orchestrator end-to-end on the laptop without a robot.

Failures bubble up as `SshBagError`. The orchestrator catches that, converts
to outcome=aborted, and continues teardown so it never leaves bags running on
remote hosts.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class SshBagError(RuntimeError):
    """Raised when an SSH bag operation fails after all retries."""


@dataclass(frozen=True)
class BagHost:
    """One remote host that runs ros2 bag record locally."""
    name: str                # "pi" or "desktop" — surfaced in error messages
    ssh_target: str          # user@host
    bag_script: str          # absolute path to tb3_bag_record.sh on the remote
    remote_tmp: str = "/tmp/experiments"
    connect_timeout_s: int = 10
    max_retries: int = 3
    mock: bool = False       # ssh_mock — local fake, no real network

    # -- operations ----------------------------------------------------------

    @staticmethod
    def _bag_stem(bag_name: str) -> str:
        """Strip the rosbag2 storage extension from `bag_name` so the script
        sees just the base name. The script appends the extension itself on
        consolidation, producing the on-disk path `<run_id>/<stem>.mcap` —
        which equals our `bag_name` and matches the pull source path.
        """
        for ext in (".mcap", ".db3"):
            if bag_name.endswith(ext):
                return bag_name[: -len(ext)]
        return bag_name

    def start_recording(self, run_id: str, bag_name: str, topics: list[str],
                        max_duration_s: int) -> None:
        """Spawn ros2 bag record on the remote host. Raises on failure."""
        if self.mock:
            self._mock_touch(run_id, bag_name, "started")
            return
        topic_arg = " ".join(shlex.quote(t) for t in topics)
        cmd = [
            self.bag_script, "start",
            "--run-id", run_id,
            "--bag-name", self._bag_stem(bag_name),
            "--topics", topic_arg,
            "--max-duration-s", str(max_duration_s),
        ]
        self._ssh(cmd, what=f"{self.name}: start({bag_name})")

    def stop_recording(self, run_id: str, bag_name: str) -> float:
        """SIGINT the bag and return elapsed seconds. Falls back to pkill.

        Parses the script's `key=value` output (elapsed_s, capped, mcap_path)
        — the last line is `mcap_path=...`, not a number, so the old "parse
        last line as float" logic silently always returned -1.0. Returns
        elapsed wall-clock seconds the bag was running; -1.0 on failure.
        """
        if self.mock:
            self._mock_touch(run_id, bag_name, "stopped")
            return 0.0
        cmd = [self.bag_script, "stop", "--run-id", run_id,
               "--bag-name", self._bag_stem(bag_name)]
        try:
            out = self._ssh(cmd, what=f"{self.name}: stop({bag_name})")
            kv: dict[str, str] = {}
            for line in out.splitlines():
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip()
            try:
                return float(kv.get("elapsed_s", "-1"))
            except ValueError:
                return -1.0
        except SshBagError:
            # Last-ditch fallback: pkill the recorder. The bag may end up
            # truncated; that's the orchestrator's signal to mark the run
            # aborted post-rsync.
            logger.warning("[%s] stop failed; trying pkill -SIGINT ros2 bag",
                           self.name)
            try:
                self._ssh_raw(f"pkill -SIGINT -f 'ros2 bag record' || true",
                              what=f"{self.name}: pkill fallback")
            except SshBagError:
                pass
            return -1.0

    def pull(self, run_id: str, bag_name: str, dest_dir: Path) -> Path:
        """scp the consolidated bag file into dest_dir.

        After tb3_bag_record.sh's `stop` runs, the bag is a single flat file
        at `<remote_tmp>/<run_id>/<stem>.<ext>` (the script removed the
        rosbag2 directory wrapper). With our convention `bag_name = stem +
        ".mcap"`, that file's remote path equals
        `<remote_tmp>/<run_id>/<bag_name>` — a plain `scp` (no `-r`) is
        enough. cwd-based destination avoids Windows OpenSSH scp confusing
        a drive letter for a host designator.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote_path = f"{self.remote_tmp}/{run_id}/{bag_name}"
        local_path = dest_dir / bag_name
        if self.mock:
            local_path.write_bytes(b"")
            return local_path

        last_err: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                subprocess.run(
                    ["scp", "-q",
                     "-o", f"ConnectTimeout={self.connect_timeout_s}",
                     "-o", "BatchMode=yes",
                     "-o", "StrictHostKeyChecking=accept-new",
                     f"{self.ssh_target}:{remote_path}",
                     bag_name],
                    check=True, capture_output=True, timeout=120,
                    cwd=str(dest_dir),
                )
                if local_path.exists() and local_path.stat().st_size > 0:
                    return local_path
                last_err = f"empty after pull (attempt {attempt})"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                stderr = getattr(e, "stderr", b"") or b""
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                last_err = f"scp failed (attempt {attempt}): {e}"
                if stderr.strip():
                    last_err = f"{last_err}  | stderr: {stderr.strip()}"
                logger.warning("[%s] %s", self.name, last_err)
            time.sleep(1.5 * attempt)
        raise SshBagError(f"{self.name}: pull failed: {last_err}")

    def cleanup_remote(self, run_id: str) -> None:
        """Best-effort: rm -rf the remote run dir after pull. Never raises."""
        if self.mock:
            return
        try:
            self._ssh_raw(f"rm -rf {shlex.quote(self.remote_tmp + '/' + run_id)}",
                          what=f"{self.name}: cleanup")
        except SshBagError:
            pass

    # -- internals -----------------------------------------------------------

    def _ssh(self, argv: list[str], *, what: str) -> str:
        return self._ssh_raw(" ".join(shlex.quote(a) for a in argv), what=what)

    def _ssh_raw(self, remote_cmd: str, *, what: str) -> str:
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = subprocess.run(
                    ["ssh",
                     "-o", f"ConnectTimeout={self.connect_timeout_s}",
                     "-o", "BatchMode=yes",
                     "-o", "StrictHostKeyChecking=accept-new",
                     self.ssh_target, remote_cmd],
                    check=True, capture_output=True, text=True, timeout=60,
                )
                if r.stderr.strip():
                    logger.debug("[%s] stderr: %s", self.name, r.stderr.strip())
                return r.stdout
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
                stderr = getattr(e, "stderr", b"") or b""
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                last_err = f"{what} attempt {attempt}: {e}"
                if stderr.strip():
                    last_err = f"{last_err}  | stderr: {stderr.strip()}"
                logger.warning("[%s] %s", self.name, last_err)
            time.sleep(1.0 * attempt)
        raise SshBagError(last_err or f"{what}: ssh failed")

    def _mock_touch(self, run_id: str, bag_name: str, marker: str) -> None:
        d = Path(self.remote_tmp) / run_id
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{bag_name}.{marker}").write_text(str(time.time()))
        except OSError:
            pass


# -- preflight ----------------------------------------------------------------

def ssh_available() -> bool:
    """Return True iff `ssh` and `scp` are on PATH (sanity check at startup)."""
    return bool(shutil.which("ssh")) and bool(shutil.which("scp"))
