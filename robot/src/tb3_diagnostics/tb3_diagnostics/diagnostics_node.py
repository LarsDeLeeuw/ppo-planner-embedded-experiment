"""Per-host ambient diagnostics for the power experiment.

Publishes:
  /diagnostics/host     (tb3_interfaces/HostDiagnostics, 1 Hz)
  /diagnostics/session  (tb3_interfaces/SessionMetadata, latched once at startup)

One instance runs per ROS host (Pi and, in centralized mode, the desktop VM).
The node auto-detects what's available rather than needing host-specific
config: chrony is always probed, WiFi RSSI is read if the interface exists,
RAPL passthrough is reported if the sysfs counter is readable. This moves
chrony/WiFi/RAPL/git/experiment.yaml state into the bag (instead of
SSH-collected metadata), so analysis reads it from bag content within the run
window.
"""

from __future__ import annotations

import os
import re
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

from tb3_interfaces.msg import HostDiagnostics, SessionMetadata

_RAPL_PKG = "/sys/class/powercap/intel-rapl:0/energy_uj"


class DiagnosticsNode(Node):

    def __init__(self) -> None:
        super().__init__("diagnostics_node")

        # Resolution order for research_project_path (first match wins):
        #   1. ROS parameter (set explicitly in a launch file or with --ros-args -p)
        #   2. RESEARCH_PROJECT_PATH env var (set per-host in /etc/environment)
        #   3. Hardcoded fallback ~/ppo-tb3-research-project/robot (matches ssh_deploy.sh default).
        # On the VM the repo lives at ~/ppo-tb3-research-project/robot, so
        # set RESEARCH_PROJECT_PATH there to override — or pass the launch param.
        default_research_path = os.environ.get(
            "RESEARCH_PROJECT_PATH",
            os.path.expanduser("~/ppo-tb3-research-project/robot"),
        )
        self.declare_parameter("rate_hz", 1.0)
        self.declare_parameter("wifi_interface", "wlan0")
        self.declare_parameter("research_project_path", default_research_path)
        self.declare_parameter("experiment_yaml_path", "")  # empty => resolve from tb3_bringup share

        rate = self.get_parameter("rate_hz").value
        self._wifi_iface = self.get_parameter("wifi_interface").value
        self._hostname = os.uname().nodename

        # -- Live host diagnostics (1 Hz) -------------------------------------
        self._host_pub = self.create_publisher(HostDiagnostics, "diagnostics/host", 10)
        self.create_timer(1.0 / rate, self._host_cb)

        # -- Latched session metadata (publish once, keep for late subscribers)
        latching_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._session_pub = self.create_publisher(
            SessionMetadata, "diagnostics/session", latching_qos)
        self._publish_session_metadata()

        self.get_logger().info(
            f"DiagnosticsNode ready on '{self._hostname}' "
            f"(wifi_iface={self._wifi_iface}, rate={rate} Hz)"
        )

    # -- live host diagnostics ------------------------------------------------

    def _host_cb(self) -> None:
        msg = HostDiagnostics()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.hostname = self._hostname

        offset_ms, rms_ms = self._read_chrony()
        msg.chrony_offset_ms = offset_ms
        msg.chrony_rms_ms = rms_ms

        rssi, quality = self._read_wifi()
        msg.wifi_rssi_dbm = rssi
        msg.wifi_link_quality = quality

        msg.rapl_passthrough_ok = self._rapl_ok()
        self._host_pub.publish(msg)

    # -- session metadata (latched) -------------------------------------------

    def _publish_session_metadata(self) -> None:
        msg = SessionMetadata()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.hostname = self._hostname
        msg.git_sha_research_project = self._git_sha()
        msg.experiment_yaml_snapshot = self._experiment_yaml()
        self._session_pub.publish(msg)
        self.get_logger().info(
            f"Latched session metadata: git={msg.git_sha_research_project[:12]} "
            f"yaml={'present' if msg.experiment_yaml_snapshot else 'missing'}"
        )

    # -- probes ---------------------------------------------------------------

    def _read_chrony(self) -> tuple[float, float]:
        """Return (last_offset_ms, rms_offset_ms); (nan, nan) on failure."""
        try:
            out = subprocess.run(
                ["chronyc", "tracking"], capture_output=True, text=True, timeout=2.0
            ).stdout
        except Exception:  # noqa: BLE001
            return float("nan"), float("nan")

        def _grab(label: str) -> float:
            m = re.search(rf"{label}\s*:\s*([-+]?[0-9.eE]+)\s*seconds", out)
            return float(m.group(1)) * 1000.0 if m else float("nan")

        return _grab("Last offset"), _grab("RMS offset")

    def _read_wifi(self) -> tuple[int, float]:
        """Return (signal_dbm, link_quality_fraction); (0, 0.0) if no WiFi."""
        try:
            out = subprocess.run(
                ["iwconfig", self._wifi_iface], capture_output=True, text=True, timeout=2.0
            ).stdout
        except Exception:  # noqa: BLE001
            return 0, 0.0

        rssi = 0
        quality = 0.0
        m = re.search(r"Signal level[=:]\s*(-?\d+)\s*dBm", out)
        if m:
            rssi = int(m.group(1))
        m = re.search(r"Link Quality[=:]\s*(\d+)/(\d+)", out)
        if m and int(m.group(2)) != 0:
            quality = int(m.group(1)) / int(m.group(2))
        return rssi, quality

    @staticmethod
    def _rapl_ok() -> bool:
        try:
            with open(_RAPL_PKG) as f:
                int(f.read().strip())
            return True
        except Exception:  # noqa: BLE001
            return False

    def _git_sha(self) -> str:
        path = self.get_parameter("research_project_path").value
        # Robot deploys don't include .git (ssh_deploy.sh rsyncs src/ and
        # scripts/ only), so prefer a .git_sha sidecar that ssh_deploy.sh
        # stamps with the build-host's commit. Falls back to live git for
        # VM/dev runs where the repo is fully present.
        sidecar = os.path.join(path, ".git_sha")
        try:
            with open(sidecar) as f:
                sha = f.read().strip()
                if sha:
                    return sha
        except OSError:
            pass
        try:
            return subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=2.0, check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    def _experiment_yaml(self) -> str:
        path = self.get_parameter("experiment_yaml_path").value
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory
                path = os.path.join(
                    get_package_share_directory("tb3_bringup"), "config", "experiment.yaml")
            except Exception:  # noqa: BLE001
                return ""
        try:
            with open(path) as f:
                return f.read()
        except Exception:  # noqa: BLE001
            return ""


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
