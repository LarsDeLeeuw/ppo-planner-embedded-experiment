#!/usr/bin/env bash
# =============================================================================
# 02_check_sensor_topics.sh — Verify rates + sanity of the always-on layer.
#
# Run this in a second SSH session WHILE 01_robot_sensors.sh is running.
#
# Implementation note: this used to call `ros2 topic hz` and `ros2 topic
# echo --once` once per topic, which spawned a fresh rclpy node each time
# and re-ran DDS discovery against the long-running power_sensor_node /
# diagnostics_node. That race produced flaky FAILs (different topics
# failed on different runs) even though every publisher was healthy at
# 100 Hz. The replacement is a single Python process that subscribes to
# all topics, runs for one observation window, and reports rates / alive
# / latched in one pass — one discovery cycle, deterministic outcome.
#
# Pass criteria:
#   /power/{solar,sbc,opencr}  : measured rate >= 85 Hz (nominal 100)
#   /sbc/thermal               : >= 1 sample arrived during the window
#   /diagnostics/host          : >= 1 sample arrived during the window
#   /diagnostics/session       : latched (TRANSIENT_LOCAL) sample received
#
# ENV:
#   WS         colcon overlay              (default: ~/turtlebot3_ws)
#   WINDOW_S   observation window seconds  (default: 6)
# =============================================================================
set -uo pipefail

WS="${WS:-$HOME/turtlebot3_ws}"
WINDOW_S="${WINDOW_S:-6}"

set +u
source /opt/ros/humble/setup.bash
[[ -f "$WS/install/setup.bash" ]] && source "$WS/install/setup.bash"
set -u

# Heredoc Python — single quoting on PYEOF prevents bash from interpolating.
# WINDOW_S is forwarded via env so the Python side sees one value.
WINDOW_S="$WINDOW_S" python3 - <<'PYEOF'
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

try:
    from tb3_interfaces.msg import (
        HostDiagnostics, PowerSample, SessionMetadata, ThermalStat,
    )
except ImportError as exc:
    print(f"FAIL: tb3_interfaces import failed ({exc}). "
          "Did you source the overlay?", file=sys.stderr)
    sys.exit(2)

WINDOW_S = float(os.environ.get("WINDOW_S", "6"))

LATCHED_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)

# (topic, type, role, min_rate_hz, qos)
TOPICS = [
    ("/power/solar",         PowerSample,     "rate",    85.0, 10),
    ("/power/sbc",           PowerSample,     "rate",    85.0, 10),
    ("/power/opencr",        PowerSample,     "rate",    85.0, 10),
    ("/sbc/thermal",         ThermalStat,     "alive",   None, 10),
    ("/diagnostics/host",    HostDiagnostics, "alive",   None, 10),
    ("/diagnostics/session", SessionMetadata, "latched", None, LATCHED_QOS),
]


class Checker(Node):
    def __init__(self):
        super().__init__("smoke_check_topics")
        self.first = {}
        self.last_t = {}
        self.count = {}
        self.last_msg = {}
        for topic, msg_type, _, _, qos in TOPICS:
            # default-arg trick to capture topic by value, not by closure
            self.create_subscription(
                msg_type, topic,
                lambda msg, t=topic: self._on_msg(t, msg),
                qos,
            )

    def _on_msg(self, topic, msg):
        now = time.monotonic()
        if topic not in self.first:
            self.first[topic] = now
        self.last_t[topic] = now
        self.count[topic] = self.count.get(topic, 0) + 1
        self.last_msg[topic] = msg


def main():
    rclpy.init()
    node = Checker()
    deadline = time.monotonic() + WINDOW_S
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    failed = 0
    print(f"=== topic checks (window {WINDOW_S:.1f}s, single discovery cycle) ===")
    for topic, _, role, min_hz, _ in TOPICS:
        n = node.count.get(topic, 0)
        if role == "rate":
            if n < 2:
                print(f"  [FAIL] {topic:25s} no samples in {WINDOW_S:.1f}s")
                failed += 1
                continue
            duration = node.last_t[topic] - node.first[topic]
            rate = (n - 1) / duration if duration > 0 else 0.0
            tag = "PASS" if rate >= min_hz else "FAIL"
            print(f"  [{tag}] {topic:25s} {rate:6.2f} Hz "
                  f"(n={n}, min {min_hz:.2f})")
            if tag == "FAIL":
                failed += 1
        else:
            tag = "PASS" if n >= 1 else "FAIL"
            label = "latched" if role == "latched" else "alive"
            print(f"  [{tag}] {topic:25s} {label} (n={n})")
            if tag == "FAIL":
                failed += 1

    sess = node.last_msg.get("/diagnostics/session")
    if sess is not None:
        sha = sess.git_sha_research_project[:12] or "?"
        yaml_state = "present" if sess.experiment_yaml_snapshot else "empty"
        print()
        print("=== /diagnostics/session ===")
        print(f"  git_sha={sha}  yaml={yaml_state}")

    print()
    print("=== power-rail snapshot (operator must check signs) ===")
    for rail in ("sbc", "opencr", "solar"):
        topic = f"/power/{rail}"
        msg = node.last_msg.get(topic)
        if msg is None:
            print(f"  [FAIL] {topic} no sample")
            failed += 1
            continue
        print(f"  /power/{rail:<7s} addr={msg.i2c_address} "
              f"bus={msg.bus_voltage_v:6.3f}V "
              f"current={msg.current_ma:8.2f}mA "
              f"cal={msg.current_calibration:.4f} "
              f"overflow={msg.overflow}")
    print()
    print("  Expected signs at rest:")
    print("    sbc      current_ma > 0   (battery powers Pi)")
    print("    opencr   current_ma > 0   (battery powers OpenCR; spikes on motion)")
    print("    solar    panel illuminated -> current_ma > 0; covered -> ~0 or slightly negative")
    print("    overflow must be False on all three. If True, drop PGA to 4_160mv.")

    node.destroy_node()
    rclpy.shutdown()

    print()
    if failed == 0:
        print("[02] all auto-checks PASSED")
        return 0
    print(f"[02] {failed} auto-check(s) FAILED")
    return 1


sys.exit(main())
PYEOF
