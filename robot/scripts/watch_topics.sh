#!/usr/bin/env bash
# =============================================================================
# watch_topics.sh
#
# INTENTION:
#   Provide a single-command diagnostic snapshot of all ROS 2 topics relevant
#   to TurtleBot3 grid navigation. Instead of opening multiple terminals and
#   running ros2 topic echo manually, this script captures one message from
#   each key topic and prints them in sequence.
#
#   Topics covered:
#     /imu      — orientation (quaternion). Used to verify robot heading before
#                 issuing grid moves. The cardinal direction quaternion values
#                 are documented in context/turtlebot3.md for reference.
#     /cmd_vel  — current velocity command being sent to the robot. Useful to
#                 confirm the control node is publishing and what it's sending.
#     /odom     — odometry estimate from wheel encoders. Shows accumulated
#                 position/velocity drift over time.
#
#   Run this on the VM (where ROS 2 is running), not on the dev machine.
#
# CONFIGURATION (env vars):
#   TIMEOUT    Seconds to wait for a message per topic before skipping  (default: 3)
#   WS         ROS 2 workspace to source                                (default: ~/turtlebot3_ws)
# =============================================================================
set -euo pipefail

WS="${WS:-$HOME/turtlebot3_ws}"
TIMEOUT="${TIMEOUT:-3}"

# ---- helpers ----
topic_snapshot() {
  local topic="$1"
  echo ""
  echo "──────────────────────────────────────"
  echo "  $topic"
  echo "──────────────────────────────────────"
  if ! ros2 topic echo "$topic" --once --timeout "$TIMEOUT" 2>/dev/null; then
    echo "  [no message received within ${TIMEOUT}s — is the node running?]"
  fi
}

set +u
source /opt/ros/humble/setup.bash
[[ -f "$WS/install/setup.bash" ]] && source "$WS/install/setup.bash"
set -u

echo "TurtleBot3 topic snapshot — $(date)"

topic_snapshot /imu
topic_snapshot /cmd_vel
topic_snapshot /odom

echo ""
echo "Active nodes:"
ros2 node list 2>/dev/null || echo "  [none found]"
