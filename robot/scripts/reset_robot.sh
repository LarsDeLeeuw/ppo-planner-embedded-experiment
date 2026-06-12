#!/usr/bin/env bash
# =============================================================================
# reset_robot.sh
#
# INTENTION:
#   Emergency stop and soft reset for the TurtleBot3. Publishes a zero Twist
#   message to /cmd_vel several times to halt any ongoing movement, then
#   optionally kills any running ros2 processes on the VM.
#
#   Use this when:
#     - The control node crashes mid-move and the robot keeps moving
#     - You want to stop the robot without SSHing in manually
#     - You're about to relaunch the node and want a clean slate
#
#   By default the script runs locally (assumes you're already on the VM or
#   that your ROS_DOMAIN_ID is shared with the robot). Set REMOTE=1 to have
#   it SSH into the VM and run there instead.
#
#   The stop command is published STOP_TIMES times with a short delay between
#   each to maximise the chance of the robot receiving it even under load.
#
# CONFIGURATION (env vars):
#   REMOTE       Set to 1 to run the stop command over SSH    (default: 0)
#   ROBOT_HOST   SSH target (used only if REMOTE=1)           (default: ubuntu@turtlebot3.local)
#   STOP_TIMES   How many zero-Twist messages to publish      (default: 5)
#   KILL_NODES   Set to 1 to also pkill ros2 processes        (default: 0)
#   WS           ROS 2 workspace to source                    (default: ~/turtlebot3_ws)
# =============================================================================
set -euo pipefail

REMOTE="${REMOTE:-0}"
ROBOT_HOST="${ROBOT_HOST:-ubuntu@turtlebot3.local}"
STOP_TIMES="${STOP_TIMES:-5}"
KILL_NODES="${KILL_NODES:-0}"
WS="${WS:-$HOME/turtlebot3_ws}"

_stop_cmd() {
  set +u
  source /opt/ros/humble/setup.bash
  [[ -f "$WS/install/setup.bash" ]] && source "$WS/install/setup.bash"
  set -u

  echo "Publishing zero Twist to /cmd_vel ($STOP_TIMES times)..."
  ros2 topic pub --times "$STOP_TIMES" /cmd_vel geometry_msgs/msg/Twist '{}'

  if [[ "${KILL_NODES:-0}" -eq 1 ]]; then
    echo "Killing ros2 run processes..."
    pkill -f "ros2 run" || echo "  [no ros2 run processes found]"
  fi

  echo "Robot stopped."
}

if [[ "$REMOTE" -eq 1 ]]; then
  echo "Sending stop command to $ROBOT_HOST..."
  ssh "$ROBOT_HOST" "
    STOP_TIMES=$STOP_TIMES
    KILL_NODES=$KILL_NODES
    WS=$WS
    $(declare -f _stop_cmd)
    _stop_cmd
  "
else
  _stop_cmd
fi
