#!/usr/bin/env bash
# =============================================================================
# 05_record_run.sh — Record a one-shot bag and verify the experiment topics.
#
# Wraps scripts/tb3_bag_record.sh (start/stop) around a single invocation of
# 04_drive_bridge.py so the bag captures a complete predict + goal + events
# cycle. Prints `ros2 bag info` plus a per-topic message-count summary.
#
# Run this on the Pi (decentralized mode) — it touches the bag-record helper
# directly (no SSH). For an actual orchestrator-driven run the helper would
# be invoked over SSH from minimap-visualizer.
#
# Pre-reqs (already running, in other terminals):
#   - scripts/smoke/01_robot_sensors.sh
#   - scripts/smoke/03_compute_stack.sh
#
# ENV / FLAGS:
#   ROBOT_IP        bridge host for 04_drive_bridge.py    (default: 127.0.0.1)
#   RUN_ID          unique run id                          (default: smoke-<epoch>)
#   NO_GOAL         set to 1 to skip the move_to_grid goal (default: 0)
#   WS              colcon overlay                         (default: ~/turtlebot3_ws)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ROBOT_IP="${ROBOT_IP:-127.0.0.1}"
RUN_ID="${RUN_ID:-smoke-$(date +%s)}"
NO_GOAL="${NO_GOAL:-0}"
WS="${WS:-$HOME/turtlebot3_ws}"

BAG_NAME="robot"

# Topic table from plan §3 (robot SBC, both modes + decentralized additions).
# All on one host here, since the smoke test runs entirely on the Pi.
TOPICS="/power/solar /power/sbc /power/opencr /sbc/thermal \
/diagnostics/host /diagnostics/session \
/cmd_vel /odom /imu \
/grid_nav_node/grid_pose /grid_nav_node/loop_stats /grid_nav_node/last_result \
/move_to_grid/_action/feedback /move_to_grid/_action/status \
/planner/metrics /experiment/events /bridge/predict_timing"

set +u
source /opt/ros/humble/setup.bash
[[ -f "$WS/install/setup.bash" ]] && source "$WS/install/setup.bash"
set -u

echo "[05] run_id=$RUN_ID  bridge=$ROBOT_IP:9090"

# ---- start the bag ----------------------------------------------------------
echo "[05/1] starting bag"
WS="$WS" bash "$REPO_DIR/scripts/tb3_bag_record.sh" start \
    --run-id "$RUN_ID" --bag-name "$BAG_NAME" --topics "$TOPICS"

# Subscription-discovery warmup. ros2 bag record needs ~1.5-2 s to discover
# and subscribe to every named topic against long-running publishers — one-
# shot events fired before this window arrives at the bag silently. 3 s is
# conservative and the smoke test isn't latency-sensitive at this stage.
echo "  waiting 3s for bag-record subscriptions to settle..."
sleep 3.0

# ---- drive the bridge -------------------------------------------------------
echo "[05/2] driving bridge"
DRIVE_ARGS=(--robot-ip "$ROBOT_IP")
[[ "$NO_GOAL" -eq 1 ]] && DRIVE_ARGS+=(--no-goal)
python3 "$SCRIPT_DIR/04_drive_bridge.py" "${DRIVE_ARGS[@]}" || \
    echo "  [warn] 04_drive_bridge.py reported failures; bag still captured for diagnosis"

# Tail to capture /grid_nav_node/last_result + run_end /experiment/events
# after 04 exits. The action result publish lags goal completion by one
# control tick; 2 s is plenty.
sleep 2.0

# ---- stop the bag -----------------------------------------------------------
echo "[05/3] stopping bag"
STOP_OUT=$(WS="$WS" bash "$REPO_DIR/scripts/tb3_bag_record.sh" stop \
    --run-id "$RUN_ID" --bag-name "$BAG_NAME")
echo "$STOP_OUT" | sed 's/^/  /'

MCAP_PATH=$(awk -F= '/^mcap_path=/ { print $2 }' <<< "$STOP_OUT")
ELAPSED=$(awk -F= '/^elapsed_s=/ { print $2 }' <<< "$STOP_OUT")
CAPPED=$(awk -F= '/^capped=/    { print $2 }' <<< "$STOP_OUT")

if [[ -z "$MCAP_PATH" || ! -f "$MCAP_PATH" ]]; then
    echo "[05] FAIL: no mcap produced at '$MCAP_PATH'"
    exit 1
fi

# ---- summarize --------------------------------------------------------------
echo
echo "[05/4] ros2 bag info $MCAP_PATH"
ros2 bag info "$MCAP_PATH" | sed 's/^/  /'

echo
echo "[05] elapsed=${ELAPSED}s capped=${CAPPED}"
echo "[05] bag at $MCAP_PATH"

# ---- per-topic count check (each expected topic must be non-empty) ----------
INFO=$(ros2 bag info "$MCAP_PATH")
FAILED=0
echo
echo "=== per-topic counts (expected > 0) ==="
for t in $TOPICS; do
    # `ros2 bag info` lines look like: "  Topic: /foo | Type: ... | Count: 123 | ..."
    COUNT=$(awk -v topic="$t" '
        $0 ~ "Topic: "topic"[[:space:]]*\\|" { match($0, /Count: [0-9]+/); print substr($0, RSTART+7, RLENGTH-7); exit }
    ' <<< "$INFO")
    COUNT="${COUNT:-0}"
    if [[ "$COUNT" -gt 0 ]]; then
        printf "  [PASS] %-45s %s\n" "$t" "$COUNT"
    else
        printf "  [FAIL] %-45s 0\n" "$t"
        FAILED=$((FAILED + 1))
    fi
done

echo
if [[ "$FAILED" -eq 0 ]]; then
    echo "[05] all topics recorded"
    exit 0
else
    echo "[05] $FAILED topic(s) had no messages — inspect the bag and the live nodes"
    exit 1
fi
