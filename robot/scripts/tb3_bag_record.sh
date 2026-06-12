#!/usr/bin/env bash
# =============================================================================
# tb3_bag_record.sh — start/stop a per-run ros2 bag recording over SSH.
#
# Invoked by the orchestrator (minimap-visualizer) on each ROS host. No ROS
# service control plane — SSH is the only transport (the orchestrator already
# needs SSH for scp pull-back). See handover_orchestrator §4 and plan §4.1.
#
# Usage:
#   tb3_bag_record.sh start --run-id ID --bag-name NAME \
#                           --topics "/a /b /c" [--max-duration-s 300]
#   tb3_bag_record.sh stop  --run-id ID --bag-name NAME
#
# On `start`: spawns `timeout --signal=SIGINT <dur> ros2 bag record -s mcap`
# in the background, writes a pid + start-epoch sidecar. The timeout enforces
# the hard cap (SIGINT closes the mcap cleanly). Idempotent: refuses to start
# if a recording for this (run_id, bag_name) is already live.
#
# On `stop`: SIGINTs the recorder, waits for a clean exit, consolidates the
# single inner *.mcap to /tmp/experiments/<run_id>/<bag_name>.mcap (so the
# orchestrator scp's a deterministic file), removes the bag dir, and prints:
#   elapsed_s=<float>
#   capped=<true|false>
#   mcap_path=/tmp/experiments/<run_id>/<bag_name>.mcap
# =============================================================================
set -euo pipefail

ROOT="/tmp/experiments"
WS="${WS:-$HOME/turtlebot3_ws}"
STORAGE="${STORAGE:-mcap}"   # plan mandates mcap; override only for local testing
# Requires `ros-humble-rosbag2-storage-mcap` on each recording host.

die() { echo "ERROR: $*" >&2; exit 1; }

# Re-derive the ROS DDS networking env from the user's login+interactive shell.
#
# The orchestrator invokes this script over NON-interactive SSH
# (`ssh HOST "tb3_bag_record.sh start ..."`). A non-interactive shell does NOT
# source ~/.bashrc, so a non-default ROS_DOMAIN_ID / RMW_IMPLEMENTATION set there
# is absent and the recorder silently falls back to ROS_DOMAIN_ID=0 — a DIFFERENT
# DDS partition than the interactively-launched compute stack. `ros2 bag record`
# then starts cleanly, subscribes to nothing, and writes a VALID-BUT-EMPTY mcap
# (clean header + footer, zero messages). This is the centralized-mode empty-bag
# bug. We fix it by pulling the relevant vars from a login+interactive shell
# (`bash -lic`, which DOES source ~/.bashrc) so the recorder shares the stack's
# partition. Vars already set in the current environment win, so an explicit
# `ROS_DOMAIN_ID=.. ssh ..` override from the orchestrator still takes precedence.
import_login_ros_env() {
    local relevant="ROS_DOMAIN_ID RMW_IMPLEMENTATION ROS_LOCALHOST_ONLY \
ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS CYCLONEDDS_URI \
FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE"
    local login_env v val
    # One login+interactive shell dump; tolerate failure (falls back to defaults).
    # `timeout` guards against a pathological ~/.bashrc that blocks on input —
    # the recorder must never hang here. -lic so ~/.bashrc IS sourced (the
    # interactive guard at its top would otherwise `return` early).
    # NOTE: explicit `if` blocks (not `test && cmd`) — under the script's `set -e`
    # a trailing false `&&` list would abort source_ros.
    login_env="$(timeout 10 bash -lic 'export -p' 2>/dev/null || true)"
    if [[ -z "$login_env" ]]; then return 0; fi
    for v in $relevant; do
        if [[ -n "${!v:-}" ]]; then continue; fi         # current env wins
        # `export -p` prints: declare -x VAR="value"  (value is double-quoted).
        val="$(sed -n "s/^declare -x $v=\"\(.*\)\"\$/\1/p" <<<"$login_env" | head -n1)"
        if [[ -n "$val" ]]; then export "$v=$val"; fi
    done
    return 0
}

# Source ROS so `ros2` is on PATH under a non-interactive SSH session, and make
# sure the DDS networking env matches the stack (see import_login_ros_env).
source_ros() {
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    [[ -f "$WS/install/setup.bash" ]] && source "$WS/install/setup.bash"
    import_login_ros_env
    set -u
}

# -- arg parsing --------------------------------------------------------------
CMD="${1:-}"; shift || true
RUN_ID=""; BAG_NAME=""; TOPICS=""; MAX_DURATION_S=300
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)        RUN_ID="$2"; shift 2 ;;
        --bag-name)      BAG_NAME="$2"; shift 2 ;;
        --topics)        TOPICS="$2"; shift 2 ;;
        --max-duration-s) MAX_DURATION_S="$2"; shift 2 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[[ -n "$RUN_ID" ]]   || die "--run-id required"
[[ -n "$BAG_NAME" ]] || die "--bag-name required"

RUN_DIR="$ROOT/$RUN_ID"
PID_FILE="$RUN_DIR/$BAG_NAME.pid"
EPOCH_FILE="$RUN_DIR/$BAG_NAME.epoch"
BAG_DIR="$RUN_DIR/$BAG_NAME"           # ros2 bag record -o target (a directory)
EXT="mcap"; [[ "$STORAGE" == "sqlite3" ]] && EXT="db3"
FINAL_MCAP="$RUN_DIR/$BAG_NAME.$EXT"   # consolidated single file for scp

case "$CMD" in
  start)
    [[ -n "$TOPICS" ]] || die "--topics required for start"
    mkdir -p "$RUN_DIR"
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        die "recording already live for $RUN_ID/$BAG_NAME (pid $(cat "$PID_FILE"))"
    fi
    [[ -e "$BAG_DIR" ]] && rm -rf "$BAG_DIR"
    source_ros

    # --include-hidden-topics is REQUIRED for underscore-prefixed action
    # topics (/move_to_grid/_action/feedback and /_action/status) — without
    # it, ros2 bag record silently refuses to subscribe even when they're
    # named explicitly. The plan §3 topic table includes those, so this
    # flag is part of the contract, not an optional convenience.
    # shellcheck disable=SC2086 — TOPICS is an intentional space-separated list
    nohup timeout --signal=SIGINT "$MAX_DURATION_S" \
        ros2 bag record -s "$STORAGE" -o "$BAG_DIR" \
            --include-hidden-topics $TOPICS \
        > "$RUN_DIR/$BAG_NAME.bag.log" 2>&1 &
    echo $! > "$PID_FILE"
    date +%s.%N > "$EPOCH_FILE"

    # -- Verify the recorder shares the stack's DDS partition -----------------
    # The signature centralized-mode failure (see import_login_ros_env) is a
    # recorder on the wrong ROS_DOMAIN_ID that subscribes to nothing and writes
    # a valid-but-EMPTY mcap. The compute stack is long-lived (up before per-run
    # recording starts), so its topics MUST already be visible here. Poll the
    # graph (--include-hidden-topics so the /_action/* topics count); if NONE of
    # the requested topics are visible the recorder is on the wrong partition (or
    # the stack is down) — tear down and exit non-zero so the orchestrator aborts
    # the run (plan §13) instead of burning experiment time on an empty bag.
    seen=0; total=0; missing=""
    for _ in $(seq 1 16); do          # up to ~8s for DDS graph discovery
        visible="$(ros2 topic list --include-hidden-topics 2>/dev/null || true)"
        seen=0; total=0; missing=""
        for t in $TOPICS; do
            total=$((total + 1))
            if grep -qxF -- "$t" <<<"$visible"; then
                seen=$((seen + 1))
            else
                missing="$missing $t"
            fi
        done
        if [[ "$seen" -gt 0 ]]; then break; fi
        sleep 0.5
    done
    if [[ "$seen" -eq 0 ]]; then
        kill -SIGINT "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -rf "$BAG_DIR"; rm -f "$PID_FILE" "$EPOCH_FILE"
        die "recorder sees 0/$total requested topics on ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} (RMW=${RMW_IMPLEMENTATION:-default}). The compute stack is on a different DDS partition or not running — the bag would be EMPTY. Aborting; ensure the recorder and the stack share ROS_DOMAIN_ID/RMW_IMPLEMENTATION."
    fi
    echo "topics_visible=$seen/$total"
    if [[ -n "$missing" ]]; then echo "topics_missing:$missing" >&2; fi
    echo "started pid=$(cat "$PID_FILE") bag_dir=$BAG_DIR"
    ;;

  stop)
    [[ -f "$PID_FILE" ]] || { echo "elapsed_s=0"; echo "capped=false"; echo "mcap_path="; exit 0; }
    PID="$(cat "$PID_FILE")"

    # SIGINT the `timeout` wrapper; it forwards to ros2 bag for a clean close.
    kill -SIGINT "$PID" 2>/dev/null || true
    for _ in $(seq 1 50); do          # up to ~5 s
        kill -0 "$PID" 2>/dev/null || break
        sleep 0.1
    done
    kill -9 "$PID" 2>/dev/null || true

    # Elapsed time + cap detection.
    START_EPOCH="$(cat "$EPOCH_FILE" 2>/dev/null || echo 0)"
    NOW="$(date +%s.%N)"
    # LC_ALL=C so awk parses/prints '.' as the decimal separator regardless of
    # the host locale (a comma here would break the orchestrator's float parse).
    ELAPSED="$(LC_ALL=C awk -v a="$NOW" -v b="$START_EPOCH" 'BEGIN{printf "%.2f", a-b}')"
    CAPPED=$(LC_ALL=C awk -v e="$ELAPSED" -v m="$MAX_DURATION_S" 'BEGIN{print (e+0.5>=m)?"true":"false"}')

    # Consolidate the single inner data file to a deterministic path.
    INNER_MCAP="$(find "$BAG_DIR" -maxdepth 1 -name "*.$EXT" 2>/dev/null | head -1 || true)"
    if [[ -n "$INNER_MCAP" ]]; then
        mv "$INNER_MCAP" "$FINAL_MCAP"
        rm -rf "$BAG_DIR"
        MCAP_OUT="$FINAL_MCAP"
    else
        MCAP_OUT=""   # nothing recorded — orchestrator treats as aborted
    fi

    rm -f "$PID_FILE" "$EPOCH_FILE"
    echo "elapsed_s=$ELAPSED"
    echo "capped=$CAPPED"
    echo "mcap_path=$MCAP_OUT"
    ;;

  *)
    die "usage: $0 {start|stop} --run-id ID --bag-name NAME [--topics ...] [--max-duration-s N]"
    ;;
esac
