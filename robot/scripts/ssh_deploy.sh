#!/usr/bin/env bash
# =============================================================================
# ssh_deploy.sh — Deploy all ROS2 packages from this VM to the robot SBC.
#
# This repo lives on the build VM. The actual TB3 robot (mDNS:
# turtlebot3.local) only ships with stock ROBOTIS bringup, so the experiment
# code has to be pushed over the network. This script rsyncs src/ and
# scripts/ to the robot, then SSHes in and triggers build_all.sh there.
#
# Run scripts/robot_bootstrap_check.sh first on a fresh robot — it verifies
# the apt/system prerequisites this script depends on (ros base, colcon,
# rsync, smbus2, mcap plugin, i2c, chrony, wifi tools) and prints a single
# apt-install command for anything missing.
#
# CONFIGURATION (env vars):
#   ROBOT_HOST   SSH target                    (default: ubuntu@turtlebot3.local)
#                The stock ROBOTIS Pi image uses user `ubuntu`. Override if
#                your robot uses a different user or hostname.
#   REMOTE_DIR   Project root on the robot     (default: ~/ppo-tb3-research-project/robot)
#                build_all.sh on the robot symlinks $REMOTE_DIR/src/* into
#                $WS/src/, so the location is otherwise unobservable.
#   LOCAL_DIR    Project root on this VM       (default: auto-detected)
# =============================================================================
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROBOT_HOST="${ROBOT_HOST:-ubuntu@turtlebot3.local}"
REMOTE_DIR="${REMOTE_DIR:-~/ppo-tb3-research-project/robot}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$LOCAL_DIR/src" ]] || die "src/ not found in $LOCAL_DIR"

# Cheap sanity check: bail out early with a clear message if SSH key auth
# isn't set up, rather than failing midway through a long rsync.
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT_HOST" "true" >/dev/null 2>&1; then
    die "ssh $ROBOT_HOST failed (host unreachable or key auth not set up).
    Try: ssh-copy-id $ROBOT_HOST"
fi

echo "[1/3] Syncing src/ and scripts/ to $ROBOT_HOST:$REMOTE_DIR"
# Ensure the destination dirs exist (the robot is a fresh box).
ssh "$ROBOT_HOST" "mkdir -p $REMOTE_DIR/src $REMOTE_DIR/scripts"
rsync -av --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache' \
    --exclude 'build/' --exclude 'install/' --exclude 'log/' \
    --exclude 'ppo_inference_service/' \
    "$LOCAL_DIR/src/" "$ROBOT_HOST:$REMOTE_DIR/src/"

rsync -av \
    "$LOCAL_DIR/scripts/" "$ROBOT_HOST:$REMOTE_DIR/scripts/"

# The robot only ever receives src/ and scripts/ via rsync, so it has no
# .git directory and `git rev-parse` there returns "unknown". Stamp the
# build-host SHA into a sidecar file that diagnostics_node reads as a
# fallback (PowerSample / SessionMetadata trace which code shipped this run).
GIT_SHA=$(git -C "$LOCAL_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
echo "[2/3] Writing .git_sha=$GIT_SHA to $ROBOT_HOST:$REMOTE_DIR/.git_sha"
ssh "$ROBOT_HOST" "printf '%s\n' '$GIT_SHA' > $REMOTE_DIR/.git_sha"

echo "[3/3] Running build_all.sh on $ROBOT_HOST"
ssh "$ROBOT_HOST" "bash $REMOTE_DIR/scripts/build_all.sh"

echo "[done] Deploy complete. To launch on the robot:"
echo "  ssh $ROBOT_HOST"
echo "  source ~/turtlebot3_ws/install/setup.bash"
echo "  bash $REMOTE_DIR/scripts/smoke/00_preflight.sh   # post-deploy verification"
echo "  bash $REMOTE_DIR/scripts/smoke/01_robot_sensors.sh"
