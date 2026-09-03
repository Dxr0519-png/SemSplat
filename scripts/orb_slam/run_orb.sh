#!/usr/bin/env bash
# Run ORB-SLAM3 RGB-D on a prepared feed and copy the trajectories back.
# Usage: run_orb.sh [orb_seq_dir]   (default: data/orb_replica)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ORB="${ORB3_DIR:-$HOME/github_project/ORB_SLAM3}"
SEQ="${1:-$REPO/data/orb_replica}"
cd "$ORB"
rm -f CameraTrajectory.txt KeyFrameTrajectory.txt
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt "$SEQ/orb.yaml" "$SEQ" "$SEQ/assoc.txt"
cp -f CameraTrajectory.txt "$SEQ/CameraTrajectory.txt"
cp -f KeyFrameTrajectory.txt "$SEQ/KeyFrameTrajectory.txt"
echo "OK: trajectories copied to $SEQ"
