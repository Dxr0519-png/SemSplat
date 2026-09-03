#!/usr/bin/env bash
# Short A/B trainings: GT-pose dataset vs ORB-estimated-pose dataset.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
ITERS=6000
.venv/bin/ns-train semsplat-replica --data data/replica_demo_ns \
    --max-num-iterations "$ITERS" --output-dir outputs/ab_gt \
    --pipeline.model.num-downscales 1 --pipeline.model.resolution-schedule 100000
.venv/bin/ns-train semsplat-replica --data data/replica_orb_est_ns \
    --max-num-iterations "$ITERS" --output-dir outputs/ab_est \
    --pipeline.model.num-downscales 1 --pipeline.model.resolution-schedule 100000
echo "done"
