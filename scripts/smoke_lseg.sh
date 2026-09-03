#!/usr/bin/env bash
# Regression smoke: default (clip) teacher path must still train after the
# teacher_kind refactor. Semantic_start_iter default 3000 > 60, so the teacher
# never loads here; this only guards the train loop / config plumbing.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
.venv/bin/python scripts/make_mini_scene.py --out /tmp/mini_scene_lseg >/dev/null
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/ns-train semsplat --data /tmp/mini_scene_lseg \
    --max-num-iterations 60 --output-dir outputs/smoke_lseg \
    --viewer.quit-on-train-completion True --viewer.websocket-port 7007
echo "clip-default smoke OK"
