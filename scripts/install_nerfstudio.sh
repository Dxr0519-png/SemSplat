#!/usr/bin/env bash
# Install nerfstudio 1.1.5 + runtime extras, then this package (editable).
# Run AFTER build_gsplat.sh (gsplat==1.4.0 must already be built for sm_120).
set -euo pipefail

export https_proxy="${PROXY:-http://127.0.0.1:7897}"
export http_proxy="${https_proxy}"

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
PY=".venv/bin/python"

echo ">> installing nerfstudio==1.1.5"
"$PY" -m pip install -q nerfstudio==1.1.5

echo ">> installing runtime extras (transformers, plyfile)"
"$PY" -m pip install -q "transformers>=4.40" plyfile pyyaml

echo ">> installing semsplat (editable)"
"$PY" -m pip install -e ".[dev]" -q

echo ">> sanity: imports + version"
"$PY" - <<'PY'
import gsplat, torch
print("gsplat", gsplat.__version__)
try:
    import nerfstudio
    print("nerfstudio OK")
except Exception as e:
    print("nerfstudio import FAILED:", repr(e)); raise
from nerfstudio.engine.trainer import TrainerConfig  # noqa
print("imports OK")
PY
echo "install_nerfstudio.sh done"
