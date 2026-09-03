#!/usr/bin/env bash
# Step 1 of env bring-up: create venv + install PyTorch 2.7.0 cu128 (native sm_120 kernels).
# NOTE: must run BEFORE gsplat source build (build_gsplat.sh) and nerfstudio install.
# Uses `uv` when available (no sudo / ensurepip needed); falls back to `python3 -m venv`.
# Requires outbound network via the local proxy at 127.0.0.1:7897 (override PROXY if needed).
set -euo pipefail

export https_proxy="${PROXY:-http://127.0.0.1:7897}"
export http_proxy="${https_proxy}"
export UV_HTTP_TIMEOUT=600

cd "$(dirname "$0")/.." || exit 1

if command -v uv >/dev/null 2>&1; then
    echo ">> using uv"
    uv venv .venv --python 3.12
    UVPY="uv pip install --python .venv/bin/python"
else
    echo ">> uv not found; falling back to python3 -m venv (needs python3-venv / ensurepip)"
    python3 -m venv .venv
    UVPY=".venv/bin/python -m pip install"
fi

echo ">> installing torch 2.7.0 cu128 (sm_120)"
$UVPY -U pip wheel setuptools 2>/dev/null || true
$UVPY torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo ">> sanity: cuda capability"
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("capability", torch.cuda.get_device_capability())
PY
echo "build_env.sh done"
