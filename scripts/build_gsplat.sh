#!/usr/bin/env bash
# Build gsplat==1.4.0 FROM SOURCE with native sm_120 (Blackwell) kernels.
# No prebuilt gsplat wheel contains sm_120, so this is mandatory on RTX 50-series.
#
# Requires an nvcc that targets compute_120 => CUDA >= 12.8. Preferred toolkit is
# a full CUDA 12.8 install at $HOME/cuda-12.8 (NVIDIA runfile, no sudo needed).
set -euo pipefail

export https_proxy="${PROXY:-http://127.0.0.1:7897}"
export http_proxy="${https_proxy}"

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
PY=".venv/bin/python"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

# ---- 1. locate an nvcc >= 12.8 (needed for compute_120) --------------------
# Preferred: a locally installed full CUDA toolkit at $HOME/cuda-12.8
# (installed from the NVIDIA runfile; no sudo needed). Fallbacks: any nvcc on
# PATH, then the pip nvidia-cuda-nvcc package (note: in recent NVIDIA splits this
# package may only provide ptxas; if so the merge step will fail and you must
# install a full toolkit).
have_nvcc() { "$1" --version 2>/dev/null | grep -qE "release 12\.[89]|release 1[3-9]"; }

CANDIDATES=()
[ -n "${CUDA_HOME:-}" ] && CANDIDATES+=("$CUDA_HOME/bin/nvcc")
[ -x "$HOME/cuda-12.8/bin/nvcc" ] && CANDIDATES+=("$HOME/cuda-12.8/bin/nvcc")
[ -n "$(command -v nvcc || true)" ] && CANDIDATES+=("$(command -v nvcc)")

NCC=""
for c in "${CANDIDATES[@]}"; do
    if [ -n "$c" ] && [ -x "$c" ] && have_nvcc "$c"; then
        NCC="$c"; break
    fi
done
if [ -n "$NCC" ]; then
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NCC")")}"
    export PATH="$(dirname "$NCC"):$PATH"
    echo ">> using nvcc: $NCC  (CUDA_HOME=$CUDA_HOME)"
    "$NCC" --version | tail -2
else
    echo "ERROR: no nvcc >= 12.8 found." >&2
    echo "Install a full CUDA 12.8 toolkit, e.g. (no sudo needed):" >&2
    echo '  curl -fL -o /tmp/cuda.run https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.28.03_linux.run' >&2
    echo '  sh /tmp/cuda.run --silent --toolkit --toolkitpath=$HOME/cuda-12.8' >&2
    exit 1
fi

# ---- 2. source-build gsplat ------------------------------------------------
echo ">> building gsplat==1.4.0 from source (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
# The PyPI sdist omits the vendored glm header submodule, so build from a clone of
# the pinned tag instead (clone --recursive pulls gsplat/cuda/csrc/third_party/glm).
SRC="$PWD/.build/gsplat-1.4.0"
if [ ! -d "$SRC" ]; then
    echo ">> cloning gsplat v1.4.0 (with glm submodule)"
    git clone --depth 1 --recursive --branch v1.4.0 \
        https://github.com/nerfstudio-project/gsplat "$SRC"
fi
# --no-build-isolation so the build sees the venv's torch (gsplat's setup imports torch)
"$PY" -m pip install --no-build-isolation --no-cache-dir "$SRC"

echo ">> smoke: gsplat import + version"
"$PY" - <<'PY'
import gsplat
from gsplat.rendering import rasterization
import torch
print("gsplat", gsplat.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
PY
echo "build_gsplat.sh done"
