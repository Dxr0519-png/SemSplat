#!/usr/bin/env bash
# Smoke test: sm_120 capability, gsplat raster forward/backward, method
# registration, and a 30-iteration training run on a synthetic mini scene.
set -euo pipefail
export https_proxy="${PROXY:-http://127.0.0.1:7897}"
export http_proxy="${https_proxy}"

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
PY=".venv/bin/python"
NS=".venv/bin/ns-train"

echo ">> 1/4 cuda capability"
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "cuda not available"
cap = torch.cuda.get_device_capability()
print("capability", cap)
assert cap == (12, 0), f"expected sm_120, got {cap}; wrong torch build (need cu128+)"
PY

echo ">> 2/4 gsplat raster forward+backward"
"$PY" - <<'PY'
import torch
from gsplat.rendering import rasterization
dev = "cuda"
N = 500
means = torch.randn(N, 3, device=dev) * 0.5
quats = torch.randn(N, 4, device=dev)
scales = (torch.rand(N, 3, device=dev) + 0.5) * 0.05
opac = torch.rand(N, device=dev) * 0.8 + 0.1
colors = torch.rand(N, 3, device=dev)
view = torch.eye(4, device=dev)[None]
K = torch.tensor([[160., 0, 80], [0, 160, 64], [0, 0, 1]], device=dev)[None]
img, alpha, meta = rasterization(means, quats, scales, opac, colors, view, K, 160, 128,
                                 sh_degree=0, render_mode="RGB", backgrounds=None)
loss = img.sum() + (1 - alpha).sum()
loss.backward()
assert means.grad is not None
print("raster OK, render", tuple(img.shape), "alpha", float(alpha.mean()))
PY

echo ">> 3/4 method registration"
"$NS" --help 2>/dev/null | grep -q "semsplat" && echo "ns-train lists semsplat" || { echo "semsplat NOT registered"; exit 1; }

echo ">> 4/4 tiny train (60 iters)"
"$PY" scripts/make_mini_scene.py --out /tmp/mini_scene
"$NS" semsplat --data /tmp/mini_scene \
    --max-num-iterations 60 \
    --viewer.quit-on-train-completion True \
    --vis viewer --output-dir /tmp/smoke_out

echo "smoke_plugin.sh PASSED"
