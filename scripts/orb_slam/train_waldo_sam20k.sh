#!/usr/bin/env bash
# waldo_kitchen 20k 全分辨率档, 教师 = 原版 OpenAI-CLIP(HF B16, samclip 默认/唯一 canonical 骨干)。
# 目的: office0_320_sam20k_reg(0.240, OpenAI) 同款配方搬到 waldo, 与 wk_oc20k(OpenCLIP 教师)
# 对照 —— 在咱们自己的 OpenAI 空间里量"训练档红利"(6k wk_sam6k 宏均 0.359 → 20k ?),
# 同时与 openclip-20k 宏均 0.367 比谁的空间在真实 LERF 场景更强。
# 配方 = office0 canonical 非默认 knob 逐项同 wk_oc20k.sh, 仅去掉 teacher-clip-backend 覆盖(=openai 默认)。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# 所有 HF 模型走离线缓存; OpenAI-CLIP 权重在 HF 缓存里(无需下载)。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/ns-train semsplat-replica \
    --data data/waldo_kitchen_ns \
    --output-dir outputs/wk_sam20k_reg \
    --vis tensorboard \
    --max-num-iterations 20000 \
    --pipeline.model.num-downscales 0 \
    --pipeline.model.resolution-schedule 3000 \
    --pipeline.model.random-scale 5.0 \
    --pipeline.model.teacher-kind samclip \
    --pipeline.model.semantic-start-iter 200 \
    --pipeline.model.teacher-sam-mode amg \
    --pipeline.model.teacher-sam-points-per-side 8 \
    --pipeline.model.teacher-sam-crop-mode ctx \
    --pipeline.model.sem-tv-mult 1.0 \
    --pipeline.model.sem-tv-depth-sigma 0.1 \
    --pipeline.model.opacity-entropy-mult 0.05 \
    --pipeline.model.opacity-entropy-start-iter 1000
echo "done"
