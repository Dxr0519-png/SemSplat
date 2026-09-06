#!/usr/bin/env bash
# waldo_kitchen 20k 全分辨率档, 教师换 LangSplat 同款 OpenCLIP(ViT-B/16 laion2b_s34b_b88k)。
# 目的: 把 6k 半分辨率快速档拉长到 canonical 20k 全分辨率档(office0_320_sam20k_reg 同思路,
# 那套是 mIoU 0.240 的最佳配方), 再跑 eval_lerf_langsplat.py 逐字协议评估, 看训练档能追回多少。
# 配方 = office0 canonical 非默认 knob(num-downscales 0 / random-scale 5 / sem-start 200 /
#         sem-tv 1.0 / opacity-entropy 0.05@1000) + openclip 教师 + waldo 数据, 其余默认。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# 教师只从本地文件 data/checkpoints/openclip/ViT-B-16_laion2b_s34b_b88k.bin 载权重,
# 无需联网; 其余 HF 模型走离线缓存。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/ns-train semsplat-replica \
    --data data/waldo_kitchen_ns \
    --output-dir outputs/wk_oc20k_reg \
    --vis tensorboard \
    --max-num-iterations 20000 \
    --pipeline.model.num-downscales 0 \
    --pipeline.model.resolution-schedule 3000 \
    --pipeline.model.random-scale 5.0 \
    --pipeline.model.teacher-kind samclip \
    --pipeline.model.semantic-start-iter 200 \
    --pipeline.model.teacher-clip-backend openclip \
    --pipeline.model.teacher-openclip-arch ViT-B-16 \
    --pipeline.model.teacher-openclip-pretrained laion2b_s34b_b88k \
    --pipeline.model.teacher-sam-mode amg \
    --pipeline.model.teacher-sam-points-per-side 8 \
    --pipeline.model.teacher-sam-crop-mode ctx \
    --pipeline.model.sem-tv-mult 1.0 \
    --pipeline.model.sem-tv-depth-sigma 0.1 \
    --pipeline.model.opacity-entropy-mult 0.05 \
    --pipeline.model.opacity-entropy-start-iter 1000
echo "done"
