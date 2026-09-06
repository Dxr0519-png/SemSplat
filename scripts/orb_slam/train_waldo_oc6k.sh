#!/usr/bin/env bash
# waldo_kitchen 6k 快速档, 教师换 LangSplat 同款 OpenCLIP(ViT-B/16 laion2b_s34b_b88k)。
# 目的: 让逐高斯语义特征落进 open_clip 共享空间, 再跑 eval_lerf_langsplat.py 的
# "完全同协议"(固定阈值0.4/固定负样本/softmax-relevancy)评估与 LangSplat 44.5 直比。
# 配方与 wk_sam6k(OpenAI-CLIP)逐项一致, 仅 teacher_clip_backend 不同。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# 教师只从本地文件 data/checkpoints/openclip/ViT-B-16_laion2b_s34b_b88k.bin 载权重,
# 无需联网; 其余 HF 模型走离线缓存。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/ns-train semsplat-replica \
    --data data/waldo_kitchen_ns \
    --output-dir outputs/wk_oc6k_reg \
    --vis tensorboard \
    --max-num-iterations 6000 \
    --pipeline.model.num-downscales 1 \
    --pipeline.model.resolution-schedule 100000 \
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
