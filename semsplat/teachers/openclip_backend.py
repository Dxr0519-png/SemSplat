"""OpenCLIP (laion2b ViT-B/16) 后端 — LangSplat 论文同款文本/图像骨干。

OpenAI-CLIP(HF transformers) 与 OpenCLIP(laion2b 训练) 是两个**不共享** embedding
空间的生态: 各自 tokenizer/投影权重/统计都不一样, 直接互查 ≈ 随机。本模块给
samclip 物体-crop 教师与 eval 端 query 提供一条 lazy 的 open_clip 通路, 使图像的
CLS 特征与文本特征一起落进 open_clip 的共享 512 空间 —— 即 LangSplat
``openclip_encoder`` 使用的那个空间。这样学生学到的逐高斯特征才能用 LangSplat 的
固定阈值(0.4)/固定负样本(relevancy)尺子去量。

两个入口都持有**同一个** open_clip 模型实例(模块级缓存, 按 arch:pretrained 键),
避免训练教师与 eval 文本各载一份吃 VRAM:

  * ``OpenClipVisual`` (lazy 构建于 SamClipTeacher 内):
        encode_tensor(x [K,3,224,224] 已按 CLIP 统计归一) -> [K,512] L2 归一 CLS@proj
    与 HF 教师 ``clip.vision_model(...).pooler_output @ visual_projection`` 一一对应。
    crop 恒为 224x224 → 无位置编码插值问题, open_clip 原生 forward 即可。
  * ``openclip_text(model_key, prompts) -> {prompt: [512] L2}`` (eval 端 query):
        ``open_clip.get_tokenizer + model.encode_text(..., normalize=True)``

单例以 字符串 身份识别: ``"openclip:ViT-B-16:laion2b_s34b_b88k"``。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from semsplat.teachers.clip_local_teacher import CLIP_MEAN, CLIP_STD

OPENCLIP_PREFIX = "openclip:"

# 本机离线权重目录(与 data/checkpoints/lseg 同约定): <arch>_<pretrained>.bin
_CKPT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "openclip"


def _resolve_pretrained(arch: str, pretrained: str) -> str:
    """tag -> 本地 bin(若已下载)否则原 tag(open_clip 联网兜底)。

    返回给 create_model_and_transforms 的 pretrained。本地路径被 open_clip 识别为
    'tag or file path'(factory.py 里 isfile 分支), 不会再走 HF 下载。
    """
    if os.path.isfile(pretrained):
        return pretrained
    local = _CKPT_DIR / f"{arch}_{pretrained}.bin"
    if local.is_file():
        return str(local)
    return pretrained


def parse_model_key(model_key: str):
    """'openclip:ARCH:PRETRAINED' -> (arch, pretrained)."""
    _, arch, pretrained = model_key.split(":", 2)
    return arch, pretrained


class OpenClipVisual:
    """Lazy open_clip model doing 224x224 CLS crop encoding in the shared 512 space.

    Mirror of the HF path inside SamClipTeacher._encode: input tensors are already
    (rgb/255 - CLIP_MEAN)/CLIP_STD, [K,3,224,224]; we push them through
    ``model.encode_image(..., normalize=True)``.
    """

    def __init__(self, arch: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k",
                 fp16: bool = True, device: str = "cuda"):
        import open_clip

        self.arch, self.pretrained = arch, pretrained
        self.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32
        # create_model_and_transforms builds visual+text towers + default transforms;
        # we keep only the weights (transforms are unused: crops are pre-sized 224).
        src = _resolve_pretrained(arch, pretrained)
        model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=src, device=device)
        self.model = model.to(self.device).eval().to(self.dtype)
        self.model.requires_grad_(False)
        vis_dim = getattr(self.model.visual, "output_dim", None) or getattr(self.model, "embed_dim", 512)
        self.dim = int(vis_dim)

    @torch.no_grad()
    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """[K,3,224,224] 已 CLIP-归一 -> [K,512] L2 归一 (float32).

        直接走 ``model.visual``(pooled CLS @ proj)再自归一, 绕开 ``encode_image``
        可能附带的 image_mean/image_std 重归一(版本差异), 保证与我们喂入的
        (rgb/255-CLIP_MEAN)/CLIP_STD 一致。
        """
        out = self.model.visual(x.to(self.device).to(self.dtype))
        return F.normalize(out.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, prompts) -> Dict[str, torch.Tensor]:
        """L2 归一 open_clip 文本向量 (与 encode_tensor 共享空间)."""
        import open_clip

        tok = open_clip.get_tokenizer(self.arch)
        text = tok(list(prompts)).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=False):
            emb = self.model.encode_text(text, normalize=True).float()
        return {p: emb[i] for i, p in enumerate(prompts)}


# 模块级单例: arch:pretrained -> OpenClipVisual
_VISUALS: Dict[str, OpenClipVisual] = {}


def get_openclip_visual(arch: str, pretrained: str, fp16: bool = True,
                        device: str = "cuda") -> OpenClipVisual:
    key = f"{arch}:{pretrained}"
    if key not in _VISUALS:
        _VISUALS[key] = OpenClipVisual(arch, pretrained, fp16=fp16, device=device)
    return _VISUALS[key]


def openclip_text(model_key: str, prompts, device: str = "cuda",
                  fp16: bool = True) -> Dict[str, torch.Tensor]:
    """''openclip:ARCH:PRETRAINED'' + prompts -> L2 归一 512 文本向量 (单例复用)."""
    arch, pretrained = parse_model_key(model_key)
    return get_openclip_visual(arch, pretrained, fp16=fp16, device=device).encode_text(prompts)
