"""Frozen Feature-3DGS-style per-pixel LSeg teacher (drop-in for ClipLocalTeacher).

Implements the same duck-type contract the SemsplatModel consumes:

    teacher.dim                      == 512 (semantic_head_dim)
    teacher.compute(image, key)      -> L2-normalized [gH, gW, 512] (fp16 LRU)

Difference vs the CLIP patch teacher: the dense features come from LSeg
(``clip_vitl16_384`` + ``demo_e200.ckpt``), which is trained for *per-pixel*
language-grounded segmentation, so each Gaussian is supervised by the identity
of the object under it instead of an entangled whole-image patch token.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F


class LSegDenseTeacher:
    def __init__(
        self,
        ckpt_path: str,
        input_long_side: int = 384,
        cell: int = 8,
        l2norm: bool = True,
        fp16: bool = True,
        max_entries: int = 2000,
        device: str = "cuda",
    ):
        from semsplat.teachers.lseg._load import build_lseg_net

        self.ckpt_path = ckpt_path
        self.input_long_side = int(input_long_side)
        self.cell = int(cell)
        self.l2norm = bool(l2norm)
        self.fp16 = fp16
        self.device = torch.device(device)
        self.dim = 512
        self._cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._cache_max = int(max_entries)

        self._net = build_lseg_net(ckpt_path, device=str(self.device), fp16=fp16)
        self._net.eval()

    # ------------------------------------------------------------- encoding
    @torch.no_grad()
    def dense_features(self, image: torch.Tensor) -> torch.Tensor:
        """[H, W, 3] float image in 0..1 -> L2-normalized [gH, gW, 512]."""
        img = image.detach().float().to(self.device)  # [H,W,3]
        H, W = img.shape[:2]
        L = self.input_long_side
        scale = L / max(H, W)
        hh, ww = max(1, int(round(H * scale))), max(1, int(round(W * scale)))

        x = img.permute(2, 0, 1)[None]  # [1,3,H,W]
        x = F.interpolate(x, size=(hh, ww), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5  # LSeg normalization (mean = std = 0.5)

        # pad to multiples of 32 so the ViT patch grid stays even (odd grids make
        # the DPT refinenet skip-connections misalign, e.g. 23 vs 24 rows)
        Hp, Wp = int(math.ceil(hh / 32.0) * 32), int(math.ceil(ww / 32.0) * 32)
        if (Hp, Wp) != (hh, ww):
            ph, pw = Hp - hh, Wp - ww
            x = F.pad(x, (0, pw, 0, ph), value=0.0)  # 0.0 after normalization = mean pixel

        net = self._net
        dtype = next(net.parameters()).dtype
        feats = net.forward_dense(x.to(dtype))  # [1,512,Hp,Wp]
        feats = F.interpolate(feats, size=(hh, ww), mode="bilinear", align_corners=False)[0]  # [512,hh,ww]

        if self.l2norm:
            feats = F.normalize(feats, dim=0)

        # average-pool into the cache grid; [gH,gW,512]
        gH, gW = max(1, int(math.ceil(hh / self.cell))), max(1, int(math.ceil(ww / self.cell)))
        pooled = F.adaptive_avg_pool2d(feats[None], (gH, gW))[0]  # [512,gH,gW]
        pooled = pooled.permute(1, 2, 0)  # [gH,gW,512]
        return pooled.float()

    # ------------------------------------------------------------- cache
    def compute(self, image: torch.Tensor, key: Optional[int] = None) -> torch.Tensor:
        """Cached dense_features; ``key`` is the cam_idx for the LRU."""
        if key is not None and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        grid = self.dense_features(image)
        if key is not None:
            if self.fp16:
                grid = grid.half()
            self._cache[key] = grid
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return grid

    def clear_cache(self) -> None:
        self._cache.clear()
