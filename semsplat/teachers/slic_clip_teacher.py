"""SLIC-superpixel + CLIP-crop dense teacher (lightweight LangSplat-style).

Why: whole-image CLIP *patch* tokens are spatially entangled (an object and its
surroundings share a patch), which makes per-Gaussian semantics learn wrong
identities. Here each SLIC superpixel isolates a region, the region's bounding
box is CLIP-encoded, and every pixel inside the region is supervised to point at
that *object-local* CLIP vector. Identity is thus provided by construction:
a crop of a monitor is CLIP-encoded as a monitor, a crop of a sofa as a sofa.

Implements the same duck-type contract as ClipLocalTeacher:

    teacher.dim                   == 512 (semantic_head_dim)
    teacher.compute(image, key)   -> L2-normalized [gH, gW, 512] (fp16 LRU)

Depends only on skimage + the already-cached OpenAI CLIP ViT-B/16 checkpoint.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_CROP = 224  # CLIP ViT-B/16 input size


class SuperClipTeacher:
    def __init__(
        self,
        image_model_name: str = "openai/clip-vit-base-patch16",
        n_segments: int = 250,
        compactness: float = 10.0,
        crop_pad: float = 1.15,
        cell: int = 16,
        fp16: bool = True,
        max_entries: int = 2000,
        device: str = "cuda",
    ):
        from semsplat.teachers.clip_local_teacher import CLIP_MEAN, CLIP_STD, load_clip

        self.image_model_name = image_model_name
        self.n_segments = int(n_segments)
        self.compactness = float(compactness)
        self.crop_pad = float(crop_pad)
        self.cell = int(cell)
        self.fp16 = fp16
        self.device = torch.device(device)
        self.dim = 512  # OpenAI CLIP ViT-B visual_projection output
        self._cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._cache_max = int(max_entries)

        self._clip = load_clip(image_model_name, fp16=fp16, device=str(self.device)).eval()
        self._mean = CLIP_MEAN.to(self.device)
        self._std = CLIP_STD.to(self.device)

    # ------------------------------------------------------------- regions
    @staticmethod
    def _merge_tiny(labels: np.ndarray, min_area: int) -> np.ndarray:
        """Merge regions smaller than min_area into their dominant neighbor."""
        out = labels.copy()
        ids, counts = np.unique(labels, return_counts=True)
        big = set(ids[counts >= min_area].tolist())
        if len(big) == 0:
            return out
        big_label = int(ids[counts.argmax()])
        small = ids[counts < min_area]
        for s in small:
            ys, xs = np.where(out == s)
            # dominant label among 4-neighbours not equal to s
            neigh = []
            for y, x in zip(ys[:200], xs[:200]):  # sample is enough
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1] and out[yy, xx] != s:
                        neigh.append(int(out[yy, xx]))
            if neigh:
                c = np.bincount(neigh)
                cand = int(np.argmax(c))
                target = cand if cand in big else big_label
            else:
                target = big_label
            out[out == s] = target
        return out

    @torch.no_grad()
    def dense_features(self, image: torch.Tensor) -> torch.Tensor:
        """[H,W,3] float 0..1 -> L2-normalized [gH,gW,512] (superpixel-CLIP targets)."""
        from skimage.segmentation import slic

        img = image.detach().cpu().numpy()  # [H,W,3] 0..1
        H, W = img.shape[:2]
        uint8 = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)

        labels = slic(
            uint8,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=1.0,
            start_label=1,
        )
        labels = self._merge_tiny(labels, min_area=max(8, int(0.0006 * H * W)))
        regions = [int(u) for u in np.unique(labels)]

        crops = []
        for r in regions:
            ys, xs = np.where(labels == r)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            hh, ww = y1 - y0, x1 - x0
            # padded square crop with a bit of context
            side = max(hh, ww)
            pad = int(round(side * (self.crop_pad - 1)))
            y0p, y1p = max(0, y0 - pad), min(H, y1 + pad)
            x0p, x1p = max(0, x0 - pad), min(W, x1 + pad)
            crops.append(img[y0p:y1p, x0p:x1p])
        if not crops:
            raise RuntimeError("no superpixels produced")
        emb = self._encode_crops(crops)  # [R,512] fp32 cpu

        # dense per-pixel target map, then average-pool to the supervision grid
        target = np.zeros((H, W, 512), dtype=np.float32)
        flat = labels.reshape(-1)
        for idx, r in enumerate(regions):
            m = flat == r
            if m.any():
                target.reshape(-1, 512)[m] = emb[idx].numpy()
        t = torch.from_numpy(target).to(self.device).float()
        gH, gW = max(1, int(math.ceil(H / self.cell))), max(1, int(math.ceil(W / self.cell)))
        pooled = F.adaptive_avg_pool2d(t.permute(2, 0, 1)[None], (gH, gW))[0].permute(1, 2, 0)
        pooled = F.normalize(pooled, dim=-1)
        return pooled.float()

    @torch.no_grad()
    def _encode_crops(self, crops):
        """CLIP-encode region crops -> [R, 512] L2-normalized (cpu fp32)."""
        import cv2

        # pad each crop to a square (grey border) then resize to CLIP size (cv2 is fast)
        out = np.empty((len(crops), _CROP, _CROP, 3), dtype=np.uint8)
        for i, c in enumerate(crops):
            arr = (np.clip(c, 0, 1) * 255.0).round().astype(np.uint8)
            h, w = arr.shape[:2]
            if h != w:
                side = max(h, w)
                canvas = np.full((side, side, 3), 128, dtype=np.uint8)
                y0, x0 = (side - h) // 2, (side - w) // 2
                canvas[y0 : y0 + h, x0 : x0 + w] = arr
                arr = canvas
            out[i] = cv2.resize(arr, (_CROP, _CROP), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(out).permute(0, 3, 1, 2).to(self.device).float() / 255.0
        x = (x - self._mean[None, :, None, None]) / self._std[None, :, None, None]
        dtype = next(self._clip.parameters()).dtype
        out = self._clip.vision_model(pixel_values=x.to(dtype))
        pooled = out.pooler_output  # [R, hidden]
        emb = self._clip.visual_projection(pooled.to(dtype)).float()  # [R, 512]
        emb = F.normalize(emb, dim=-1)
        return emb.detach().cpu()

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
