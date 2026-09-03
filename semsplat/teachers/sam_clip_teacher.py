"""SAM object-mask + CLIP-crop teacher (whole-object isolation).

Upgrade over SuperClipTeacher: masks come from SAM-B instance proposals rather
than SLIC colour blobs, so each region is an entire object. The object is
cropped out on a neutral background (mask applied, background painted a constant
gray by default; black is also selectable) and CLIP-encoded -> every pixel of
that object points at an *object-local* identity vector. Background pixels that
belong to no object are left unassigned (coverage=False) so they are never
forced into a wrong class.

Same duck-type contract as the other teachers, plus an optional coverage map:

    teacher.dim                              == 512
    teacher.compute(image, key)              -> [gH, gW, 512]  (fp16 LRU)
    teacher.coverage(image, key) (optional)  -> [gH, gW]  float 0..1 (cell coverage)
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class SamClipTeacher:
    def __init__(
        self,
        image_model_name: str = "openai/clip-vit-base-patch16",
        sam_model_name: str = "facebook/sam-vit-base",
        grid: int = 6,               # grid x grid point prompts
        iou_thr: float = 0.86,       # keep masks whose predicted IoU >= thr
        min_area_frac: float = 0.02,
        max_area_frac: float = 0.95,
        mask_bg: float = 0.5,        # background value in 0..1 (0.5 neutral gray)
        cell: int = 16,
        fp16: bool = True,
        max_entries: int = 2000,
        device: str = "cuda",
    ):
        from semsplat.teachers.clip_local_teacher import CLIP_MEAN, CLIP_STD, load_clip

        self.sam_model_name = sam_model_name
        self.grid = int(grid)
        self.iou_thr = float(iou_thr)
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.mask_bg = float(mask_bg)
        self.cell = int(cell)
        self.fp16 = fp16
        self.device = torch.device(device)
        self.dim = 512
        self._cache: "OrderedDict[int, tuple]" = OrderedDict()
        self._cache_max = int(max_entries)

        from transformers import SamModel, SamProcessor
        self._sam = SamModel.from_pretrained(sam_model_name).to(self.device).eval()
        if fp16:
            self._sam = self._sam.half()
        self._proc = SamProcessor.from_pretrained(sam_model_name)
        for p in self._sam.parameters():
            p.requires_grad_(False)

        self._clip = load_clip(image_model_name, fp16=fp16, device=str(self.device)).eval()
        self._mean = CLIP_MEAN.to(self.device)
        self._std = CLIP_STD.to(self.device)

    # --------------------------------------------------------------- masks
    @torch.no_grad()
    def _sam_masks(self, img_np) -> list:
        """Whole-object binary masks (bool HxW) from grid-point SAM proposals."""
        h, w = img_np.shape[:2]
        step = self.grid
        pts = []
        for i in range(step):
            for j in range(step):
                x = int((j + 0.5) * w / step)
                y = int((i + 0.5) * h / step)
                pts.append([x, y])
        masks, scores = [], []
        # SAM forward per point (HF API treats all points as one prompt group)
        for (x, y) in pts:
            inp = self._proc(img_np, input_points=[[[x, y]]], return_tensors="pt").to(self.device)
            inp["pixel_values"] = inp["pixel_values"].to(next(self._sam.parameters()).dtype)
            out = self._sam(**inp)
            iou = float(out.iou_scores[0, 0, 0].float())
            if iou < self.iou_thr:
                continue
            post = self._proc.image_processor.post_process_masks(
                out.pred_masks.cpu(),
                inp["original_sizes"].cpu(),
                inp["reshaped_input_sizes"].cpu(),
            )[0][0]  # [3,H,W] bool
            mask = post[0].numpy()  # highest-quality level
            frac = mask.mean()
            if not (self.min_area_frac < frac < self.max_area_frac):
                continue
            masks.append(mask)
            scores.append(iou)
        # greedy dedupe by IoU so one object is not counted many times
        keep = []
        for m, s in sorted(zip(masks, scores), key=lambda t: -t[1]):
            if all(self._iou(m, km) < 0.85 for km in keep):
                keep.append(m)
            if len(keep) >= 40:
                break
        return keep

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / max(union, 1)

    @torch.no_grad()
    def _object_emb(self, img_np, masks) -> list:
        """CLIP vector per whole-object mask (object shown on neutral background)."""
        import cv2

        h, w = img_np.shape[:2]
        crops = []
        for m in masks:
            ys, xs = np.where(m)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            side = max(y1 - y0, x1 - x0)
            pad = int(round(side * 0.06))
            Y0, Y1 = max(0, y0 - pad), min(h, y1 + pad)
            X0, X1 = max(0, x0 - pad), min(w, x1 + pad)
            crop = np.full((Y1 - Y0, X1 - X0, 3), self.mask_bg, np.float32)
            mm = m[Y0:Y1, X0:X1]
            crop[mm] = img_np[Y0:Y1, X0:X1][mm]
            side2 = max(crop.shape[:2])
            canvas = np.full((side2, side2, 3), int(self.mask_bg * 255), np.uint8)
            ch, cw = crop.shape[:2]
            cy, cx = (side2 - ch) // 2, (side2 - cw) // 2
            canvas[cy : cy + ch, cx : cx + cw] = np.clip(crop * 255, 0, 255).astype(np.uint8)
            crops.append(cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_LINEAR))
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(self.device).float() / 255.0
        x = (x - self._mean[None, :, None, None]) / self._std[None, :, None, None]
        dtype = next(self._clip.parameters()).dtype
        out = self._clip.vision_model(pixel_values=x.to(dtype))
        emb = self._clip.visual_projection(out.pooler_output.to(dtype)).float()
        return F.normalize(emb, dim=-1).detach().cpu()  # [K,512]

    # ------------------------------------------------------------- grid
    @torch.no_grad()
    def _dense(self, image: torch.Tensor):
        img = image.detach().cpu().numpy()
        H, W = img.shape[:2]
        # SAM on a downscaled image (faster), masks upsampled to full size
        import cv2
        scale = 512 / max(H, W)
        img_small = cv2.resize(np.clip(img, 0, 1), (max(1, int(W * scale)), max(1, int(H * scale)))) if scale < 1 else img
        rgb = (np.clip(img_small, 0, 1) * 255).round().astype(np.uint8)
        masks_small = self._sam_masks(rgb)
        import cv2
        masks = [cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for m in masks_small]
        if not masks:
            # no confident object -> single full-image "background" mask keeps loss meaningful
            masks = [np.ones((H, W), bool)]
        emb = self._object_emb(img, masks)  # [K,512]
        target = np.zeros((H, W, 512), np.float32)
        cover = np.zeros((H, W), np.float32)
        for k, m in enumerate(masks):
            target[m] = emb[k].numpy()
            cover[m] = 1.0
        gH, gW = max(1, math.ceil(H / self.cell)), max(1, math.ceil(W / self.cell))
        t = torch.from_numpy(target).permute(2, 0, 1)[None].to(self.device)
        c = torch.from_numpy(cover)[None, None].to(self.device)
        tp = F.adaptive_avg_pool2d(t, (gH, gW))[0].permute(1, 2, 0)  # [gH,gW,512]
        cp = F.adaptive_avg_pool2d(c, (gH, gW))[0, 0]  # [gH,gW]
        tp = F.normalize(tp.float(), dim=-1)
        return tp, cp

    def dense_features(self, image: torch.Tensor) -> torch.Tensor:
        return self._dense(image)[0]

    def compute(self, image: torch.Tensor, key: Optional[int] = None) -> torch.Tensor:
        if key is not None and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        grid, cover = self._dense(image)
        if key is not None:
            if self.fp16:
                grid = grid.half()
                cover = cover.half()
            self._cache[key] = (grid, cover)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return grid

    def coverage(self, image: torch.Tensor, key: Optional[int] = None) -> torch.Tensor:
        if key is not None and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][1]
        grid, cover = self._dense(image)
        if key is not None:
            if self.fp16:
                grid, cover = grid.half(), cover.half()
            self._cache[key] = (grid, cover)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return cover

    def clear_cache(self) -> None:
        self._cache.clear()
