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

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from semsplat.seg_cleanup import remove_small_components


def logit_stability(logits: np.ndarray, offset: float) -> np.ndarray:
    """Official-AMG-style stability score over pre-sigmoid mask logits.

    Fraction of confidently-positive pixels (logit > +offset) among the pixels
    that are at least not-confidently-negative (logit > -offset). A sharp, clean
    mask scores ~1; a soft/noisy decision boundary scores lower.
    """
    inter = (logits > offset).sum(axis=(-1, -2))
    union = (logits > -offset).sum(axis=(-1, -2))
    return inter / np.maximum(union, 1)


def _ctx_window(ys: np.ndarray, xs: np.ndarray, h: int, w: int, ctx_margin: float):
    """Context window around a mask bbox, clipped to the frame.

    Window = bbox expanded by ``ctx_margin * bbox_side`` on every side, clamped to
    the image. A floor/ceiling/wall mask that already spans the frame therefore
    yields (close to) the whole frame -- CLIP keeps the top/bottom/side spatial
    layout -- while a small object keeps a tight object-dominant window.
    """
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    side = max(y1 - y0, x1 - x0)
    mg = int(round(side * ctx_margin))
    Y0, Y1 = max(0, y0 - mg), min(h, y1 + mg)
    X0, X1 = max(0, x0 - mg), min(w, x1 + mg)
    return Y0, Y1, X0, X1


def _ctx_composite(img: np.ndarray, mask: np.ndarray, Y0: int, Y1: int, X0: int,
                   X1: int, blur_sigma: float, darken: float) -> np.ndarray:
    """Mask-sharp on strongly-blurred (or darkened) real background, uint8.

    Mask pixels inside the window keep their true colour; every other pixel of the
    window is either a Gaussian blur of the window (sigma auto-scaled to a 512px
    reference) when ``darken <= 0``, or the window scaled by ``darken``. No neutral
    gray is ever introduced inside the window.
    """
    import cv2
    win = np.clip(img[Y0:Y1, X0:X1] * 255.0, 0, 255).round().astype(np.uint8)
    mm = mask[Y0:Y1, X0:X1]
    out = win.copy()
    if darken > 0:
        bg = win.astype(np.float32) * float(darken)
        out[~mm] = np.clip(bg[~mm], 0, 255).round().astype(np.uint8)
    else:
        chh, cww = win.shape[:2]
        sigma = max(3.0, float(blur_sigma) * max(chh, cww) / 512.0)
        blurred = cv2.GaussianBlur(win, (0, 0), sigmaX=sigma, sigmaY=sigma)
        out[~mm] = blurred[~mm]
    return out


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
        # ---- context prompt guidance (dual-branch CLIP fusion) ----
        context_ratio: float = 0.6,  # context pad = ratio * object bbox side, per side
        context_weight: float = 0.0,  # 0 => disabled (baseline gray-bg object-only path)
        # ---- global-first wall folding (depth-aided), default off ----
        depth_dir: Optional[str] = None,  # dir of <cam_idx>.png uint16 mm depth + parent transforms.json (K)
        fold_enabled: bool = False,       # master switch for plane detection + folding
        fold_tol_m: float = 0.03,         # plane-support residual (m)
        fold_min_plane_frac: float = 0.15,  # dominant plane must cover >= this fraction of valid depth
        fold_max_plane_n: int = 2,        # fit up to this many planes (back + side wall)
        fold_max_area_frac: float = 0.03,  # only fold independent masks <= this frame fraction (protect tv/clock)
        fold_rgb_maxdiff: Optional[float] = 0.15,  # None = pure depth; else veto fold when mask colour differs from its wall ring
        mask_min_comp_px: int = 0,        # drop <min_area speck islands inside a SAM mask before encoding (0 = off)
        # ---- native AMG mask generator (default) ----
        sam_mode: str = "amg",        # 'amg' = native auto-mask grid | 'grid' = legacy 6x6 per-point loop
        points_per_side: int = 8,     # amg: points_per_side^2 grid of single-point prompts
        stability_thr: float = 0.9,   # amg: keep candidates whose logit stability >= thr
        stability_offset: float = 1.0,  # amg: logit offset for the stability score
        stability_big_area_frac: float = 0.08,  # amg: large masks (>= this frame frac) are exempt from the stability gate (soft low-texture planes)
        max_masks: int = 40,          # amg: cap on kept masks after greedy IoU dedup
        mask_overlap: float = 0.85,   # amg: greedy IoU dedup overlap threshold
        amg_long_side: int = 1024,    # amg: SAM working-res long side (grid keeps 512)
        # ---- contextual CLIP crops (default) ----
        crop_mode: str = "ctx",       # 'ctx' = adaptive blur/darken window | 'gray' = legacy mask_bg crop
        ctx_margin: float = 0.25,     # ctx: window expand = margin * bbox side, per side
        ctx_blur_sigma: float = 24.0,  # ctx: gaussian sigma at a 512px window (auto-scaled)
        ctx_darken: float = 0.0,      # ctx: >0 darkens the non-mask bg by this factor instead of blurring
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

        # context branch
        self.context_ratio = float(context_ratio)
        self.context_weight = float(context_weight)

        # native AMG mask generator + contextual CLIP crops
        self.sam_mode = sam_mode
        self.points_per_side = int(points_per_side)
        self.stability_thr = float(stability_thr)
        self.stability_offset = float(stability_offset)
        self.stability_big_area_frac = float(stability_big_area_frac)
        self.max_masks = int(max_masks)
        self.mask_overlap = float(mask_overlap)
        self._sam_long_side = int(amg_long_side) if sam_mode == "amg" else 512
        self.crop_mode = crop_mode
        self.ctx_margin = float(ctx_margin)
        self.ctx_blur_sigma = float(ctx_blur_sigma)
        self.ctx_darken = float(ctx_darken)

        # global-first wall folding
        self.mask_min_comp_px = int(mask_min_comp_px)
        self.fold_enabled = bool(fold_enabled)
        self.fold_tol_m = float(fold_tol_m)
        self.fold_min_plane_frac = float(fold_min_plane_frac)
        self.fold_max_plane_n = int(fold_max_plane_n)
        self.fold_max_area_frac = float(fold_max_area_frac)
        self.fold_rgb_maxdiff = None if fold_rgb_maxdiff is None else float(fold_rgb_maxdiff)
        self._depth_dir = Path(depth_dir) if depth_dir else None
        self._K = None
        self._K_missing = False
        self._no_open3d = False
        self._warned_shape = False
        self._depth_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._last_n_folded = 0
        if self._depth_dir is not None and not (self._depth_dir.parent / "transforms.json").exists():
            print(f"[SamClipTeacher] depth_dir {self._depth_dir} has no parent transforms.json (needed for K); "
                  "global-first wall folding disabled")
            self._depth_dir = None

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
        if self.sam_mode == "amg":
            return self._sam_masks_amg(img_np)
        return self._sam_masks_grid(img_np)

    @torch.no_grad()
    def _sam_masks_grid(self, img_np) -> list:
        """Legacy 6x6 single-point loop (kept byte-identical for A/B repro)."""
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

    @torch.no_grad()
    def _sam_masks_amg(self, img_np) -> list:
        """Native automatic-mask generator (official-AMG style), one batched forward.

        A dense ``points_per_side^2`` grid of single-point prompts is fed as
        separate prompt groups in one ``SamModel`` call -> per point, 3 candidate
        masks + raw logits. Every point keeps at most one candidate (best of its 3
        by predicted IoU) that passes the predicted-IoU, logit-stability and area
        filters; the pool is then greedily deduped by IoU exactly like the grid
        path. Returns bool HxW masks in the input image frame.
        """
        h, w = img_np.shape[:2]
        step = self.points_per_side
        groups = []  # each grid point = its own single-point prompt group
        for i in range(step):
            for j in range(step):
                x = int((j + 0.5) * w / step)
                y = int((i + 0.5) * h / step)
                groups.append([[x, y]])
        inp = self._proc(img_np, input_points=[groups], return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].to(next(self._sam.parameters()).dtype)
        out = self._sam(**inp)
        iou = out.iou_scores.squeeze(0).float().cpu().numpy()      # [N,3] sigmoid
        log = out.pred_masks.squeeze(0).float().cpu().numpy()      # [N,3,256,256] logits
        ups = self._proc.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu(),
        )[0].numpy()  # [N,3,Hw,Ww] bool
        stab = logit_stability(log, self.stability_offset)          # [N,3]
        cand, scores = [], []
        for p in range(iou.shape[0]):
            for c in np.argsort(-iou[p]):  # best-of-3 first, keep the first pass
                if iou[p, c] < self.iou_thr:
                    continue
                frac = ups[p, c].mean()
                if not (self.min_area_frac < frac < self.max_area_frac):
                    continue
                # large soft low-texture planes (floor/wall/ceiling) have blurry
                # boundaries -> low stability; exempt them so stuff coverage is kept.
                if stab[p, c] < self.stability_thr and frac < self.stability_big_area_frac:
                    continue
                cand.append(ups[p, c])
                scores.append(float(iou[p, c]))
                break
        keep = []
        for m, s in sorted(zip(cand, scores), key=lambda t: -t[1]):
            if all(self._iou(m, km) < self.mask_overlap for km in keep):
                keep.append(m)
            if len(keep) >= self.max_masks:
                break
        del iou, log, ups, stab
        return keep

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / max(union, 1)

    @torch.no_grad()
    def _object_emb(self, img_np, masks) -> list:
        """CLIP vector per whole-object mask.

        crop_mode='gray' keeps the legacy object-on-constant-background crop
        (``mask_bg``), byte-for-byte. crop_mode='ctx' (default) builds an adaptive
        contextual crop: the window is the mask bbox expanded by ``ctx_margin``
        (clipped to the frame), mask pixels stay sharp and everything else in the
        window is strongly blurred (or darkened when ``ctx_darken > 0``). Big
        stuff (floor/ceiling/wall) thus fills (most of) the frame, restoring the
        top/bottom/side spatial layout CLIP needs; small objects keep scale.

        Dual-branch context guidance when ``self.context_weight > 0``: besides the
        main crop, a second crop keeps a ``context_ratio`` band of *real*
        surroundings around the bbox and the two CLIP vectors are fused
        ``v = (1-w)*main + w*context``. ``w == 0`` disables it.
        """
        import cv2

        h, w = img_np.shape[:2]
        ctx_on = self.context_weight > 0.0
        crops, ctx = [], []
        for m in masks:
            ys, xs = np.where(m)
            if self.crop_mode == "gray":
                # ---- legacy gray-background object crop ----
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
            else:
                # ---- contextual crop: sharp mask on blurred/darkened real bg ----
                Y0, Y1, X0, X1 = _ctx_window(ys, xs, h, w, self.ctx_margin)
                comp = _ctx_composite(img_np, m, Y0, Y1, X0, X1,
                                      self.ctx_blur_sigma, self.ctx_darken)
                chh, cww = comp.shape[:2]
                if chh != cww:
                    side3 = max(chh, cww)
                    top, left = (side3 - chh) // 2, (side3 - cww) // 2
                    # square via replicate border only -- never neutral gray inside
                    comp = cv2.copyMakeBorder(comp, top, side3 - chh - top, left,
                                              side3 - cww - left, cv2.BORDER_REPLICATE)
                crops.append(cv2.resize(comp, (224, 224), interpolation=cv2.INTER_LINEAR))
            if ctx_on:
                # context branch: keep real background around the bbox (no gray fill).
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                side = max(y1 - y0, x1 - x0)
                pc = int(round(side * self.context_ratio))
                Y0c, Y1c = max(0, y0 - pc), min(h, y1 + pc)
                X0c, X1c = max(0, x0 - pc), min(w, x1 + pc)
                c = np.clip(img_np[Y0c:Y1c, X0c:X1c] * 255.0, 0, 255).round().astype(np.uint8)
                chh, cww = c.shape[:2]
                if chh != cww:
                    side3 = max(chh, cww)
                    top, left = (side3 - chh) // 2, (side3 - cww) // 2
                    # replicate edge pixels to a square -- never neutral gray, which
                    # would reintroduce the background bias this branch removes.
                    c = cv2.copyMakeBorder(c, top, side3 - chh - top, left, side3 - cww - left,
                                           cv2.BORDER_REPLICATE)
                ctx.append(cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR))

        def _encode(arr) -> torch.Tensor:
            x = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).to(self.device).float() / 255.0
            x = (x - self._mean[None, :, None, None]) / self._std[None, :, None, None]
            dtype = next(self._clip.parameters()).dtype
            out = self._clip.vision_model(pixel_values=x.to(dtype))
            emb = self._clip.visual_projection(out.pooler_output.to(dtype)).float()
            return F.normalize(emb, dim=-1).detach().cpu()  # [K,512]

        emb = _encode(crops)
        if ctx_on and len(ctx):
            emb_ctx = _encode(ctx)
            emb = F.normalize((1.0 - self.context_weight) * emb + self.context_weight * emb_ctx, dim=-1)
        return emb

    # --------------------------------------------------------- depth / planes
    def _depth_K(self) -> Optional[np.ndarray]:
        """3x3 intrinsics, read once from the transforms.json next to depth_dir."""
        if self._K is not None or self._K_missing:
            return self._K
        try:
            t = json.loads((self._depth_dir.parent / "transforms.json").read_text())
            fx = float(t["fl_x"])
            fy = float(t.get("fl_y", fx))
            cx = float(t.get("cx", t.get("w", 640) / 2))
            cy = float(t.get("cy", t.get("h", 480) / 2))
            self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
        except Exception as e:  # missing/malformed transforms -> folding stays off
            if not self._warned_shape:
                print(f"[SamClipTeacher] could not read K from transforms.json: {e}; folding disabled")
                self._warned_shape = True
            self._K_missing = True
        return self._K

    def _load_depth(self, key: int) -> Optional[np.ndarray]:
        """uint16-mm radial depth for cam ``key`` -> float32 m (H,W), LRU-cached."""
        import cv2
        if self._depth_dir is None:
            return None
        if key in self._depth_cache:
            self._depth_cache.move_to_end(key)
            return self._depth_cache[key]
        d = None
        for name in (f"{int(key):06d}.png", f"{int(key):04d}.png", f"{int(key)}.png"):
            p = self._depth_dir / name
            if not p.exists():
                continue
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            if raw.dtype != np.uint16:
                print(f"[SamClipTeacher] depth {p} is {raw.dtype}, expected uint16 -- frame {key} skipped")
                return None
            d = raw.astype(np.float32) * 0.001  # mm -> m (radial)
            break
        if d is None:
            return None
        self._depth_cache[key] = d
        self._depth_cache.move_to_end(key)
        while len(self._depth_cache) > self._cache_max:
            self._depth_cache.popitem(last=False)
        return d

    def _global_planes(self, depth: np.ndarray, H: int, W: int) -> list:
        """Fit up to ``fold_max_plane_n`` dominant depth planes -> bool HxW supports.

        Valid depth (0.2..8 m) is back-projected to the camera frame with the repo's
        radial-depth convention X=(u-cx)/fx*d, Y=-(v-cy)/fy*d, Z=-d, sampled on a
        stride-4 grid, then RANSAC via open3d. Each accepted plane must support
        >= ``fold_min_plane_frac`` of the valid depth. No wall/floor/ceiling
        classification: RANSAC takes whatever plane dominates the view and folding
        is surface-agnostic (floor/ceiling fragments fold too).
        """
        if self._no_open3d:
            return []
        K = self._depth_K()
        if K is None:
            return []
        valid = (depth > 0.2) & (depth < 8.0)
        ys4, xs4 = np.nonzero(valid[::4, ::4])
        if ys4.size < 50:
            return []
        ys4 = ys4 * 4 + 2
        xs4 = xs4 * 4 + 2
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        dd = depth[ys4, xs4]
        pts = np.stack([(xs4 - cx) * dd / fx, (ys4 - cy) * dd / fy, -dd], axis=1).astype(np.float64)
        try:
            import open3d as o3d
        except Exception as e:
            self._no_open3d = True
            if not self._warned_shape:
                print(f"[SamClipTeacher] open3d unavailable ({e}); wall folding disabled")
                self._warned_shape = True
            return []
        # full-res valid-depth coordinates -> exact per-plane support masks
        vy, vx = np.nonzero(valid)
        vd = depth[vy, vx]
        vX = (vx - cx) * vd / fx
        vY = (vy - cy) * vd / fy
        vZ = -vd

        remaining = np.ones(len(pts), bool)
        supports = []
        for _ in range(self.fold_max_plane_n):
            if int(remaining.sum()) < 30:
                break
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[remaining])
            plane_m, _inl = pcd.segment_plane(
                distance_threshold=self.fold_tol_m, ransac_n=3, num_iterations=1000)
            a, b, c, e = (float(v) for v in plane_m)
            denom = math.sqrt(a * a + b * b + c * c)
            d_pl = np.abs(a * vX + b * vY + c * vZ + e) / denom
            near = d_pl <= self.fold_tol_m
            if near.sum() < max(200, int(self.fold_min_plane_frac * vd.size)):
                break
            sup = np.zeros((H, W), bool)
            sup[vy[near], vx[near]] = True
            supports.append(sup)
            # drop sampled points on this plane before fitting the next one
            d_pts = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + e) / denom
            remaining &= d_pts > self.fold_tol_m
        return supports

    def _albedo_ok(self, mask: np.ndarray, support: np.ndarray, img: np.ndarray) -> bool:
        """True => candidate may be folded (its mean colour matches the surrounding ring)."""
        if self.fold_rgb_maxdiff is None:
            return True
        import cv2
        m = (np.asarray(mask) > 0).astype(np.uint8)
        ring = cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0
        ring &= (m == 0) & support
        ys, xs = np.where(ring)
        if ys.size < 30:  # too little wall ring visible -> conservative keep
            return False
        if ys.size > 400:
            sel = np.linspace(0, ys.size - 1, 400).astype(int)
            ys, xs = ys[sel], xs[sel]
        wall_mean = img[ys, xs].reshape(-1, 3).mean(0)
        my, mx = np.where(m > 0)
        if my.size > 1000:
            sel = np.linspace(0, my.size - 1, 1000).astype(int)
            my, mx = my[sel], mx[sel]
        obj_mean = img[my, mx].reshape(-1, 3).mean(0)
        return float(np.abs(obj_mean - wall_mean).mean()) <= self.fold_rgb_maxdiff

    def _fold_wall_fragments(self, masks, supports, img, H: int, W: int) -> list:
        """Global-first: collapse coplanar fragments of a dominant plane into plane
        "wall" entities; genuinely separate masks (non-coplanar, colour-distinct or
        big, e.g. a flush tv-screen) are kept and carved out of the entities so they
        never receive wall supervision. Returns the pool of masks to CLIP-encode.
        """
        if not supports:
            return masks
        union = np.zeros((H, W), bool)
        for s in supports:
            union |= s
        if not union.any():
            return masks
        max_px = self.fold_max_area_frac * H * W
        kept, n_folded = [], 0
        for m in masks:
            m = np.asarray(m) > 0
            n = int(m.sum())
            if (n and float((m & union).sum()) / n >= 0.8 and n <= max_px
                    and self._albedo_ok(m, union, img)):
                n_folded += 1  # flush fragment -> folds into the plane identity
            else:
                kept.append(m)
        self._last_n_folded = n_folded
        # entity = plane support minus every kept object, so the protected stuff
        # (tv-screen, clock, protruding furniture) never receives plane identity.
        carve = np.zeros((H, W), bool)
        for m in kept:
            carve |= m
        entities = [s & ~carve for s in supports]
        entities = [e for e in entities if e.any()]
        if n_folded == 0 and not entities:
            return masks
        return kept + entities

    # ------------------------------------------------------------- grid
    @torch.no_grad()
    def _dense(self, image: torch.Tensor, key: Optional[int] = None):
        img = image.detach().cpu().numpy()
        H, W = img.shape[:2]
        # SAM on a downscaled image (faster), masks upsampled to full size
        import cv2
        # Downscale only when the frame exceeds the working long side (512 for the
        # legacy grid path -- byte-comparable; amg_long_side otherwise, default
        # 1024 => native pixels for 640x480 office0). Never upscale.
        scale = min(1.0, self._sam_long_side / max(H, W))
        img_small = cv2.resize(np.clip(img, 0, 1), (max(1, int(W * scale)), max(1, int(H * scale)))) if scale < 1 else img
        rgb = (np.clip(img_small, 0, 1) * 255).round().astype(np.uint8)
        masks_small = self._sam_masks(rgb)
        masks = [cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for m in masks_small]
        if not masks:
            # no confident object -> single full-image "background" mask keeps loss meaningful
            masks = [np.ones((H, W), bool)]
        # Item1 hook: excise sub-min_area speck islands inside a candidate SAM mask
        if self.mask_min_comp_px > 1:
            cleaned = []
            for m in masks:
                c = remove_small_components(m, self.mask_min_comp_px)
                if c.any():
                    cleaned.append(c)
            if cleaned:
                masks = cleaned
        if not masks:
            masks = [np.ones((H, W), bool)]
        # Item3 hook: global-first wall folding (default off; needs per-cam depth)
        if self.fold_enabled and self._depth_dir is not None and key is not None:
            depth = self._load_depth(key)
            if depth is None:
                pass  # frame has no depth file -> masks unchanged, gracefully no fold
            elif depth.shape != (H, W):
                if not self._warned_shape:
                    print(f"[SamClipTeacher] depth {depth.shape} != image {(H, W)}; folding skipped "
                          "(expected num_downscales=0)")
                    self._warned_shape = True
            else:
                supports = self._global_planes(depth, H, W)
                if supports:
                    masks = self._fold_wall_fragments(masks, supports, img, H, W)
        emb = self._object_emb(img, masks)  # [K,512]
        # Pool per-mask CLIP vectors straight into the grid (no [H,W,512] dense map).
        gH, gW = max(1, math.ceil(H / self.cell)), max(1, math.ceil(W / self.cell))
        cell_px = (H * W) / float(gH * gW)  # approx pixels per grid cell (for coverage)
        sums = np.zeros((gH, gW, 512), np.float32)
        cnts = np.zeros((gH, gW), np.float32)
        emb_np = emb.numpy()
        for m, e in zip(masks, emb_np):
            ys, xs = np.where(m)
            if ys.size == 0:
                continue
            cy = np.minimum(ys // self.cell, gH - 1)
            cx = np.minimum(xs // self.cell, gW - 1)
            np.add.at(sums, (cy, cx), e)
            np.add.at(cnts, (cy, cx), 1.0)
        ok = cnts > 0
        pooled = np.zeros((gH, gW, 512), np.float32)
        pooled[ok] = sums[ok] / cnts[ok, None]
        cover = np.minimum(cnts / cell_px, 1.0)
        tp = torch.from_numpy(pooled).to(self.device)
        tp = F.normalize(tp, dim=-1)
        cp = torch.from_numpy(cover).to(self.device)
        del sums, cnts, emb_np
        return tp, cp

    def dense_features(self, image: torch.Tensor) -> torch.Tensor:
        return self._dense(image)[0]

    def compute(self, image: torch.Tensor, key: Optional[int] = None) -> torch.Tensor:
        if key is not None and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        grid, cover = self._dense(image, key)
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
        grid, cover = self._dense(image, key)
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
        self._depth_cache.clear()
