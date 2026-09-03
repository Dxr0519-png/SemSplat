#!/usr/bin/env python
"""Stage-0 feasibility for the LSeg teacher swap.

Checks, on one real Replica frame:
  1. demo_e200.ckpt loads with no missing/unexpected keys in net.pretrained/net.scratch.
  2. forward is finite; per-call time and peak VRAM at input_long_side=384 (and 480).
  3. LSeg pixel features correlate with OpenAI-CLIP ViT-B/16 patch features at the
     same image location (mean cosine) -> confirms both live in the same shared
     text space, so the existing HF base-CLIP text encoder can stay for queries.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from semsplat.teachers.lseg._load import build_lseg_net  # noqa: E402
from semsplat.teachers.lseg_local_teacher import LSegDenseTeacher  # noqa: E402

IMG = "data/replica_orb_est_ns/images/000015.jpg"
CKPT = "data/checkpoints/lseg/demo_e200.ckpt"


def main() -> None:
    img = cv2.imread(IMG)[:, :, ::-1] / 255.0
    img = torch.from_numpy(np.ascontiguousarray(img)).float().cuda()
    print("frame", IMG, "size", tuple(img.shape[:2]))

    # 1) load report
    net = build_lseg_net(CKPT, device="cuda", fp16=True)

    # 2) per-call time + peak VRAM at two input sizes
    for L in (384, 480):
        t = LSegDenseTeacher(CKPT, input_long_side=L, cell=8, device="cuda", fp16=True)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            grid = t.dense_features(img)  # cold (no cache)
        torch.cuda.synchronize()
        dt = time.time() - t0
        row_norm = torch.norm(grid.float(), dim=-1).mean().item()
        print(f"long_side={L}: grid={tuple(grid.shape)} dtype={grid.dtype} "
              f"cold-forward~{dt*1000:.0f}ms rowL2~{row_norm:.3f}")

    # 3) alignment vs OpenAI-CLIP ViT-B/16 patch teacher (shared-space check)
    from semsplat.teachers.clip_local_teacher import ClipLocalTeacher
    clip_t = ClipLocalTeacher(model_name="openai/clip-vit-base-patch16", fp16=False, device="cuda")
    lseg = LSegDenseTeacher(CKPT, input_long_side=384, cell=1, l2norm=True, device="cuda")
    with torch.no_grad():
        gl = lseg.compute(img, key=1)          # [gH,gW,512], near full-res grid
        gc = clip_t.dense_features(img)        # [gHc,gWc,512] (1/16 patch grid)
    gl2 = F.interpolate(gl.float().permute(2, 0, 1)[None], size=gc.shape[:2], mode="bilinear",
                        align_corners=False)[0].permute(1, 2, 0)
    cos = F.normalize(gl2, dim=-1) @ F.normalize(gc.float(), dim=-1).transpose(-1, -2)
    sim = torch.diagonal(cos, dim1=-2, dim2=-1).mean().item()
    print(f"mean cos(LSeg_pixel, CLIP-B/16_patch) @ same loc = {sim:.3f}   (shared-space signal)")

    print("peak VRAM MB:", torch.cuda.max_memory_allocated() // 2 ** 20)


if __name__ == "__main__":
    main()
