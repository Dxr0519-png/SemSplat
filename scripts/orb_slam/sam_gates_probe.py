#!/usr/bin/env python
"""SAM-AMG gate probe: old vs tightened gate sets on office0 frames.

Cheap (GPU, teacher-only, no training) pre-check before committing a 20k run to
the tightened SAM gates. For a handful of frames we run the exact AMG front-end
the samclip teacher uses (``SamClipTeacher._sam_masks`` after the same 1024
long-side downscale as ``object_semantics``) under two gate sets and report:

  n         # masks kept
  cov       pixel fraction covered by >=1 kept mask
  max_frac  area fraction of the largest kept mask
  n_big     # kept masks with area >= 8% of the frame  (the tier that is
            *exempt* from the stability gate -- floors/walls/ceilings)
  big_cov   pixel fraction covered by the >=8% masks (proxy for plane coverage)

Purpose: confirm the tightened gates (esp. iou_thr 0.86->0.92, which has NO
big-area exemption) do not silently kill the whole-wall/floor/ceiling plane
masks. If plane coverage collapses (big_cov -> ~0) the gates are too aggressive
and the iou raise should be scaled back before burning a 20k training.

Usage:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/orb_slam/sam_gates_probe.py \
      --ns data/office0_est320_ns [--frames 0 60 150 220 290] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# old = reg-baseline gates (0.86 / 0.90 / 0.85), new = tightened (0.92/0.97/0.5)
GATE_SETS = {
    "old": dict(iou_thr=0.86, stability_thr=0.90, mask_overlap=0.85),
    "new": dict(iou_thr=0.92, stability_thr=0.97, mask_overlap=0.50),
    "mid": dict(iou_thr=0.88, stability_thr=0.97, mask_overlap=0.50),
}
BIG_FRAC = 0.08  # stability-big-area exemption threshold used by the teacher


def stats(masks: list, H: int, W: int) -> dict:
    n = len(masks)
    if n == 0:
        return dict(n=0, cov=0.0, max_frac=0.0, n_big=0, big_cov=0.0)
    areas = np.array([m.mean() for m in masks])
    union = np.zeros((H, W), bool)
    for m in masks:
        union |= m
    big = areas >= BIG_FRAC
    big_union = np.zeros((H, W), bool)
    for m in masks:
        if m.mean() >= BIG_FRAC:
            big_union |= m
    return dict(
        n=n,
        cov=float(union.mean()),
        max_frac=float(areas.max()),
        n_big=int(big.sum()),
        big_cov=float(big_union.mean()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=Path, default=Path("data/office0_est320_ns"))
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 60, 150, 220, 290])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    meta = json.loads((args.ns / "transforms.json").read_text())
    imgs = {int(Path(f["file_path"]).stem): args.ns / f["file_path"] for f in meta["frames"]}
    missing = [k for k in args.frames if k not in imgs]
    if missing:
        raise SystemExit(f"missing frames in dataset: {missing}")

    from semsplat.teachers.sam_clip_teacher import SamClipTeacher

    teacher = SamClipTeacher(
        image_model_name="openai/clip-vit-base-patch16",
        sam_model_name="facebook/sam-vit-base",
        sam_mode="amg",
        points_per_side=8,
        crop_mode="ctx",
        fp16=True,
        device=args.device,
    )
    teacher._sam_long_side = 1024  # mirror object_semantics default (native here)

    rows = []
    for k in args.frames:
        bgr = cv2.imread(str(imgs[k]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        scale = min(1.0, teacher._sam_long_side / max(H, W))
        if scale < 1.0:
            rgb = cv2.resize(rgb, (max(1, int(W * scale)), max(1, int(H * scale))))
        for name, gates in GATE_SETS.items():
            teacher.iou_thr = gates["iou_thr"]
            teacher.stability_thr = gates["stability_thr"]
            teacher.mask_overlap = gates["mask_overlap"]
            masks = teacher._sam_masks(rgb)
            s = stats(masks, H, W)
            rows.append((k, name, s))
            print(f"frame {k:3d} {name:3s}  n={s['n']:2d} cov={s['cov']:.3f} "
                  f"max_frac={s['max_frac']:.3f} n_big={s['n_big']:2d} big_cov={s['big_cov']:.3f}")

    # plane-survival verdict: mean big_cov under each gate set
    for name in GATE_SETS:
        bc = np.mean([r[2]["big_cov"] for r in rows if r[1] == name])
        cov = np.mean([r[2]["cov"] for r in rows if r[1] == name])
        n = np.mean([r[2]["n"] for r in rows if r[1] == name])
        print(f"[{name}] mean n={n:.1f} mean cov={cov:.3f} mean big_cov={bc:.3f}")


if __name__ == "__main__":
    main()
