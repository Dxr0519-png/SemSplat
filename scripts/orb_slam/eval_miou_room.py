#!/usr/bin/env python
"""Closed-set mIoU vs Replica room/office per-frame semantic GT.

GT maps (semantic_class/*.png) store Replica *class ids* (see the scene's
info_semantic.json 'classes'). We render our open-vocab argmax over ``prompts``,
map prompt index i -> GT class id gt_ids[i], and compute per-class IoU over
pixels with valid GT (id != 0) and valid alpha.

Usage:
    eval_miou_room.py --config <cfg.yml> --ns <ns_dir> --gt-dir data/room0_est/frames/semantic \
        --prompts "sofa with cushions,black office chair,table,office desk,potted green plant,plain room wall,carpeted floor,flat white ceiling" \
        --gt-ids 76 20 80 34 44 93 40 31 --frames 0 10 20 ...
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
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup
from semsplat.inference import scoring
from semsplat.teachers.clip_local_teacher import encode_text_prompts, resolve_text_model_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ns", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True, help="dir of <k>.png GT class-id maps")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--gt-ids", type=int, nargs="+", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True, help="dataset image indices (0..N-1)")
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/room0_miou.txt"))
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    assert len(prompts) == len(args.gt_ids)
    C = len(prompts)
    meta = json.loads((args.ns / "transforms.json").read_text())
    byf = {int(Path(f["file_path"]).stem): f for f in meta["frames"]}

    _, pipeline, _, _ = eval_setup(args.config)
    model = pipeline.model
    model.eval()
    text = resolve_text_model_name(model.config)
    emb = torch.stack([encode_text_prompts(prompts, text)[p] for p in prompts], dim=0).to(model.device)

    tp = np.zeros(C); fp = np.zeros(C); fn = np.zeros(C)
    for k in args.frames:
        f = byf[k]
        M = np.asarray(f["transform_matrix"], float)[:3, :4]
        cam = Cameras(camera_to_worlds=torch.tensor(M)[None].float().to(model.device),
                      fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                      width=meta["w"], height=meta["h"]).to(model.device)
        maps, alpha = scoring.semantic_maps_from_prompts(model, cam, emb)
        maps = maps.detach().cpu().numpy()
        alpha = alpha.detach().cpu().numpy()
        valid = alpha[..., 0] > 0.5
        pred = maps.argmax(-1)
        pred = pred[valid]
        gtf = args.gt_dir / f"{k}.png"
        if not gtf.exists():
            gtf = args.gt_dir / f"{k:06d}.png"
        gt = cv2.imread(str(gtf), -1)
        gt = gt[valid].ravel()
        gt[gt == 0] = -1  # void -> ignore (never equals any target id)
        for i, gid in enumerate(args.gt_ids):
            p = pred == i
            g = gt == gid
            tp[i] += int((p & g).sum())
            fp[i] += int((p & (gt != gid) & (gt != -1)).sum())
            fn[i] += int((g & ~p).sum())

    ious = {prompts[i]: (tp[i] / max(tp[i] + fp[i] + fn[i], 1)) for i in range(C)}
    present = [i for i in range(C) if (tp[i] + fn[i]) > 0]
    miou = float(np.mean([ious[prompts[i]] for i in present])) if present else 0.0
    lines = ["metric  IoU"]
    for i in range(C):
        lines.append(f"{prompts[i]:24s} {ious[prompts[i]]:.3f}")
    lines.append(f"{'mIoU (classes present in GT)':24s} {miou:.3f}   classes_present={len(present)}/{C}")
    txt = "\n".join(lines) + "\n"
    args.out.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
