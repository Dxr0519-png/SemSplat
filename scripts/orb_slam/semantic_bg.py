#!/usr/bin/env python
"""Background-aware open-vocab semantic maps (no retrain).

Renders per-pixel class scores (reusing semsplat scoring), then marks a pixel as
*background* when either
  - a background prompt text wins (--bg-prompt), or
  - its best-class confidence is too low (--top-score-thr), or
  - the best-vs-second margin is too small (--margin-thr).
Background is drawn dark-gray (distinct from the unlit 40-gray used for no-alpha).

Outputs per frame: *_sem.png (class map with optional background) + *_scores.npz +
montage [GT ref | map] in --out-dir; plus a stdout per-frame background-% report.
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

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from semsplat.inference import scoring
from semsplat.teachers.clip_local_teacher import encode_text_prompts, resolve_text_model_name


def _palette(n: int) -> np.ndarray:
    import colorsys

    cols = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.9)
        cols.append([int(255 * r), int(255 * g), int(255 * b)])
    return np.asarray(cols, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ns", type=Path, required=True, help="ns dataset dir (poses)")
    ap.add_argument("--rgb-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[15, 60, 120, 200, 280])
    ap.add_argument("--prompts", default="sofa with cushions,black office chair,wooden coffee table,office desk,potted green plant,plain room wall,carpeted floor,flat white ceiling")
    ap.add_argument("--bg-prompt", default=None, help="extra background text prompt")
    ap.add_argument("--top-score-thr", type=float, default=None)
    ap.add_argument("--margin-thr", type=float, default=None)
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    cols = _palette(len(prompts))
    meta = json.loads((args.ns / "transforms.json").read_text())
    byf = {f["file_path"]: f for f in meta["frames"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, pipeline, _, _ = eval_setup(args.config)
    model = pipeline.model
    model.eval()
    text = resolve_text_model_name(model.config)
    embd = torch.stack([encode_text_prompts(prompts, text)[p] for p in prompts], dim=0).to(model.device)
    if args.bg_prompt:
        bg_emb = encode_text_prompts([args.bg_prompt], text)[args.bg_prompt].to(model.device)
        bg_index = len(prompts)  # extra column

    BG = np.array([28, 28, 28], np.uint8)  # background marker colour
    report = {}
    for idx in args.frames:
        f = byf[f"images/{idx:06d}.jpg"]
        M = np.asarray(f["transform_matrix"], float)[:3, :4]
        cam = Cameras(camera_to_worlds=torch.tensor(M)[None].float().to(model.device),
                      fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                      width=meta["w"], height=meta["h"]).to(model.device)
        maps, alpha = scoring.semantic_maps_from_prompts(model, cam, embd)  # [H,W,C]
        maps = maps.detach().cpu().numpy()
        alpha = alpha.detach().cpu().numpy()
        if maps.ndim == 4:
            maps, alpha = maps[0], alpha[0]
        valid = alpha[..., 0] > 0.5
        nbase = len(prompts)
        if args.bg_prompt:
            # re-run with the background column appended and use that for argmax
            full, _ = scoring.semantic_maps_from_prompts(model, cam, torch.cat([embd, bg_emb[None]], 0))
            maps = full.detach().cpu().numpy()
            if maps.ndim == 4:
                maps = maps[0]
        H, W = maps.shape[:2]
        # top-2 via partition (no fancy-index broadcast blowups)
        top2 = np.partition(maps, -2, axis=-1)
        top_score = top2[..., -1]
        sec = top2[..., -2]
        margin = top_score - sec
        top_class = maps[..., :nbase].argmax(-1)  # background column (nbase) handled separately

        bgmask = ~valid
        if args.bg_prompt:
            bgmask = bgmask | (maps[..., nbase] > maps[..., :nbase].max(-1))
        if args.top_score_thr is not None:
            bgmask = bgmask | (top_score < args.top_score_thr)
        if args.margin_thr is not None:
            bgmask = bgmask | (margin < args.margin_thr)

        arg = top_class.copy()
        arg[bgmask] = 255
        rgb = np.zeros((H, W, 3), np.uint8)
        for c in range(nbase):
            rgb[arg == c] = cols[c]
        rgb[arg == 255] = BG
        rgb[~valid] = 40
        cv2.imwrite(str(args.out_dir / f"{idx:04d}_sem.png"), rgb[:, :, ::-1])
        np.savez_compressed(args.out_dir / f"{idx:04d}_scores.npz", scores=maps, argmax=arg,
                            top_score=top_score, margin=margin, valid=valid)
        ref = cv2.imread(str(args.rgb_dir / f"{idx}.jpg"))
        panel = np.concatenate([ref, rgb[:, :, ::-1]], 1)
        cv2.imwrite(str(args.out_dir / f"{idx:04d}_montage.jpg"), panel)
        bgfrac = 100.0 * bgmask.mean()
        report[idx] = bgfrac
        print(f"frame {idx}: background {bgfrac:.1f}%   top-score median {np.median(top_score[valid]):.3f} "
              f"margin median {np.median(margin[valid]):.3f}")

    (args.out_dir / "prompts.json").write_text(json.dumps({"prompts": prompts, "colors": cols.tolist(),
                                                           "bg_prompt": args.bg_prompt}, indent=2))


if __name__ == "__main__":
    main()
