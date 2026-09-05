#!/usr/bin/env python
"""Closed-set mIoU + optional query-time geometric prior over stuff planes.

Same closed-set benchmark as eval_miou_room.py, plus ``--prior norm``: at query
time we rasterize the gaussian *world positions* to a position map, recover the
per-pixel surface normal by finite differences, and orient it toward the camera.
Then for pixels whose text-argmax is already one of the stuff planes
(wall/floor/ceiling) we arbitrate with the surface orientation in the scene's
world up:
    normal . up  <  -cos_thr   -> boost ceiling   (surface faces down = it's above you)
    normal . up  >  +cos_thr   -> boost floor     (surface faces up)
    |normal . up| <  horiz_thr -> boost wall      (off by default: never demote a stuff label)
Objects (top-1 not a stuff plane) are left untouched, so tabletops/rugs/sofas
whose own class already won are never re-bucketed.

This is pure query-time geometry (no retraining): a deployment that has its own
pose + reconstruction (ORB-SLAM + the same 3DGS) can apply it identically.

Usage mirrors eval_miou_room.py:
    eval_geo_prior.py --config <cfg.yml> --ns <ns_dir> --gt-dir <semantic dir> \
        --prompts "..." --gt-ids ... --frames ... [--prior norm] \
        [--geo-up 0 1 0] [--geo-cos-thr 0.85] [--geo-horiz-thr 0.3] [--geo-delta 2.0] \
        [--geo-iwall 4 --geo-ifloor 5 --geo-iceil 6]
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

EPS = 1e-6


@torch.no_grad()
def world_position_map(model, camera) -> torch.Tensor:
    """Rasterize gaussian world positions -> expected [H,W,3] position map."""
    pos, alpha = scoring.rasterize_class_scores(model, camera, model.means.detach())
    return pos, alpha


@torch.no_grad()
def surface_upness(model, camera, up: torch.Tensor, cos_thr: float):
    """Per-pixel signed camera-facing normal component along ``up`` -> [H,W].

    Positive = surface faces up (floor), negative = faces down (ceiling),
    ~0 = vertical (wall). Derived from the rasterized world-position map.
    """
    pos, _ = world_position_map(model, camera)          # [H,W,3]
    pad = torch.nn.functional.pad(pos.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")[0]  # [3,H+2,W+2]
    dud = (pad[:, 1:-1, 2:] - pad[:, 1:-1, :-2])         # central diff over u (x)
    dvd = (pad[:, 2:, 1:-1] - pad[:, :-2, 1:-1])         # central diff over v (y)
    n = torch.cross(dud.permute(1, 2, 0), dvd.permute(1, 2, 0), dim=-1)  # [H,W,3] ~world normal
    cam_pos = camera.camera_to_worlds[0, :3, 3]
    to_cam = cam_pos[None, None] - pos
    flip = (n * to_cam).sum(-1) > 0  # orient toward the camera (verified: ceiling<0, floor>0)
    n = torch.where(flip[..., None], -n, n)
    nlen = n.norm(dim=-1).clamp_min(EPS)
    n = n / nlen[..., None]
    return (n * up[None, None]).sum(-1)  # signed world-upness, [-1,1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ns", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--gt-ids", type=int, nargs="+", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/geo_miou.txt"))
    ap.add_argument("--dump-scores", type=Path, default=None)
    ap.add_argument("--prior", choices=["none", "norm"], default="norm")
    ap.add_argument("--geo-up", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    ap.add_argument("--geo-cos-thr", type=float, default=0.85)
    ap.add_argument("--geo-horiz-thr", type=float, default=0.0,
                    help="if >0, pixels whose |normal.up| is below this are demoted to wall "
                         "(default 0.0 = never demote; only arbitrate floor/ceiling + wall->plane)")
    ap.add_argument("--geo-delta", type=float, default=2.0)
    ap.add_argument("--geo-iwall", type=int, default=4)
    ap.add_argument("--geo-ifloor", type=int, default=5)
    ap.add_argument("--geo-iceil", type=int, default=6)
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
    up = torch.tensor(args.geo_up, dtype=torch.float32, device=model.device)
    up = up / up.norm()

    tp = np.zeros(C); fp = np.zeros(C); fn = np.zeros(C)
    if args.dump_scores is not None:
        args.dump_scores.mkdir(parents=True, exist_ok=True)
    for k in args.frames:
        f = byf[k]
        M = np.asarray(f["transform_matrix"], float)[:3, :4]
        cam = Cameras(camera_to_worlds=torch.tensor(M)[None].float().to(model.device),
                      fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                      width=meta["w"], height=meta["h"]).to(model.device)
        maps, alpha = scoring.semantic_maps_from_prompts(model, cam, emb)
        maps = maps.detach(); alpha = alpha.detach()
        if args.prior == "norm":
            nx = surface_upness(model, cam, up, args.geo_cos_thr)   # [H,W]
            pred = maps.argmax(-1)
            stuff = (pred == args.geo_iwall) | (pred == args.geo_ifloor) | (pred == args.geo_iceil)
            d = torch.zeros_like(maps)
            c_ceil = stuff & (nx < -args.geo_cos_thr)
            c_floor = stuff & (nx > args.geo_cos_thr)
            c_wall = stuff & (nx.abs() < args.geo_horiz_thr)
            d[..., args.geo_iceil][c_ceil] = args.geo_delta
            d[..., args.geo_ifloor][c_floor] = args.geo_delta
            d[..., args.geo_iwall][c_wall] = args.geo_delta
            maps = maps + d
        maps = maps.cpu().numpy(); alpha = alpha.cpu().numpy()
        if maps.ndim == 4:
            maps, alpha = maps[0], alpha[0]
        valid = alpha[..., 0] > 0.5
        if args.dump_scores is not None:
            full_arg = maps.argmax(-1).astype(np.uint8)
            full_arg[~valid] = 255
            top2 = np.partition(maps, -2, axis=-1)
            np.savez_compressed(args.dump_scores / f"{k:04d}_scores.npz",
                                scores=maps.astype(np.float16), argmax=full_arg,
                                top_score=top2[..., -1].astype(np.float16),
                                margin=(top2[..., -1] - top2[..., -2]).astype(np.float16),
                                valid=valid)
        pred = maps.argmax(-1)
        pred = pred[valid]
        gtf = args.gt_dir / f"{k}.png"
        if not gtf.exists():
            gtf = args.gt_dir / f"{k:06d}.png"
        gt = cv2.imread(str(gtf), -1)
        gt = gt[valid].ravel()
        gt[gt == 0] = -1
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
    if args.dump_scores is not None:
        (args.dump_scores / "prompts.json").write_text(
            json.dumps({"prompts": prompts, "gt_ids": args.gt_ids, "frames": args.frames,
                        "prior": args.prior, "geo_up": args.geo_up,
                        "geo_cos_thr": args.geo_cos_thr, "geo_delta": args.geo_delta}, indent=2))
        print("wrote per-frame npz + prompts.json to", args.dump_scores)


if __name__ == "__main__":
    main()
