#!/usr/bin/env python
"""Side-by-side open-vocabulary semantic argmax maps: GT-pose model vs ORB-est.

Reuses query_cli's text encoding + score rasterization, but builds cameras from
each model's own transforms.json so we can compare the SAME raw frames.

Outputs (per view, in --out-dir):
    <frame>_sem_gt.png / <frame>_sem_est.png      argmax class colour maps
    <frame>_montage.jpg                           [GT ref | GT sem | ORB sem]
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

_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from semsplat.inference import scoring
from semsplat.teachers.clip_local_teacher import encode_text_prompts


def _palette(n: int) -> np.ndarray:
    import colorsys

    cols = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        cols.append([int(255 * r), int(255 * g), int(255 * b)])
    return np.asarray(cols, dtype=np.uint8)


def _camera(transform_matrix, meta, device):
    c2w = np.asarray(transform_matrix, float)[:3, :4]
    return Cameras(
        camera_to_worlds=torch.tensor(c2w, dtype=torch.float32)[None].to(device),
        fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
        width=meta["w"], height=meta["h"], distortion_params=None,
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)


def _sem_map(model, camera, emb, palette, nclasses):
    device = model.device
    maps, alpha = scoring.semantic_maps_from_prompts(model, camera.to(device), emb)
    maps = maps.detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy()
    valid = alpha[..., 0] > 0.5
    H, W = maps.shape[:2]
    argmax = np.full((H, W), nclasses, dtype=np.uint16)
    argmax[valid] = maps.argmax(axis=-1)[valid]
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(nclasses):
        rgb[argmax == c] = palette[c]
    rgb[~valid] = 40
    return rgb


def _load_meta(ns_dir: Path):
    tf = json.loads((ns_dir / "transforms.json").read_text())
    return tf, {f["file_path"]: f for f in tf["frames"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-gt", type=Path, required=True)
    ap.add_argument("--config-est", type=Path, required=True)
    ap.add_argument("--ns-gt", type=Path, required=True)
    ap.add_argument("--ns-est", type=Path, required=True)
    ap.add_argument("--rgb-dir", type=Path, required=True, help="raw Replica color dir")
    ap.add_argument("--out-dir", type=Path, default=Path("results/orb_ate/semantic"))
    ap.add_argument("--frames", type=int, nargs="+", default=[15, 60, 120, 200, 280])
    ap.add_argument("--prompts", default="sofa,chair,table,desk,monitor,plant,wall,floor,ceiling")
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    palette = _palette(len(prompts))
    device = "cuda"

    meta_gt, f_gt = _load_meta(args.ns_gt)
    meta_est, f_est = _load_meta(args.ns_est)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {"gt": {}, "est": {}}
    for tag, cfg, ns, meta, frames in (("gt", args.config_gt, args.ns_gt, meta_gt, f_gt),
                                       ("est", args.config_est, args.ns_est, meta_est, f_est)):
        print(f">> load {tag} pipeline")
        _, pipeline, _, _ = eval_setup(cfg)
        model = pipeline.model
        model.eval()
        teacher = model.config.teacher_model_name
        print("   encode prompts via", teacher)
        emb = torch.stack([encode_text_prompts(prompts, teacher)[p] for p in prompts], dim=0).to(device)
        for idx in args.frames:
            f = frames[f"images/{idx:06d}.jpg"]
            cam = _camera(f["transform_matrix"], meta, device)
            results[tag][idx] = _sem_map(model, cam, emb, palette, len(prompts))
        del pipeline, model
        torch.cuda.empty_cache()

    for idx in args.frames:
        raw = cv2.imread(str(args.rgb_dir / f"{idx}.jpg"))
        gt_bgr = cv2.cvtColor(results["gt"][idx], cv2.COLOR_RGB2BGR)
        est_bgr = cv2.cvtColor(results["est"][idx], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(args.out_dir / f"{idx:04d}_sem_gt.png"), gt_bgr)
        cv2.imwrite(str(args.out_dir / f"{idx:04d}_sem_est.png"), est_bgr)
        h = raw.shape[0]
        panel = np.concatenate([raw, gt_bgr, est_bgr], axis=1)
        for k, lab in enumerate(["GT reference", "GT-pose semantics", "ORB-pose semantics"]):
            cv2.putText(panel, lab, (k * raw.shape[1] + 12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.out_dir / f"{idx:04d}_montage.jpg"), panel)
        print("wrote", args.out_dir / f"{idx:04d}_montage.jpg")

    with open(args.out_dir / "prompts.json", "w") as fh:
        json.dump({"prompts": prompts, "colors": palette.tolist()}, fh, indent=2)
    print("class colours saved in prompts.json")


if __name__ == "__main__":
    main()
