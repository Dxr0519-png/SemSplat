#!/usr/bin/env python
"""Render side-by-side reconstruction views from two trained semsplat runs.

For a set of *dataset image indices* (which map 1:1 onto raw Replica frames
0..N-1 here), each model is rendered with ITS OWN dataset pose for that index
(each model lives in its own world frame), and the result is compared with the
original ground-truth RGB frame.

Outputs (per view):
    results/orb_ate/vis/<frame>_gt.png          ground-truth RGB
    results/orb_ate/vis/<frame>_recon_gt.png    render from GT-pose model
    results/orb_ate/vis/<frame>_recon_est.png   render from ORB-est model
    results/orb_ate/vis/<frame>_montage.png     [ref | GT | ORB] with labels

Usage:
    compare_views.py --config-gt <gt.yml> --config-est <est.yml> \
        [--frames 15 60 120 200 280] [--ns-gt data/<gt_ns> --ns-est data/<est_ns>] [--rgb data/replica_demo/frames/color]
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

# torch>=2.6 weights_only patch (same as query_cli / ns_eval_wrapper)
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup


def _make_camera(transform_matrix, fl_x, fl_y, cx, cy, w, h, device):
    c2w = np.asarray(transform_matrix, float)[:3, :4]
    return Cameras(
        camera_to_worlds=torch.tensor(c2w, dtype=torch.float32)[None].to(device),
        fx=fl_x, fy=fl_y, cx=cx, cy=cy,
        width=w, height=h,
        distortion_params=None,
        camera_type=CameraType.PERSPECTIVE,
    ).to(device)


def _render(model, cam):
    out = model.get_outputs_for_camera(cam)
    rgb = out["rgb"].clamp(0.0, 1.0).detach().cpu().numpy()
    return (rgb * 255.0).round().astype(np.uint8)


def _load_meta(ns_dir: Path):
    tf = json.loads((ns_dir / "transforms.json").read_text())
    return tf, {f["file_path"]: f for f in tf["frames"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-gt", type=Path, required=True)
    ap.add_argument("--config-est", type=Path, required=True)
    ap.add_argument("--ns-gt", type=Path, required=True, help="GT-pose ns dataset dir (for poses)")
    ap.add_argument("--ns-est", type=Path, required=True, help="EST-pose ns dataset dir (for poses)")
    ap.add_argument("--rgb-dir", type=Path, required=True, help="raw Replica color dir for reference")
    ap.add_argument("--out-dir", type=Path, default=Path("results/orb_ate/vis"))
    ap.add_argument("--frames", type=int, nargs="+", default=[15, 60, 120, 200, 280])
    args = ap.parse_args()

    meta_gt, frames_gt = _load_meta(args.ns_gt)
    meta_est, frames_est = _load_meta(args.ns_est)
    assert len(frames_gt) == len(frames_est) == 300

    def _get_meta(meta, frames, idx):
        f = frames[f"images/{idx:06d}.jpg"]
        return f["transform_matrix"], meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"], meta["w"], meta["h"]

    device = "cuda"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recons = {"gt": {}, "est": {}}
    for tag, cfg, ns in (("gt", args.config_gt, args.ns_gt), ("est", args.config_est, args.ns_est)):
        print(f"loading {tag} pipeline ...")
        _, pipeline, _, _ = eval_setup(cfg)
        model = pipeline.model
        meta = meta_gt if tag == "gt" else meta_est
        frames = frames_gt if tag == "gt" else frames_est
        # rebuild per-frame meta by index; keys are images/000000.jpg
        for idx in args.frames:
            key = f"images/{idx:06d}.jpg"
            f = frames[key]
            cam = _make_camera(f["transform_matrix"], meta["fl_x"], meta["fl_y"],
                               meta["cx"], meta["cy"], meta["w"], meta["h"], device)
            recons[tag][idx] = _render(model, cam)
        del pipeline, model
        torch.cuda.empty_cache()

    for idx in args.frames:
        # reference GT RGB frame (raw color is the same image both arms trained on)
        raw = cv2.imread(str(args.rgb_dir / f"{idx}.jpg"))  # BGR
        gt_rgb = recons["gt"][idx]      # RGB
        est_rgb = recons["est"][idx]    # RGB
        gt_bgr = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
        est_bgr = cv2.cvtColor(est_rgb, cv2.COLOR_RGB2BGR)
        base = args.out_dir / f"{idx:04d}"
        cv2.imwrite(str(base) + "_ref.jpg", raw)
        cv2.imwrite(str(base) + "_gt.jpg", gt_bgr)
        cv2.imwrite(str(base) + "_est.jpg", est_bgr)

        h = raw.shape[0]
        panel = np.concatenate([raw, gt_bgr, est_bgr], axis=1)
        labels = ["GT reference", "GT-pose reconstruction", "ORB-pose reconstruction"]
        for k, lab in enumerate(labels):
            cv2.putText(panel, lab, (k * raw.shape[1] + 12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(base) + "_montage.jpg", panel)
        print("wrote", base, "(*_ref/_gt/_est/_montage.jpg)")

    print("open results/orb_ate/vis/*_montage.jpg to compare")


if __name__ == "__main__":
    main()
