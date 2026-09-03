#!/usr/bin/env python
"""Generate a tiny synthetic nerfstudio dataset (transforms.json + images) for
smoke-testing the training loop without COLMAP or real data.

Cameras orbit a small cluster of colored blobs at the origin. The images are
simple gradient patterns; geometry is intentionally trivial. This is only for
verifying that ``ns-train semsplat`` runs forward/backward and writes a ckpt.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from PIL import Image


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)) -> np.ndarray:
    f = np.asarray(target, float) - np.asarray(eye, float)
    f = f / np.linalg.norm(f)
    up = np.asarray(up, float)
    back = -f
    right = np.cross(back, up)
    right = right / np.linalg.norm(right)
    up2 = np.cross(right, back)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up2
    c2w[:3, 2] = back
    c2w[:3, 3] = eye
    return c2w


def synth_image(H: int, W: int, seed: int) -> np.ndarray:
    """Cheap deterministic colorful gradient image in [0, 255]."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), dtype=np.float32)
    base = rng.uniform(0.2, 0.8, 3)
    img[:] = base
    cx, cy = W * rng.uniform(0.3, 0.7), H * rng.uniform(0.3, 0.7)
    col = rng.uniform(0.4, 1.0, 3)
    r2 = ((xx - cx) / (W * 0.18)) ** 2 + ((yy - cy) / (H * 0.18)) ** 2
    blob = np.clip(1 - r2, 0, 1)[..., None]
    img += blob * col
    # subtle gradient so PSNR > 0 is meaningful
    img += 0.15 * (yy / H)[..., None] * np.array([0.2, 0.35, 0.1])
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/mini_scene")
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=160)
    ap.add_argument("--fov-x", type=float, default=50.0)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    fx = (args.W / 2.0) / math.tan(math.radians(args.fov_x) / 2.0)
    fy = fx
    cx, cy = args.W / 2.0, args.H / 2.0

    frames = []
    for i in range(args.n_frames):
        yaw = 2 * math.pi * i / args.n_frames
        eye = [3.0 * math.cos(yaw), 1.2 * math.sin(2 * i), 3.0 * math.sin(yaw)]
        c2w = look_at(eye)
        name = f"frame_{i:04d}.png"
        Image.fromarray(synth_image(args.H, args.W, seed=i)).save(os.path.join(args.out, "images", name))
        frames.append(
            {
                "file_path": f"images/{name}",
                "transform_matrix": c2w.tolist(),
            }
        )

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": args.W,
        "h": args.H,
        "frames": frames,
    }
    with open(os.path.join(args.out, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"wrote {args.n_frames} frames to {args.out}")


if __name__ == "__main__":
    main()
