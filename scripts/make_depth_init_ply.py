#!/usr/bin/env python
"""Build a PLY pointcloud init for splatfacto by unprojecting Replica depth.

Matches the ordering used by replica_to_transforms.py with --subset N --offset 0
(ns image index i == original frame id i). Add `"ply_file_path"` to the existing
transforms.json and nerfstudio will seed the Gaussians from these points.

Usage:
    python scripts/make_depth_init_ply.py \
        --scene data/replica_demo --ns data/replica_demo_ns --n-frames 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

MM = 0.001  # uint16 depth is millimeters in this NICE-SLAM Demo layout


def _read_matrix16(path: Path) -> np.ndarray:
    vals = np.fromstring(path.read_text(), sep=" ")
    return vals.reshape(4, 4)


def _matrix_c2w_gl(p: np.ndarray) -> np.ndarray:
    """Replica poses are camera-to-world in the OpenGL/COLMAP convention used by
    nerfstudio (camera looks down -Z, y up). Returned unchanged."""
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True, help="raw Replica scene dir (frames/...)")
    ap.add_argument("--ns", type=Path, required=True, help="nerfstudio dataset dir to patch")
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--frame-stride", type=int, default=6)
    ap.add_argument("--pixel-stride", type=int, default=3)
    ap.add_argument("--voxel", type=float, default=0.02, help="downsample grid cell (meters)")
    ap.add_argument("--max-depth-m", type=float, default=8.0)
    ap.add_argument("--min-depth-m", type=float, default=0.05)
    ap.add_argument("--max-points", type=int, default=500000)
    args = ap.parse_args()

    raw = args.scene
    K = _read_matrix16(raw / "frames/intrinsic/intrinsic_depth.txt")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pose_dir = raw / "frames/pose"
    depth_dir = raw / "frames/depth"

    pts_all = []
    for i in range(0, args.n_frames, args.frame_stride):
        d = Image.open(depth_dir / f"{i}.png")
        depth = np.asarray(d, dtype=np.float32) * MM
        H, W = depth.shape
        pose = _matrix_c2w_gl(_read_matrix16(pose_dir / f"{i}.txt"))
        R = pose[:3, :3]
        t = pose[:3, 3]

        yy, xx = np.mgrid[0:H:args.pixel_stride, 0:W:args.pixel_stride]
        depth_s = depth[:: args.pixel_stride, :: args.pixel_stride]
        valid = (depth_s > args.min_depth_m) & (depth_s < args.max_depth_m)
        xx, yy, zz = xx[valid], yy[valid], depth_s[valid]

        # OpenGL camera frame: x right, y up, forward -Z.
        # Treat depth as distance along the ray.
        ray = np.stack(
            [(xx - cx) / fx, -(yy - cy) / fy, -np.ones_like(xx)], axis=-1
        )
        ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
        cam = ray * zz[..., None]  # [M,3]
        world = (R @ cam.T).T + t  # [M,3]
        pts_all.append(world)

    if not pts_all:
        raise SystemExit("no points generated")
    P = np.concatenate(pts_all, axis=0)
    # coarse voxel downsample
    keys = np.round(P / args.voxel)
    _, uniq = np.unique(keys, axis=0, return_index=True)
    P = P[uniq]
    if len(P) > args.max_points:
        idx = np.random.default_rng(0).choice(len(P), args.max_points, replace=False)
        P = P[idx]
    print("init points:", len(P))

    ply_path = args.ns / "points3d.ply"
    verts = np.zeros(P.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    verts["x"], verts["y"], verts["z"] = P[:, 0], P[:, 1], P[:, 2]
    verts["red"] = 128
    verts["green"] = 128
    verts["blue"] = 128

    from plyfile import PlyData, PlyElement

    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(ply_path))
    tf = json.loads((args.ns / "transforms.json").read_text())
    tf["ply_file_path"] = "points3d.ply"
    (args.ns / "transforms.json").write_text(json.dumps(tf, indent=2))
    print("wrote", ply_path, "and added ply_file_path to transforms.json")


if __name__ == "__main__":
    main()
