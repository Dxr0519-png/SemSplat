"""Convert a processed Replica RGB-D sequence (GT poses, no COLMAP) into a
nerfstudio ``transforms.json`` dataset.

Targets the NICE-SLAM Demo / scene layout (as served from
https://cvg-data.inf.ethz.ch/nice-slam/), which looks like::

    <scene>/
      frames/
        color/<id>.jpg          (images; could also be png)
        pose/<id>               (16 floats, camera-to-world, no extension)
        intrinsic/intrinsic_color.txt   (3x4 camera matrix K)
        depth/...

Explicit flags are provided for the common alternatives; ``--probe`` prints what
was found without writing anything. Poses are written to ``transforms.json``
as-is (Replica GT poses are camera-to-world in nerfstudio's OpenGL convention;
if a quick train looks mirrored, re-run with ``--invert-pose``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg"}
_SUB_OK = ("color", "rgb", "images", "frames")


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def _parse_intrinsics(txt: str):
    rows = [[float(x) for x in ln.split()] for ln in txt.strip().splitlines() if ln.strip()]
    K = np.asarray(rows)
    if K.shape == (4, 4):  # 3x4 K padded with an identity bottom row
        K = K[:3, :4]
    if K.shape in ((3, 3), (3, 4)):
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    raise ValueError(f"unexpected intrinsic shape {rows and np.asarray(rows).shape}")


def discover_color(scene: Path, color_dir: str | None = None) -> list:
    def _files(d: Path):
        return [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]

    if color_dir:
        return sorted((f for f in (scene / color_dir).rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXT), key=_natural_key)

    best_dir, best_score = None, -1
    for d in [p for p in scene.rglob("*") if p.is_dir()]:
        low = str(d).lower()
        if "depth" in low:
            continue
        name = d.name.lower()
        score = 0
        if name in ("color", "rgb", "images", "image", "frame"):
            score += 3
        if not _files(d):
            continue
        score += 1  # has images directly
        if score > best_score:
            best_score, best_dir = score, d
    if best_dir is not None:
        return sorted(_files(best_dir), key=_natural_key)
    return []


def discover_poses(scene: Path, pose_dir: str | None = None) -> list:
    def _one_file(f: Path):
        try:
            return _matrix16(f.read_text())
        except ValueError:
            return None

    if pose_dir:
        base = scene / pose_dir
        files = sorted([f for f in base.iterdir() if f.is_file()], key=_natural_key)
        poses = []
        for f in files:
            m = _one_file(f)
            if m is None:
                return []
            poses.append(m)
        if poses:
            return poses
    for cand in (scene, scene / "frames"):
        if not cand.exists():
            continue
        for d in sorted(cand.iterdir()):
            if d.is_dir() and d.name.lower() in ("pose", "poses"):
                files = sorted([f for f in d.iterdir() if f.is_file()], key=_natural_key)
                poses = [_one_file(f) for f in files]
                if poses and all(m is not None for m in poses):
                    return poses
    return []


def _matrix16(txt: str) -> np.ndarray:
    vals = np.fromstring(txt, sep=" ")
    if vals.size == 16:
        return vals.reshape(4, 4)
    raise ValueError(f"expected 16 floats, got {vals.size}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replica RGB-D -> nerfstudio transforms.json")
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--color-dir", default=None, help="e.g. frames/color")
    ap.add_argument("--pose-dir", default=None, help="e.g. frames/pose")
    ap.add_argument("--intrinsics-file", default=None, help="path (relative to scene) to a 3x3/3x4 K file")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--invert-pose", action="store_true")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    imgs = discover_color(args.scene, args.color_dir)
    poses = discover_poses(args.scene, args.pose_dir)

    if args.probe:
        print(f"color images : {len(imgs)}  first: {imgs[0].relative_to(args.scene) if imgs else '-'}")
        print(f"poses        : {len(poses)}")
        if imgs:
            from PIL import Image

            with Image.open(imgs[0]) as im:
                print("first image size:", im.size)
        return

    n = min(len(imgs), len(poses)) if imgs and poses else 0
    if n == 0:
        raise SystemExit(
            f"no matched frames (imgs={len(imgs)}, poses={len(poses)}); run --probe, "
            "and pass --color-dir/--pose-dir if not the default Demo layout"
        )

    fx = fy = cx = cy = None
    if args.intrinsics_file and not (args.fx or args.fy or args.cx is not None or args.cy is not None):
        fx, fy, cx, cy = _parse_intrinsics((args.scene / args.intrinsics_file).read_text())
    fx = args.fx or fx
    fy = args.fy or fy or fx
    cx = args.cx if args.cx is not None else cx
    cy = args.cy if args.cy is not None else cy
    if not fx or not fy or cx is None or cy is None:
        raise SystemExit("intrinsics missing: pass --intrinsics-file or --fx --fy --cx --cy")

    from PIL import Image

    with Image.open(imgs[0]) as im:
        w, h = im.size

    if args.subset:
        imgs = imgs[args.offset : args.offset + args.subset]
        poses = poses[args.offset : args.offset + args.subset]

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (img, M) in enumerate(zip(imgs, poses)):
        dst = img_dir / f"{i:06d}{img.suffix.lower()}"
        if not dst.exists():
            if args.copy:
                shutil.copyfile(img, dst)
            else:
                try:
                    os.symlink(img.resolve(), dst)
                except OSError:
                    shutil.copyfile(img, dst)
        c2w = np.linalg.inv(M) if args.invert_pose else M
        frames.append({"file_path": f"images/{dst.name}", "transform_matrix": c2w.tolist()})

    (args.out / "transforms.json").write_text(
        json.dumps(
            {"camera_model": "OPENCV", "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": w, "h": h, "frames": frames},
            indent=2,
        )
    )
    print(f"wrote {len(frames)} frames -> {args.out / 'transforms.json'}")


if __name__ == "__main__":
    main()
