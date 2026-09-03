#!/usr/bin/env python
"""Bridge an ORB-SLAM3 CameraTrajectory.txt into a NICE-SLAM-layout scene whose
poses are the *estimated* ones, so the existing replica_to_transforms.py and
make_depth_init_ply.py can be reused unchanged.

Pose convention
---------------
ORB-SLAM3 TUM output is world-from-camera in the OpenCV camera convention
(+x right, +y down, +z forward). This repo's depth unprojection uses the
OpenGL/COLMAP convention (+x right, +y up, looking down -z). We convert with

    c2w_gl = [ R_wc @ diag(1,-1,-1) | t_wc ; 0 0 0 1 ]

(diag det = +1 -> pure rotation, world handedness unchanged; ORB's metric
world is kept as-is, no GT needed).

Frames that ORB-SLAM3 failed to track (no trajectory entry for their
timestamp) are filled from the nearest tracked neighbour and reported, so the
output scene still has one pose per raw frame (Replica Demo is a 0..N-1
sequential layout).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from scripts.orb_slam.common import load_matrix16, natural_key, orb_to_gl_c2w, read_tum

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, required=True, help="ORB CameraTrajectory.txt (TUM)")
    ap.add_argument("--scene", type=Path, required=True, help="raw Replica scene (frames/color ...)")
    ap.add_argument("--out", type=Path, required=True, help="estimated-pose scene dir to create")
    ap.add_argument("--color-dir", default="frames/color")
    ap.add_argument("--depth-dir", default="frames/depth")
    ap.add_argument("--pose-dir-gt", default="frames/pose")
    ap.add_argument("--intrinsics-dir", default="frames/intrinsic")
    ap.add_argument("--ts-step", type=float, default=0.02, help="ts per fed frame (match replica_prep_orb)")
    ap.add_argument("--no-fill", action="store_true", help="do not fill untracked frames (drop them)")
    ap.add_argument("--probe", action="store_true", help="report only")
    args = ap.parse_args()

    traj = read_tum(args.traj)
    colors = sorted((f for f in (args.scene / args.color_dir).iterdir() if f.suffix.lower() in IMG_EXT), key=natural_key)
    N = len(colors)
    print(f"traj entries : {len(traj)}   raw frames: {N}")

    # map ORB timestamp -> pose (ORB world, OpenCV camera convention)
    ts_to_T = {t: T for t, T in traj}
    est = {}  # raw index -> (4x4 T_wc ORB)
    for t, T in traj:
        i = int(round(t / args.ts_step))
        est[i] = T

    tracked = sorted(i for i in range(N) if i in est)
    print(f"directly tracked frames: {len(tracked)}/{N} "
          f"({100.0 * len(tracked) / max(N, 1):.1f}%)")
    missing = [i for i in range(N) if i not in est]
    if missing:
        print(f"untracked frame ids: {missing[:20]}{' ...' if len(missing) > 20 else ''}")

    if args.probe:
        return

    # fill untracked from nearest tracked neighbour unless --no-fill
    def _fill(i: int):
        lo, hi = i - 1, i + 1
        while lo >= 0 or hi < N:
            if lo >= 0 and lo in est:
                return est[lo]
            if hi < N and hi in est:
                return est[hi]
            lo, hi = lo - 1, hi + 1
        return None

    if missing and not args.no_fill:
        n_fill = 0
        for i in missing:
            T = _fill(i)
            if T is not None:
                est[i] = T
                n_fill += 1
        print(f"filled {n_fill} untracked frame(s) from nearest tracked pose")
        tracked = sorted(i for i in range(N) if i in est)

    # build the scene dir
    frames_out = args.out / "frames"
    for sub in ("color", "depth", "pose"):
        (frames_out / sub).mkdir(parents=True, exist_ok=True)
    (frames_out / "intrinsic").mkdir(parents=True, exist_ok=True)

    color_dir = args.scene / args.color_dir
    depth_dir = args.scene / args.depth_dir
    for c in colors:
        dst = frames_out / "color" / c.name
        if not dst.exists():
            os.symlink(c.resolve(), dst)
    d_dst = frames_out / "depth"
    # depth files mirror color stems with .png
    for c in colors:
        d_src = depth_dir / (c.stem + ".png")
        if not d_src.exists():
            raise SystemExit(f"missing depth {d_src}")
        dst = d_dst / (c.stem + ".png")
        if not dst.exists():
            os.symlink(d_src.resolve(), dst)

    int_src = args.scene / args.intrinsics_dir
    for f in sorted(int_src.iterdir()):
        if f.is_file():
            (frames_out / "intrinsic" / f.name).write_bytes(f.read_bytes())

    n_written = 0
    for i in range(N):
        T = est.get(i)
        if T is None:
            continue  # only reachable with --no-fill
        c2w = orb_to_gl_c2w(T)
        txt = "\n".join(" ".join(f"{v:.9f}" for v in row) for row in c2w)
        (frames_out / "pose" / f"{i}.txt").write_text(txt + "\n")
        n_written += 1

    # marker of directly-tracked ids for downstream eval / fair A/B subsets
    np.save(args.out / "tracked_ids.npy", np.asarray(sorted(k for k in est.keys()), dtype=np.int64))
    print(f"wrote {n_written} estimated poses -> {frames_out / 'pose'}")
    print(f"  color/depth symlinked, intrinsics copied; tracked_ids.npy saved")


if __name__ == "__main__":
    main()
