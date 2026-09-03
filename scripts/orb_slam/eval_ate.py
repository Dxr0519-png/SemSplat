#!/usr/bin/env python
"""Quantify ORB-SLAM3 trajectory quality against Replica ground-truth poses.

Pure-numpy Umeyama-style SE(3) alignment (scale fixed to 1, metric RGB-D),
then report absolute trajectory error (ATE) on the camera centres of the
frames ORB-SLAM3 actually tracked.

Outputs
-------
results/orb_ate/ate_summary.txt   human-readable numbers
results/orb_ate/ate.npz           indices / est / gt / aligned residual
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from scripts.orb_slam.common import load_matrix16, read_tum, rigid_align


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, required=True, help="ORB CameraTrajectory.txt (TUM)")
    ap.add_argument("--gt-scene", type=Path, required=True, help="raw Replica scene with GT poses")
    ap.add_argument("--gt-pose-dir", default="frames/pose")
    ap.add_argument("--ts-step", type=float, default=0.02, help="match replica_prep_orb")
    ap.add_argument("--out-dir", type=Path, default=Path("results/orb_ate"))
    ap.add_argument("--title", default="orb_slam3_replica")
    args = ap.parse_args()

    traj = read_tum(args.traj)
    gt_dir = args.gt_scene / args.gt_pose_dir
    gt_ids = sorted(int(p.stem) for p in gt_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    print(f"traj entries: {len(traj)}   gt poses available: {len(gt_ids)}")

    est_by_idx, ts_sorted = {}, sorted(e[0] for e in traj)
    for t, T in traj:
        est_by_idx[int(round(t / args.ts_step))] = T

    idx = sorted(est_by_idx)
    total = len(gt_ids)
    n_tracked = sum(1 for i in idx if i in set(gt_ids))
    frac = 100.0 * n_tracked / max(total, 1)
    print(f"tracked frames: {n_tracked}/{total} ({frac:.1f}%)")

    E, G = [], []
    used = []
    for i in idx:
        gp = gt_dir / f"{i}.txt"
        if not gp.exists():
            continue
        E.append(est_by_idx[i][:3, 3])
        G.append(load_matrix16(gp)[:3, 3])
        used.append(i)
    E, G = np.asarray(E), np.asarray(G)

    R, t = rigid_align(E, G)
    aligned = E @ R.T + t
    resid = np.linalg.norm(aligned - G, axis=1)

    rmse = float(np.sqrt((resid ** 2).mean()))
    mean = float(resid.mean())
    med = float(np.median(resid))
    mx = float(resid.max())

    # trajectory length in ORB's own (metric) world from the tracked order
    ordered = np.asarray([est_by_idx[i][:3, 3] for i in sorted(est_by_idx)])
    length = float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())

    print(f"ATE RMSE : {rmse*100:.1f} cm   ({rmse/length*100:.2f}% of traj len {length:.2f} m)")
    print(f"ATE mean / median / max : {mean*100:.1f} / {med*100:.1f} / {mx*100:.1f} cm")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summ = args.out_dir / "ate_summary.txt"
    summ.write_text(
        "\n".join(
            [
                f"title      : {args.title}",
                f"traj file  : {args.traj}",
                f"tracked    : {n_tracked}/{total} ({frac:.1f}%)",
                f"traj len(m): {length:.3f}",
                f"ate_rmse_m : {rmse:.5f}",
                f"ate_mean_m : {mean:.5f}",
                f"ate_median_m: {med:.5f}",
                f"ate_max_m  : {mx:.5f}",
                f"ate_pct_len: {100.0*rmse/length:.3f}",
            ]
        )
        + "\n"
    )
    np.savez(
        args.out_dir / "ate.npz",
        idx=np.asarray(used), est=E, gt=G, residual=resid,
        R_est_to_gt=R, t_est_to_gt=t, length=length, rmse=rmse,
    )
    print(f"wrote {summ}")
    print(f"wrote {args.out_dir / 'ate.npz'}")


if __name__ == "__main__":
    main()
