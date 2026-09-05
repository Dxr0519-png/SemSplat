#!/usr/bin/env python
"""Aggregate recall / IoU / false-positive-on-stuff for monitored classes from an
eval npz dump (``--dump-scores`` output of eval_miou_room.py / eval_geo_prior.py).

Motivation: the negative-contrastive "flat-wall veto" locks the same geometrically-
flat, coplanar cells that REAL flush objects (television screen / door / table
lamp / wall clock on a wall) occupy -- so unlike the earlier ×0 experiment, the
active hinge can erode real-object recall. This script is the tripwire: it reports,
over the same 50-frame aggregation the mIoU benchmarks use, per monitored class

    recall          = TP / (TP + FN)
    IoU
    FP-on-stuff     = predicted-class pixels whose GT is a stuff plane
                      (wall/floor/ceiling) -- the "wall speckle" phantom pixels

so a regression arm (reg vs +contrast) can be compared pixel-for-pixel.

Usage mirrors eval_miou_room.py:
    measure_thing_recall.py --scores-dir results/orb_ate/negcon/<arm>_npz \
        --gt-dir data/office0_est320/frames/semantic \
        [--monitor "television screen,door,table lamp,wall clock"] \
        [--stuff-gt-ids 93 40 31] --out results/orb_ate/negcon/thing_recall.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load_frame_npz(scores_dir: Path, k: int):
    p = scores_dir / f"{k:04d}_scores.npz"
    if not p.exists():  # some dumps may 6-digit the index
        p = scores_dir / f"{k:06d}_scores.npz"
    z = np.load(p)
    return z["argmax"], z["valid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", type=Path, required=True,
                    help="dir holding per-frame *_scores.npz + prompts.json (eval dump)")
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--monitor", default="television screen,door,table lamp,wall clock",
                    help="comma-separated prompt names to watch (subset of the eval vocab)")
    ap.add_argument("--stuff-gt-ids", type=int, nargs="+", default=[93, 40, 31],
                    help="GT ids counted as stuff planes (office0: wall 93 floor 40 ceiling 31)")
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/negcon/thing_recall.txt"))
    args = ap.parse_args()

    meta = json.loads((args.scores_dir / "prompts.json").read_text())
    prompts = meta["prompts"]
    frames = meta["frames"]
    monitor = [s.strip() for s in args.monitor.split(",") if s.strip()]
    idx = {name: i for i, name in enumerate(prompts)}
    missing = [m for m in monitor if m not in idx]
    if missing:
        raise SystemExit(f"monitored prompts not in eval vocab: {missing}\nvocab={prompts}")
    stuff = np.asarray(args.stuff_gt_ids)

    tp = np.zeros(len(monitor))
    fp = np.zeros(len(monitor))
    fn = np.zeros(len(monitor))
    fponstuff = np.zeros(len(monitor))
    for k in frames:
        argmax, valid = _load_frame_npz(args.scores_dir, int(k))
        gtf = args.gt_dir / f"{k}.png"
        if not gtf.exists():
            gtf = args.gt_dir / f"{int(k):06d}.png"
        gt = cv2.imread(str(gtf), -1)
        gt = gt[valid]
        gt[gt == 0] = -1
        is_stuff = np.isin(gt, stuff)
        for j, name in enumerate(monitor):
            p = argmax[valid] == idx[name]
            g = gt == meta["gt_ids"][idx[name]]
            tp[j] += int((p & g).sum())
            fp[j] += int((p & ~g & (gt != -1)).sum())
            fn[j] += int((g & ~p).sum())
            fponstuff[j] += int((p & is_stuff).sum())

    lines = [f"{'class':24s} {'recall':>8s} {'IoU':>8s} {'FP-on-stuff':>12s}"]
    for j, name in enumerate(monitor):
        rec = tp[j] / max(tp[j] + fn[j], 1)
        iou = tp[j] / max(tp[j] + fp[j] + fn[j], 1)
        lines.append(f"{name:24s} {rec:8.3f} {iou:8.3f} {int(fponstuff[j]):12d}")
    txt = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(txt)
    print(txt)
    print(f"aggregated {len(frames)} frames, {len(monitor)} monitored classes -> {args.out}")


if __name__ == "__main__":
    main()
