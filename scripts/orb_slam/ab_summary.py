#!/usr/bin/env python
"""Merge two ns-eval metrics.json (GT-pose arm vs ORB-estimated-pose arm) into a
small A/B summary table at results/orb_ate/ab_summary.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root


def _load(p: Path):
    j = json.loads(Path(p).read_text())
    res = j.get("results", j)
    return {k: float(np.mean(v)) if isinstance(v, list) else float(v) for k, v in res.items() if k in ("psnr", "ssim", "lpips")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True, help="ns-eval metrics.json for GT-pose arm")
    ap.add_argument("--est", type=Path, required=True, help="ns-eval metrics.json for est-pose arm")
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/ab_summary.txt"))
    args = ap.parse_args()

    gt = _load(args.gt)
    est = _load(args.est)
    keys = sorted(set(gt) | set(est))
    lines = [f"{'metric':8s} {'GT-pose':>10s} {'EST-pose':>10s} {'delta(EST-GT)':>14s}"]
    for k in keys:
        g, e = gt.get(k, float("nan")), est.get(k, float("nan"))
        d = e - g
        lines.append(f"{k:8s} {g:10.4f} {e:10.4f} {d:+14.4f}")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
