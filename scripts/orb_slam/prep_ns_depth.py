#!/usr/bin/env python
"""Prepare per-cam-idx depth files next to an ns RGB dataset (global-first folding).

Replica-derived est frame dirs (e.g. ``office0_est320/frames/depth``) are keyed by
the *local* 300-frame id (depth 0..299 == GT depth 0,3,..,897), which equals the ns
dataset camera index -- so by default (``--id-mode index``) cam k is matched to the
depth file numbered k. If instead your --depth-root is the *full source* dir keyed
by source frame id (e.g. ``Downloads/.../Sequence_1/depth`` with ``depth_0..897``),
use ``--id-mode source``, which resolves each ns image's real path and matches the
trailing source integer (ns cam k -> rgb_3k -> depth_3k).

The teacher looks depth up straight by cam idx, so this writes
``<ns>/depth/{cam:06d}.png`` (copy, or --symlink) after validating uint16-mm depth
of the same size as the ns frame.

Usage::
    prep_ns_depth.py --ns data/office0_est320_ns \\
        --depth-root data/office0_est320/frames/depth          # index mode
    prep_ns_depth.py --ns data/office0_est320_ns --id-mode source \\
        --depth-root /home/dxr/Downloads/office0_gt/office_0/Sequence_1/depth
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def _trailing_int(p: Path) -> int | None:
    """Last digit-run in the file *stem* (e.g. rgb_87 -> 87, 000100 -> 100)."""
    digs = re.findall(r"\d+", p.stem)
    return int(digs[-1]) if digs else None


def _find_depth(root: Path, src: int) -> Path | None:
    """Depth file matching source id: exact names first, else unique trailing-int scan."""
    for name in (f"{src}.png", f"{src:06d}.png", f"{src:04d}.png"):
        p = root / name
        if p.exists():
            return p
    hits = [p for p in root.glob("*.png") if _trailing_int(p) == src]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"  !! {src}: {len(hits)} depth files match ({[h.name for h in hits]}); taking {hits[0].name}")
        return hits[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=Path, required=True, help="ns dataset dir (contains transforms.json + images/)")
    ap.add_argument("--depth-root", type=Path, required=True, help="depth dir to draw from (uint16-mm *.png)")
    ap.add_argument("--id-mode", choices=["index", "source"], default="index",
                    help="index: depth numbered by ns cam idx (est-style, default); "
                         "source: depth numbered by resolved source frame id")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <ns>/depth")
    ap.add_argument("--symlink", action="store_true", help="symlink instead of copying (saves disk)")
    args = ap.parse_args()

    ns = args.ns.resolve()
    tpath = ns / "transforms.json"
    if not tpath.exists():
        raise SystemExit(f"no {tpath}")
    import json

    meta = json.loads(tpath.read_text())
    frames = meta["frames"]
    if not frames:
        raise SystemExit("no frames in transforms.json")
    H, W = int(meta.get("h", 0)), int(meta.get("w", 0))
    if not H or not W:
        raise SystemExit("transforms.json missing h/w")

    out = (args.out_dir or ns / "depth").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (ns / "images").is_dir():
        raise SystemExit(f"expected {ns}/images/")

    root = args.depth_root.resolve()
    n_ok = n_skip = 0
    for k, f in enumerate(frames):
        rel = f["file_path"]
        full = (ns / rel).resolve()
        if not full.exists():
            print(f"cam {k}: image {full} missing -- skipping")
            n_skip += 1
            continue
        src = _trailing_int(full)
        if args.id_mode == "source":
            if src is None:
                print(f"cam {k}: no source id in {full.name} -- skipping")
                n_skip += 1
                continue
            target = src
            desc = f"source {src}"
        else:
            target = k  # est-style depth dirs are keyed by the local ns cam id
            desc = f"cam {k}"
        df = _find_depth(root, target)
        if df is None:
            print(f"cam {k}: no depth for {desc} -- skipping")
            n_skip += 1
            continue
        d = cv2.imread(str(df), cv2.IMREAD_UNCHANGED)
        if d is None or d.dtype != np.uint16:
            print(f"cam {k}: depth {df} not uint16 -- skipping")
            n_skip += 1
            continue
        if d.shape[:2] != (H, W):
            print(f"cam {k}: depth {d.shape[:2]} != frame {(H, W)} -- skipping")
            n_skip += 1
            continue
        dst = out / f"{k:06d}.png"
        if args.symlink:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            try:
                os.symlink(df, dst)
            except OSError:
                dst.symlink_to(df.resolve())  # absolute symlink fallback
        else:
            if not cv2.imwrite(str(dst), d):
                print(f"cam {k}: failed to write {dst}")
                n_skip += 1
                continue
        n_ok += 1

    print(f"wrote {n_ok}/{len(frames)} depth frames -> {out}  (skipped {n_skip})")


if __name__ == "__main__":
    main()
