#!/usr/bin/env python
"""colmap_to_ns.py — 把 raw COLMAP 场景(LERF 等)转成 semsplat 可吃的 nerfstudio transforms.json。

产出布局（与 data/office0_est320_ns 同构，供 semsplat-replica preset 原样使用）：
  <out>/transforms.json   （含 intrinsics/poses + ply_file_path）
  <out>/images/*.jpg      （symlink 到源 COLMAP images）
  <out>/sparse_pc.ply     （colmap_to_json 生成，默认作种子点云）

用法：
  .venv/bin/python scripts/colmap_to_ns.py --scene data/lerf_ovs/waldo_kitchen \
      --out data/waldo_kitchen_ns
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nerfstudio.process_data.colmap_utils import colmap_to_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="COLMAP 场景根目录（含 images/ + sparse/0）")
    ap.add_argument("--out", required=True, help="输出 nerfstudio 数据目录")
    args = ap.parse_args()

    scene = Path(args.scene)
    out = Path(args.out)
    recon = scene / "sparse" / "0"
    img_src = scene / "images"
    assert recon.exists() and (recon / "cameras.bin").exists(), f"缺 sparse/0: {recon}"
    assert img_src.exists(), f"缺 images: {img_src}"

    out.mkdir(parents=True, exist_ok=True)
    # images/ symlink
    img_dst = out / "images"
    img_dst.mkdir(exist_ok=True)
    nlink = 0
    for f in sorted(img_src.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lnk = img_dst / f.name
        if not lnk.exists():
            lnk.symlink_to(f.resolve())
            nlink += 1
    print(f"[colmap_to_ns] symlinked {nlink} images -> {img_dst}")

    n = colmap_to_json(recon_dir=recon, output_dir=out)
    print(f"[colmap_to_ns] colmap_to_json frames = {n}")

    # 校验
    meta = json.loads((out / "transforms.json").read_text())
    ks = {k: meta.get(k) for k in ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "ply_file_path", "applied_transform")}
    print("[colmap_to_ns] meta:", json.dumps(ks, indent=None, default=str)[:400])
    print(f"[colmap_to_ns] frames={len(meta['frames'])} 首帧 file_path={meta['frames'][0]['file_path']}")


if __name__ == "__main__":
    main()
