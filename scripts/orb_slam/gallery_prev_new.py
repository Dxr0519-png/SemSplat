#!/usr/bin/env python
"""Five-panel A/B gallery: 原图 / 重建图 / 语义真值 / 之前语义预测 / 当前语义预测.

Comparison page for the "3D spatial regularization (polish)" experiment: the two
semantic-prediction columns come from pre-dumped npz dirs (the eval_miou_room.py
schema: ``{k:04d}_scores.npz`` with keys ``argmax``/``valid``), so both runs must
have been dumped with the *identical* ``--prompts`` ordering for colours to align.
The RGB 重建图 is re-rendered live from ``--config`` (the new run's checkpoint).

Optional ``--geo-npz`` adds a 6th column: the same run + the query-time geometric
stuff-plane prior (eval_geo_prior.py --prior norm dump). All semantic columns
must share the same prompts ordering / palette.

Colour palette / legend / helpers are shared with gallery_five_panels.py, so the
class colours line up across every gallery page in results/orb_ate/.

Usage:
    gallery_prev_new.py --config outputs/office0_320_sam20k_reg/.../config.yml \
        --ns data/office0_est320_ns \
        --gt-dir data/office0_est320/frames/semantic \
        --prev-npz results/orb_ate/reg_cmp/prev_npz \
        --new-npz results/orb_ate/reg_cmp/new_npz \
        [--geo-npz results/orb_ate/reg_cmp/geo_npz] \
        [--miou-prev results/orb_ate/reg_cmp/prev_miou.txt] \
        [--miou-new results/orb_ate/reg_cmp/new_miou.txt] \
        [--miou-geo results/orb_ate/reg_cmp/geo_miou.txt] \
        --frames 0 30 66 120 168 240 288 \
        --out results/orb_ate/reg_cmp/gallery
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

from gallery_five_panels import (  # noqa: E402  (same dir when run as a script)
    CLS_ZH,
    DARK,
    GTIDS,
    colorize_argmax,
    colorize_gt,
    palette,
    render_rgb,
)
from nerfstudio.cameras.cameras import Cameras  # noqa: E402
from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402

GRAY_UNLIT_NOTES = (
    "<br/><span style='background:#1c1c1c;color:#bbb;padding:2px 8px'>语义真值灰 = 不在13类内的GT</span>"
    " &nbsp; "
    "<span style='background:#282828;color:#bbb;padding:2px 8px'>预测深灰 = alpha≤0.5 未点亮</span>"
)


def _miou(path: Path) -> float | None:
    """Parse 'mIoU (classes present in GT) <value>' from an eval_miou_room text out."""
    try:
        for ln in path.read_text().splitlines():
            m = re.search(r"mIoU.*\s([0-9]+\.[0-9]+)\s", ln)
            if m:
                return float(m.group(1))
    except OSError:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="new(+正则) run config.yml (重建图列)")
    ap.add_argument("--ns", type=Path, default=Path("data/office0_est320_ns"))
    ap.add_argument("--gt-dir", type=Path, default=Path("data/office0_est320/frames/semantic"))
    ap.add_argument("--prev-npz", type=Path, required=True, help="之前(20k-AMG) 预测 npz dir")
    ap.add_argument("--new-npz", type=Path, required=True, help="当前(+正则) 预测 npz dir")
    ap.add_argument("--geo-npz", type=Path, default=None, help="可选:当前+几何先验 预测 npz dir (加第6列)")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--miou-prev", type=Path, default=None)
    ap.add_argument("--miou-new", type=Path, default=None)
    ap.add_argument("--miou-geo", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/reg_cmp/gallery"))
    args = ap.parse_args()

    colors = palette(len(GTIDS))
    meta = json.loads((args.ns / "transforms.json").read_text())
    byf = {int(Path(f["file_path"]).stem): f for f in meta["frames"]}
    W, H = int(meta["w"]), int(meta["h"])

    _, pipeline, _, _ = eval_setup(args.config)
    model = pipeline.model
    model.eval()

    out = args.out
    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    legend = "<div style='margin:10px'>" + " &nbsp; ".join(
        f'<span style="background:rgb{tuple(c)};color:#000;padding:2px 8px;'
        f'border-radius:4px">{nm}</span>'
        for c, nm in zip(colors, CLS_ZH)) + GRAY_UNLIT_NOTES + "</div>"

    mprev, mnew, mgeo = (_miou(p) if p else None for p in (args.miou_prev, args.miou_new, args.miou_geo))
    iou_str = ""
    if mprev is not None or mnew is not None or mgeo is not None:
        iou_str = (f" | mIoU 之前={mprev:.3f} (20k-AMG)" if mprev is not None else "")
        iou_str += (f" | mIoU 当前={mnew:.3f} (+正则)" if mnew is not None else "")
        iou_str += (f" | mIoU 当前+几何先验={mgeo:.3f}" if mgeo is not None else "")

    def imgcell(png_rel: str, caption: str, note: str = "") -> str:
        n = (f'<div style="color:#9aa;text-align:center">{note}</div>') if note else ""
        return (f'<div style="flex:1;min-width:0"><img src="{png_rel}" style="width:100%;'
                f'display:block;border:1px solid #333"/>'
                f'<div style="text-align:center;padding:5px 2px;background:{DARK};'
                f'font-weight:600;color:#fff">{caption}</div>{n}</div>')

    blocks = []
    for k in args.frames:
        f = byf[k]
        M = np.asarray(f["transform_matrix"], float)[:3, :4]
        cam = Cameras(camera_to_worlds=torch.tensor(M)[None].float().to(model.device),
                      fx=meta["fl_x"], fy=meta.get("fl_y", meta["fl_x"]),
                      cx=meta["cx"], cy=meta["cy"], width=W, height=H).to(model.device)

        orig = cv2.imread(str(args.ns / f["file_path"]))[:, :, ::-1]
        recon = render_rgb(model, cam)
        gtf = args.gt_dir / f"{k}.png"
        if not gtf.exists():
            gtf = args.gt_dir / f"{k:06d}.png"
        gt = cv2.imread(str(gtf), -1)
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        gtcol = colorize_gt(gt, colors)

        def predcol(npz_dir: Path):
            z = np.load(str(npz_dir / f"{k:04d}_scores.npz"))
            return colorize_argmax(z["argmax"].astype(np.int64), z["valid"].astype(bool), colors)

        prev = predcol(args.prev_npz)
        new = predcol(args.new_npz)
        geo = predcol(args.geo_npz) if args.geo_npz is not None else None

        # fraction of lit pixels that flipped label (previous vs current)
        za, zb = np.load(str(args.prev_npz / f"{k:04d}_scores.npz")), np.load(str(args.new_npz / f"{k:04d}_scores.npz"))
        va, vb = za["valid"].astype(bool), zb["valid"].astype(bool)
        lit = va & vb
        chg = 100.0 * (za["argmax"][lit] != zb["argmax"][lit]).mean() if lit.any() else 0.0

        # fraction of lit pixels the geometric prior re-buckets (vs current)
        geo_note = ""
        if geo is not None:
            zc = np.load(str(args.geo_npz / f"{k:04d}_scores.npz"))
            vc = zc["valid"].astype(bool)
            litg = vb & vc
            chgg = 100.0 * (zc["argmax"][litg] != zb["argmax"][litg]).mean() if litg.any() else 0.0
            geo_note = f"vs 当前 改判 {chgg:.1f}%"

        imgs = [orig, recon, gtcol, prev, new]
        tags = ["orig", "recon", "gt", "prev", "new"]
        names = ["原图", "重建图", "语义真值", "之前语义预测 · 20k-AMG", "当前语义预测 · +空间正则"]
        notes = ["", "", "", "", ""]
        if geo is not None:
            imgs.append(geo)
            tags.append("geo")
            names.append("当前 + 查询期几何先验 · 法向")
            notes.append(geo_note)
        cells = []
        for im, tag, nm, note in zip(imgs, tags, names, notes):
            rel = f"panels/{k:04d}_{tag}.png"
            cv2.imwrite(str(out / rel), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cells.append(imgcell(rel, nm, note))
        five = np.concatenate([cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                                  value=(30, 30, 30)) for im in imgs], axis=1)
        cv2.imwrite(str(out / f"{k:04d}_five.png"), cv2.cvtColor(five, cv2.COLOR_RGB2BGR))
        blocks.append(
            f'<div style="margin:28px auto;max-width:1700px">'
            f'<h3 style="color:#eee;margin:6px 2px">frame {k} '
            f'<span style="color:#9aa;font-weight:400">· 点亮像素改判 {chg:.1f}%</span></h3>'
            f'<div style="display:flex;gap:6px">{"".join(cells)}</div></div>')
        print(f"frame {k} done", flush=True)

    title = ("office0 · 320 · 20k —— 空间正则化抛光 对比：原图 / 重建 / 语义真值 / "
             "之前(20k-AMG) / 当前(+Depth-aware TV + Opacity Entropy)")
    if args.geo_npz is not None:
        title = ("office0 · 320 · 20k —— 空间正则化 + 查询期几何先验 对比：原图 / 重建 / 语义真值 / "
                 "之前(20k-AMG) / 当前(+空间正则) / 当前+法向先验")
    (out / "index.html").write_text(
        "<!doctype html><html><body style='background:#111;color:#ddd;font-family:sans-serif'>"
        "<h2>" + title + iou_str + "</h2>"
        f"<div style='max-width:1700px'>{legend}</div>"
        + "".join(blocks) + "</body></html>")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
