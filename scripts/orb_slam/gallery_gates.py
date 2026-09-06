#!/usr/bin/env python
"""Five-panel gallery for the SAM-gate A/B: 原图 / 重建 / 真值 / reg / gates.

Compares the canonical reg run (office0_320_sam20k_reg, mIoU 0.240) against the
tightened-SAM-gates run (office0_320_sam20k_gates, mIoU 0.213). Both prediction
columns come from eval_miou_room npz dumps sharing the *identical* ``--prompts``
ordering so the 13-class palette aligns; the RGB 重建图 is re-rendered live from
``--config`` (the gates run's checkpoint).

Colour helpers / legend are shared with gallery_five_panels.py.

Usage:
    gallery_gates.py --config outputs/office0_320_sam20k_gates/.../config.yml \
        --ns data/office0_est320_ns \
        --gt-dir data/office0_est320/frames/semantic \
        --reg-npz results/orb_ate/reg_cmp/new_npz \
        --gates-npz results/orb_ate/gates_cmp \
        [--miou-reg results/orb_ate/reg_cmp/new_miou.txt] \
        [--miou-gates results/orb_ate/gates_cmp/new_miou.txt] \
        --frames 0 66 120 168 240 288 \
        --out results/orb_ate/gates_cmp/gallery
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

from gallery_five_panels import (  # noqa: E402
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
    ap.add_argument("--config", type=Path, required=True, help="gates run config.yml (重建图列)")
    ap.add_argument("--ns", type=Path, default=Path("data/office0_est320_ns"))
    ap.add_argument("--gt-dir", type=Path, default=Path("data/office0_est320/frames/semantic"))
    ap.add_argument("--reg-npz", type=Path, required=True, help="reg(+正则) 预测 npz dir")
    ap.add_argument("--gates-npz", type=Path, required=True, help="gates(门限调紧) 预测 npz dir")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--miou-reg", type=Path, default=None)
    ap.add_argument("--miou-gates", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/gates_cmp/gallery"))
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

    mreg, mgates = (_miou(p) if p else None for p in (args.miou_reg, args.miou_gates))
    iou_str = ""
    if mreg is not None:
        iou_str += f" | mIoU reg(+空间正则)={mreg:.3f}"
    if mgates is not None:
        iou_str += f" | mIoU gates(门限调紧)={mgates:.3f}"

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

        regcol = predcol(args.reg_npz)
        gcol = predcol(args.gates_npz)

        za, zb = np.load(str(args.reg_npz / f"{k:04d}_scores.npz")), np.load(str(args.gates_npz / f"{k:04d}_scores.npz"))
        va, vb = za["valid"].astype(bool), zb["valid"].astype(bool)
        lit = va & vb
        chg = 100.0 * (za["argmax"][lit] != zb["argmax"][lit]).mean() if lit.any() else 0.0

        imgs = [orig, recon, gtcol, regcol, gcol]
        tags = ["orig", "recon", "gt", "reg", "gates"]
        names = ["原图", "重建图", "语义真值", "reg · +空间正则 (0.240)", "gates · SAM门限调紧 (0.213)"]
        cells = []
        for im, tag, nm in zip(imgs, tags, names):
            rel = f"panels/{k:04d}_{tag}.png"
            cv2.imwrite(str(out / rel), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cells.append(imgcell(rel, nm))
        five = np.concatenate([cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                                  value=(30, 30, 30)) for im in imgs], axis=1)
        cv2.imwrite(str(out / f"{k:04d}_five.png"), cv2.cvtColor(five, cv2.COLOR_RGB2BGR))
        blocks.append(
            f'<div style="margin:28px auto;max-width:1700px">'
            f'<h3 style="color:#eee;margin:6px 2px">frame {k} '
            f'<span style="color:#9aa;font-weight:400">· reg→gates 点亮像素改判 {chg:.1f}%</span></h3>'
            f'<div style="display:flex;gap:6px">{"".join(cells)}</div></div>')
        print(f"frame {k} done", flush=True)

    title = ("office0 · 320 · 20k —— SAM-AMG 门限调紧 A/B：原图 / 重建 / 语义真值 / "
             "reg(标准参考) / gates(stability 0.97 · iou 0.88 · overlap 0.5)")
    (out / "index.html").write_text(
        "<!doctype html><html><body style='background:#111;color:#ddd;font-family:sans-serif'>"
        "<h2>" + title + iou_str + "</h2>"
        f"<div style='max-width:1700px'>{legend}</div>"
        + "".join(blocks) + "</body></html>")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
