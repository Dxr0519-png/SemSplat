#!/usr/bin/env python
"""Five-panel labeled gallery: 原图 / 重建图 / 语义真值 / 语义预测 / 小幅清理.

One labeled row per frame, so it is unambiguous which image is which. Reuses
already-computed artifacts where possible (no retraining, no teacher run):

  * 原图         -- dataset input RGB        data/<scene>_ns/images/{k:06d}.png
  * 重建图       -- 3DGS RGB render of the given config's checkpoint (model reload)
  * 语义真值     -- Replica GT class-id map, colored for the prompt classes,
                    everything else grey (classes outside the prompt set)
  * 语义预测     -- baseline rendered 13-class argmax from <base>/<k>_scores.npz
  * 小幅清理     -- mild connected-island cleanup of that prediction
                    (remove_enclosed_small_islands, hosts wall+ceiling, default
                    threshold) from <mild>/<k>_clean_scores.npz

Captions live *under* each image in the HTML (not baked into the pixels). The
13-class colour palette matches the earlier A/B galleries (HSV golden angle), so
colours line up across pages.

Usage:
    gallery_five_panels.py --config outputs/.../config.yml \
        --ns data/office0_est320_ns \
        --gt-dir data/office0_est320/frames/semantic \
        --base-npz results/orb_ate/vis_ab_base_npz \
        --mild-npz results/orb_ate/vis_ab/clean \
        --frames 0 45 90 120 150 200 260 \
        --out results/orb_ate/vis_five
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

# 13 prompt classes, index == npz argmax column order (matches earlier galleries)
GTIDS = [76, 20, 80, 44, 93, 40, 31, 47, 37, 87, 98, 22, 35]
CLS_ZH = ["沙发 sofa", "椅子 chair", "桌子 table", "绿植 plant", "墙 wall",
          "地板 floor", "天花板 ceiling", "台灯 lamp", "门 door", "电视屏 tv",
          "地毯 rug", "挂钟 clock", "笔筒 desk"]
GRAY = np.array([28, 28, 28], np.uint8)   # GT pixels outside the prompt set
UNLIT = np.array([40, 40, 40], np.uint8)  # no-alpha pixels in predictions
DARK = "#0b0b0f"


def palette(n: int):
    cols = []
    for i in range(n):
        h = (i * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.92)
        cols.append((np.array([r, g, b]) * 255).astype(np.uint8))
    return cols


def colorize_argmax(argmax: np.ndarray, valid: np.ndarray, colors) -> np.ndarray:
    """Colour a class-id map (indices 0..C-1, 255 = unlit/bg) consistently."""
    rgb = np.zeros((*argmax.shape, 3), np.uint8)
    for i, c in enumerate(colors):
        rgb[argmax == i] = c
    rgb[argmax == 255] = UNLIT
    rgb[~valid] = UNLIT
    return rgb


def colorize_gt(g: np.ndarray, colors) -> np.ndarray:
    rgb = np.zeros((*g.shape, 3), np.uint8)
    for i, gid in enumerate(GTIDS):
        rgb[g == gid] = colors[i]
    rgb[~np.isin(g, GTIDS)] = GRAY
    return rgb


def render_rgb(model, cam) -> np.ndarray:
    out = model.get_outputs_for_camera(cam)
    rgb = out["rgb"].clamp(0, 1).detach().cpu().numpy()
    return (rgb * 255).round().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="baseline config.yml")
    ap.add_argument("--ns", type=Path, default=Path("data/office0_est320_ns"))
    ap.add_argument("--gt-dir", type=Path, default=Path("data/office0_est320/frames/semantic"))
    ap.add_argument("--base-npz", type=Path, default=Path("results/orb_ate/vis_ab_base_npz"))
    ap.add_argument("--mild-npz", type=Path, default=Path("results/orb_ate/vis_ab/clean"))
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 45, 90, 120, 150, 200, 260])
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/vis_five"))
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

    # legend chips (one per class, then the two grey keys)
    legend = "<div style='margin:10px'>" + " &nbsp; ".join(
        f'<span style="background:rgb{tuple(c)};color:#000;padding:2px 8px;'
        f'border-radius:4px">{nm}</span>'
        for c, nm in zip(colors, CLS_ZH))
    legend += ("<br/><span style='background:#1c1c1c;color:#bbb;padding:2px 8px'>语义真值灰 = 不在13类内的GT</span>"
               " &nbsp; "
               "<span style='background:#282828;color:#bbb;padding:2px 8px'>预测/清理深灰 = alpha≤0.5 未点亮</span>"
               "</div>")

    def imgcell(png_rel: str, caption: str, note: str = "") -> str:
        n = ('<div style="color:#9aa">' + note + '</div>') if note else ""
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

        # 原图
        orig = cv2.imread(str(args.ns / f["file_path"]))[:, :, ::-1]
        # 重建图
        recon = render_rgb(model, cam)
        # 语义真值
        gtf = args.gt_dir / f"{k}.png"
        if not gtf.exists():
            gtf = args.gt_dir / f"{k:06d}.png"
        gt = cv2.imread(str(gtf), -1)
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        gtcol = colorize_gt(gt, colors)
        # 语义预测 (baseline)
        z = np.load(str(args.base_npz / f"{k:04d}_scores.npz"))
        pred = colorize_argmax(z["argmax"].astype(np.int64), z["valid"].astype(bool), colors)
        # 小幅清理 (温和)
        zm = np.load(str(args.mild_npz / f"{k:04d}_clean_scores.npz"))
        mild = colorize_argmax(zm["argmax"].astype(np.int64), zm["valid"].astype(bool), colors)

        imgs = [orig, recon, gtcol, pred, mild]
        tags = ["orig", "recon", "gt", "pred", "mild"]
        names = ["原图", "重建图", "语义真值", "语义预测", "小幅清理 · 温和"]
        cells = []
        for (im, tag, nm) in zip(imgs, tags, names):
            rel = f"panels/{k:04d}_{tag}.png"
            cv2.imwrite(str(out / rel), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cells.append(imgcell(rel, nm))
        five = np.concatenate([cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                                  value=(30, 30, 30)) for im in imgs], axis=1)
        cv2.imwrite(str(out / f"{k:04d}_five.png"), cv2.cvtColor(five, cv2.COLOR_RGB2BGR))
        blocks.append(
            f'<div style="margin:28px auto;max-width:1500px">'
            f'<h3 style="color:#eee;margin:6px 2px">frame {k}</h3>'
            f'<div style="display:flex;gap:6px">{"".join(cells)}</div></div>')
        print(f"frame {k} done", flush=True)

    (out / "index.html").write_text(
        "<!doctype html><html><body style='background:#111;color:#ddd;font-family:sans-serif'>"
        "<h2>office0 · 320 · 6k baseline —— 原图 / 重建 / 语义真值 / 语义预测 / 小幅清理(温和 0.08%)</h2>"
        f"<div style='max-width:1500px'>{legend}</div>"
        f'<p style="color:#999;margin:4px">每行同一帧 5 张图；重建 = 3DGS 该帧渲染，与</p>'
        + "".join(blocks) + "</body></html>")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
