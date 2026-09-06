#!/usr/bin/env python
"""Three-panel gallery for a GT-pose demo run: 原图 / 重建图 / 语义图 (+PSNR).

Targets the standard-reg run on ``data/replica_demo_gl_ns`` (no semantic GT, so
the semantic column is *open-vocabulary*: query words -> CLIP-text vs per-Gauss
sem-feature cosine argmax, rasterised per camera). PSNR is computed live between
the native 原图 and the rendered 重建图 over alpha>0.5 pixels only.

Semantic panel / colours/legend mirror ``semsplat-query`` (same prompts are used
so it can be eyeballed against the earlier ``results/demo3`` output).

Usage:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/orb_slam/gallery_demo_reg.py \
      --config outputs/demo_gt_20k_reg/replica_demo_gl_ns/semsplat-replica/<ts>/config.yml \
      --ns data/replica_demo_gl_ns \
      --prompts "wall,floor,ceiling,sofa,chair,table,lamp,window" \
      --frames 0 50 100 150 200 250 299 \
      --out results/orb_ate/demo_reg/gallery
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

from nerfstudio.cameras.cameras import Cameras  # noqa: E402
from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402

from semsplat.inference import scoring  # noqa: E402
from semsplat.teachers.clip_local_teacher import (  # noqa: E402
    encode_text_prompts,
    resolve_text_model_name,
)

UNLIT = np.array([40, 40, 40], np.uint8)
DARK = "#0b0b0f"


def palette(n: int) -> list:
    cols = []
    for i in range(n):
        h = (i * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.92)
        cols.append((np.array([r, g, b]) * 255).astype(np.uint8))
    return cols


def render_rgb(model, cam) -> np.ndarray:
    out = model.get_outputs_for_camera(cam)
    rgb = out["rgb"].clamp(0, 1).detach().cpu().numpy()
    return (rgb * 255).round().astype(np.uint8)


def colorize_argmax(argmax: np.ndarray, valid: np.ndarray, colors) -> np.ndarray:
    rgb = np.zeros((*argmax.shape, 3), np.uint8)
    for i, c in enumerate(colors):
        rgb[argmax == i] = c
    rgb[argmax == 255] = UNLIT
    rgb[~valid] = UNLIT
    return rgb


def psnr_valid(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    """PSNR (dB) over valid pixels; a,b uint8 RGB, valid bool [H,W]."""
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    mse = float((((a - b) ** 2).mean(axis=-1)[valid].mean()))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ns", type=Path, default=Path("data/replica_demo_gl_ns"))
    ap.add_argument("--prompts", required=True, help="comma-separated open-vocab query words")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/demo_reg/gallery"))
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    colors = palette(len(prompts))

    meta = json.loads((args.ns / "transforms.json").read_text())
    byf = {int(Path(f["file_path"]).stem): f for f in meta["frames"]}
    missing = [k for k in args.frames if k not in byf]
    if missing:
        raise SystemExit(f"missing frame ids in dataset: {missing}")
    W, H = int(meta["w"]), int(meta["h"])

    _, pipeline, _, _ = eval_setup(args.config)
    model = pipeline.model
    model.eval()
    dev = model.device

    print(">> encoding prompts:", prompts, flush=True)
    text_model = resolve_text_model_name(model.config)
    emb_dict = encode_text_prompts(prompts, text_model, device=str(dev))
    emb = torch.stack([emb_dict[p] for p in prompts], dim=0).to(dev)

    panels = args.out / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    legend = "<div style='margin:10px'>" + " &nbsp; ".join(
        f'<span style="background:rgb{tuple(c)};color:#000;padding:2px 8px;'
        f'border-radius:4px">{w}</span>' for c, w in zip(colors, prompts)
    ) + (
        "<br/><span style='background:#282828;color:#bbb;padding:2px 8px'>语义图深灰 = "
        "alpha≤0.5 未点亮 / 黑格 = 该词未胜出则按 argmax</span></div>"
    )

    def imgcell(png_rel: str, caption: str) -> str:
        return (f'<div style="flex:1;min-width:0"><img src="{png_rel}" style="width:100%;'
                f'display:block;border:1px solid #333"/>'
                f'<div style="text-align:center;padding:5px 2px;background:{DARK};'
                f'font-weight:600;color:#fff">{caption}</div></div>')

    psnrs = []
    blocks = []
    for k in args.frames:
        f = byf[k]
        M = np.asarray(f["transform_matrix"], float)[:3, :4]
        cam = Cameras(camera_to_worlds=torch.tensor(M)[None].float().to(dev),
                      fx=meta["fl_x"], fy=meta.get("fl_y", meta["fl_x"]),
                      cx=meta["cx"], cy=meta["cy"], width=W, height=H).to(dev)

        orig = cv2.imread(str(args.ns / f["file_path"]))[:, :, ::-1]
        recon = render_rgb(model, cam)

        maps, alpha = scoring.semantic_maps_from_prompts(model, cam, emb)
        maps = maps.detach().cpu().numpy()
        alpha = alpha.detach().cpu().numpy()
        valid = alpha[..., 0] > 0.5
        argmax = np.full((H, W), 255, np.uint16)
        argmax[valid] = maps.argmax(axis=-1)[valid]
        semcol = colorize_argmax(argmax, valid, colors)

        p = psnr_valid(orig, recon, valid)
        psnrs.append(p)

        imgs = [orig, recon, semcol]
        tags = ["orig", "recon", "sem"]
        names = ["原图", f"重建图", f"语义图 · 开放词表 (argmax)"]
        cells = []
        for im, tag, nm in zip(imgs, tags, names):
            rel = f"panels/{k:04d}_{tag}.png"
            cv2.imwrite(str(args.out / rel), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cells.append(imgcell(rel, nm))
        blocks.append(
            f'<div style="margin:24px auto;max-width:1700px">'
            f'<h3 style="color:#eee;margin:6px 2px">frame {k} · PSNR {p:.2f} dB</h3>'
            f'<div style="display:flex;gap:6px">{"".join(cells)}</div></div>')
        print(f"frame {k}  PSNR {p:.2f} dB", flush=True)

    mean_p = float(np.mean(psnrs)) if psnrs else float("nan")
    (args.out / "index.html").write_text(
        "<!doctype html><html><body style='background:#111;color:#ddd;font-family:sans-serif'>"
        "<h2>replica_demo_gl_ns · 标准 reg(20k, samclip+空间正则) —— 原图 / 重建图 / 语义图"
        f" · 平均PSNR {mean_p:.2f} dB (alpha>0.5)</h2>"
        f"<div style='max-width:1700px'>{legend}</div>"
        + "".join(blocks) + "</body></html>")
    print(f"wrote {args.out / 'index.html'}  (mean PSNR {mean_p:.2f} dB)")


if __name__ == "__main__":
    main()
