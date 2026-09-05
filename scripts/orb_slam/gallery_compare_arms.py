#!/usr/bin/env python
"""Side-by-side before/after gallery for the SAM-AMG semantic A/B.

One labelled row per frame. Columns:
    原图 | 重建图 | 语义真值 | 臂1(基线/旧) | 臂2(新arm) | 差异
Two arms are compared with the *same* prompt set (P_new) so the difference
column isolates the teacher/mask change, not the prompts. The difference panel
paints disagreeing pixels in arm-2's class colour on a grey "agreed" ground and
reports changed% (of lit pixels) per frame in the caption.

Arms are npz dumps from eval_miou_room.py --dump-scores (schema keys
scores/argmax/top_score/margin/valid + prompts.json sidecar). GT colouring and
the 13-class palette are shared with gallery_five_panels.py, so colours line up.

Usage:
    gallery_compare_arms.py --config outputs/office0_320_sam6k_amg/.../config.yml \
        --ns data/office0_est320_ns --gt-dir data/office0_est320/frames/semantic \
        --frames 0 48 96 144 192 240 288 \
        --arm baseline=results/orb_ate/ab_amg/baseline_new \
        --arm amg=results/orb_ate/ab_amg/amg_new \
        --out results/orb_ate/ab_amg/gallery
"""
from __future__ import annotations

import argparse
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

from gallery_five_panels import (  # noqa: E402  (same dir, shared palette/GT helpers)
    GTIDS,
    CLS_ZH,
    DARK,
    colorize_argmax,
    colorize_gt,
    palette,
    render_rgb,
)


def delta_overlay(a: np.ndarray, b: np.ndarray, colors) -> tuple[np.ndarray, float]:
    """Agreed lit pixels grey; arm-2 colour where they disagree. Returns rgb + changed%."""
    rgb = np.zeros((*a.shape, 3), np.uint8)
    rgb[..., :] = (18, 18, 20)   # both unlit / no geometry
    lit = (a != 255) & (b != 255)
    same = lit & (a == b)
    diff = lit & (a != b)
    only_a = (a != 255) & (b == 255)  # lit in arm1 only -> dim arm-1 colour
    only_b = (b != 255) & (a == 255)  # lit in arm2 only -> arm-2 colour
    rgb[same] = (128, 128, 128)
    for i, c in enumerate(colors):
        rgb[diff & (b == i)] = c
        rgb[only_a & (a == i)] = c // 2
        rgb[only_b & (b == i)] = c
    changed = 100.0 * diff.sum() / max(lit.sum(), 1)
    return rgb, changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="model config (usually the new arm) for recon RGB")
    ap.add_argument("--ns", type=Path, default=Path("data/office0_est320_ns"))
    ap.add_argument("--gt-dir", type=Path, default=Path("data/office0_est320/frames/semantic"))
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="label=npz_dir",
                    help="repeatable; exactly two arms with identical prompts.json")
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/ab_amg/gallery"))
    args = ap.parse_args()

    if len(args.arm) != 2:
        raise SystemExit("need exactly two --arm args (label=npz_dir)")
    arms = []
    for spec in args.arm:
        label, path = spec.split("=", 1)
        d = Path(path)
        prompts = json.loads((d / "prompts.json").read_text())["prompts"]
        if arms and arms[0]["prompts"] != prompts:
            raise SystemExit(f"arm prompt sets differ:\n  {arms[0]['label']}: {arms[0]['prompts']}\n  {label}: {prompts}")
        arms.append({"label": label, "dir": d, "prompts": prompts})
    if len(arms[0]["prompts"]) != len(GTIDS):
        raise SystemExit(f"arm prompt count {len(arms[0]['prompts'])} != {len(GTIDS)} GT ids")

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
        for c, nm in zip(colors, CLS_ZH))
    legend += ("<br/><span style='background:#808080;color:#111;padding:2px 8px'>差异列 灰 = 两臂一致</span>"
               " &nbsp; "
               "<span style='background:#808080;color:#111;padding:2px 8px'>差异列 彩色 = 仅新arm被判为该色（旧arm不同）</span>"
               " &nbsp; "
               "<span style='background:#1c1c1c;color:#bbb;padding:2px 8px'>语义真值灰 = 不在13类内的GT</span>"
               " &nbsp; "
               "<span style='background:#282828;color:#bbb;padding:2px 8px'>深灰 = alpha≤0.5 未点亮</span>"
               "</div>")

    def imgcell(png_rel: str, caption: str, note: str = "") -> str:
        n = (f'<div style="color:#9aa">{note}</div>') if note else ""
        return (f'<div style="flex:1;min-width:0"><img src="{png_rel}" style="width:100%;'
                f'display:block;border:1px solid #333"/>'
                f'<div style="text-align:center;padding:5px 2px;background:{DARK};'
                f'font-weight:600;color:#fff">{caption}</div>{n}</div>')

    a1, a2 = arms[0], arms[1]
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

        za = np.load(str(a1["dir"] / f"{k:04d}_scores.npz"))
        zb = np.load(str(a2["dir"] / f"{k:04d}_scores.npz"))
        ma = za["argmax"].astype(np.int64); va = za["valid"].astype(bool)
        mb = zb["argmax"].astype(np.int64); vb = zb["valid"].astype(bool)
        if ma.shape != mb.shape:
            raise SystemExit(f"frame {k}: arm shapes differ {ma.shape} vs {mb.shape}")
        pred_a = colorize_argmax(ma, va, colors)
        pred_b = colorize_argmax(mb, vb, colors)
        delta, changed = delta_overlay(ma, mb, colors)

        imgs = [orig, recon, gtcol, pred_a, pred_b, delta]
        tags = ["orig", "recon", "gt", "arm1", "arm2", "delta"]
        names = ["原图", "重建图", "语义真值", f"臂1 · {a1['label']}", f"臂2 · {a2['label']}", "差异(不一致=臂2色)"]
        cells = []
        for (im, tag, nm) in zip(imgs, tags, names):
            rel = f"panels/{k:04d}_{tag}.png"
            cv2.imwrite(str(out / rel), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cells.append(imgcell(rel, nm, ""))
        strip = np.concatenate([cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                                   value=(30, 30, 30)) for im in imgs], axis=1)
        cv2.imwrite(str(out / f"{k:04d}_six.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        blocks.append(
            f'<div style="margin:28px auto;max-width:2000px">'
            f'<h3 style="color:#eee;margin:6px 2px">frame {k} —— 不一致占已点亮像素 '
            f'<span style="color:#ffb4a2">{changed:.1f}%</span></h3>'
            f'<div style="display:flex;gap:6px">{"".join(cells)}</div></div>')
        print(f"frame {k}: changed {changed:.1f}%", flush=True)

    (out / "index.html").write_text(
        "<!doctype html><html><body style='background:#111;color:#ddd;font-family:sans-serif'>"
        f"<h2>office0 · 320 · 6k —— SAM-AMG+上下文裁剪 A/B（两臂同用空间化 P_new，差异列隔离教师改动）</h2>"
        f'<p style="color:#999;margin:2px">臂1 = {a1["label"]}（npz {a1["dir"]}）　臂2 = {a2["label"]}（npz {a2["dir"]}）</p>'
        f'<p style="color:#999;margin:2px">每个底部 caption 显示该列说明；差异列 灰=一致、彩色=仅臂2 判为该类</p>'
        f"<div style='max-width:2000px'>{legend}</div>"
        + "".join(blocks) + "</body></html>")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
