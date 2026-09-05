#!/usr/bin/env python
"""Effect-comparison gallery for the geometric target-injection arm (injCeil).

Purpose (user asked for a visual before deciding the next step after Arm B′):
show the ceiling-fix vs wall-flood trade-off of ceiling-only target injection
next to the baselines, so the decision "is 0.196 mIoU worth a fixed ceiling" can
be made on images, not just numbers.

Six visual columns per frame (same 13-class palette as all earlier galleries):
    原图 | 语义真值 | reg 0.240 | hinge A′ 0.236 | injCeil 0.196 | geo oracle 0.255
plus a 7th delta column injCeil↔geo: since geo oracle is the no-flood reference
(query-time relabel of reg, walls untouched), agreement is grey and the ceiling
colour splashes across *walls* where injCeil disagrees = the 2.6M-px wall flood
made visible.

Everything is read from existing eval artifacts — no model reload, no retrain:
  * 原图         data/office0_est320_ns/images/{k:06d}.png (via transforms.json)
  * 语义真值     data/office0_est320/frames/semantic/{k}.png (Replica GT ids)
  * arms         eval_miou_room.py --dump-scores npz dirs, schema keys
                  scores/argmax/top_score/margin/valid (+prompts.json sidecar);
                  argmax 255 = unlit. argmax index i <-> GTIDS[i] as usual.
A one-line badge under each frame header reports per-arm *ceiling* IoU over lit
pixels (same formula as eval_miou_room.py, class idx 6 / GT id 31) and the
injCeil flood FP (pred=ceiling on non-ceiling GT), so the per-frame trade-off is
quantified right under the pictures.

Usage:
    gallery_injceil.py \
        --ns data/office0_est320_ns \
        --gt-dir data/office0_est320/frames/semantic \
        --reg-dir results/orb_ate/reg_cmp/new_npz \
        --hinge-dir results/orb_ate/negc_aprime/npz \
        --inj-dir results/orb_ate/injceil/npz \
        --geo-dir results/orb_ate/reg_cmp/geo_npz \
        --frames 42 60 90 102 156 270 294 \
        --out results/orb_ate/gallery_injceil
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- 13-class palette & GT helpers (copied from gallery_five_panels so this
# script needs no nerfstudio/torch imports and no model reload)
GTIDS = [76, 20, 80, 44, 93, 40, 31, 47, 37, 87, 98, 22, 35]
CLS_ZH = ["沙发 sofa", "椅子 chair", "桌子 table", "绿植 plant", "墙 wall",
          "地板 floor", "天花板 ceiling", "台灯 lamp", "门 door", "电视屏 tv",
          "地毯 rug", "挂钟 clock", "笔筒 desk"]
GRAY = np.array([28, 28, 28], np.uint8)   # GT pixels outside the prompt set
UNLIT = np.array([40, 40, 40], np.uint8)  # no-alpha pixels in predictions
DARK = "#0b0b0f"
CEIL_IDX = 6          # prompt/GTIDS column of "flat white ceiling"
CEIL_GID = GTIDS[CEIL_IDX]  # 31


def palette(n: int):
    cols = []
    for i in range(n):
        h = (i * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.92)
        cols.append((np.array([r, g, b]) * 255).astype(np.uint8))
    return cols


def colorize_argmax(argmax: np.ndarray, valid: np.ndarray, colors) -> np.ndarray:
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


def delta_overlay(a: np.ndarray, b: np.ndarray, colors) -> tuple[np.ndarray, float]:
    """a = reference (geo oracle), b = arm (injCeil). Agreed lit pixels grey;
    b's class colour where they disagree. Returns rgb + changed% of lit."""
    rgb = np.zeros((*a.shape, 3), np.uint8)
    rgb[..., :] = (18, 18, 20)
    lit = (a != 255) & (b != 255)
    same = lit & (a == b)
    diff = lit & (a != b)
    rgb[same] = (128, 128, 128)
    for i, c in enumerate(colors):
        rgb[diff & (b == i)] = c
    rgb[(a != 255) & (b == 255)] = (90, 30, 30)   # geo lit, injCeil unlit
    rgb[(b != 255) & (a == 255)] = (30, 30, 90)   # injCeil lit, geo unlit
    changed = 100.0 * diff.sum() / max(lit.sum(), 1)
    return rgb, changed


def ceiling_ious(d: Path) -> tuple[list[int], list[int], list[int], dict]:
    """Per-frame ceiling tp/fp/fn across one arm dir + verified prompts order."""
    prompts = json.loads((d / "prompts.json").read_text())["prompts"]
    if prompts[CEIL_IDX] != "flat white ceiling":
        raise SystemExit(f"{d}: prompt[{CEIL_IDX}]={prompts[CEIL_IDX]!r}, want 'flat white ceiling'")
    out = {}
    for npz in sorted(d.glob("*_scores.npz")):
        z = np.load(npz)
        arg = z["argmax"]; valid = z["valid"]
        out[int(npz.stem.split("_")[0])] = (arg, valid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=Path, default=Path("data/office0_est320_ns"))
    ap.add_argument("--gt-dir", type=Path, default=Path("data/office0_est320/frames/semantic"))
    ap.add_argument("--reg-dir", type=Path, required=True)
    ap.add_argument("--hinge-dir", type=Path, required=True)
    ap.add_argument("--inj-dir", type=Path, required=True)
    ap.add_argument("--geo-dir", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/orb_ate/gallery_injceil"))
    args = ap.parse_args()

    # labels carry the aggregate mIoU so each column header is self-explanatory
    ARMS = [
        ("reg", args.reg_dir, "0.240"),
        ("hinge A′", args.hinge_dir, "0.236"),
        ("injCeil", args.inj_dir, "0.196"),
        ("geo oracle", args.geo_dir, "0.255"),
    ]
    arms = [{"key": key, "label": f"{key} · mIoU {miou}",
             "npz": ceiling_ious(d), "dir": d} for key, d, miou in ARMS]

    colors = palette(len(GTIDS))
    meta = json.loads((args.ns / "transforms.json").read_text())
    byf = {int(Path(f["file_path"]).stem): f for f in meta["frames"]}
    W, H = int(meta["w"]), int(meta["h"])

    out = args.out
    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    legend = ("<div style='margin:12px'>" + " &nbsp; ".join(
        f'<span style="background:rgb{tuple(c)};color:#000;padding:2px 8px;'
        f'border-radius:4px">{nm}</span>'
        for c, nm in zip(colors, CLS_ZH))
        + "<br/><span style='color:#999'>语义真值灰 = 不在13类内的 GT；深灰 = 未点亮(alpha≤0.5)。"
        "每帧上方数字为各臂在天花板类上的本帧 IoU（eval 同口径，仅点亮&非void）与 injCeil 溢出的 FP 像素。</span>"
        "<br/><span style='color:#999'>差异列以 geo oracle 为基准：灰 = 两臂一致；"
        "<span style='color:#b0b0ff'>天花板蓝</span>(实际为该类色) = injCeil 判成它而 geo 不同（墙上的天花板色斑 = 墙灌）；"
        "暗红 = geo 点亮而 injCeil 熄；暗蓝 = 反之。</span></div>")

    # ---- aggregate tables from the eval (doc §5), kept static so the page is self-contained
    cls_rows = [
        ("sofa with cushions", "0.422", "0.422", "0.434", "0.340"),
        ("black office chair", "0.865", "0.865", "0.834", "0.699"),
        ("table", "0.648", "0.648", "0.612", "0.490"),
        ("potted green plant", "0.000", "0.000", "0.000", "0.000"),
        ("plain room wall", "0.435", "0.407", "0.380", "0.245"),
        ("floor", "0.057", "0.034", "0.035", "0.041"),
        ("<b>flat white ceiling</b>", "0.005", "0.249", "0.217", "<b>0.280</b>"),
        ("table lamp", "0.178", "0.178", "0.197", "0.042"),
        ("door", "0.128", "0.128", "0.084", "0.160"),
        ("television screen", "0.179", "0.179", "0.119", "0.153"),
        ("rug", "0.200", "0.200", "0.158", "0.102"),
        ("wall clock / desk organizer", "0.000", "0.000", "0.000", "0.000"),
        ("<b>mIoU</b>", "<b>0.240</b>", "<b>0.255</b>", "<b>0.236</b>", "<b>0.196</b>"),
    ]
    head = "<tr><th>class</th><th>reg</th><th>geo oracle</th><th>hinge A′</th><th>injCeil</th></tr>"
    table = ("<table style='border-collapse:collapse;font-size:12px;margin:4px auto'>" + head
             + "".join(f"<tr><td style='padding:1px 10px;border:1px solid #333'>{n}</td>"
                       + "".join(f"<td style='padding:1px 10px;border:1px solid #333;text-align:right'>{v}</td>"
                                 for v in row)
                       + "</tr>" for n, *row in cls_rows) + "</table>")

    def imgcell(png_rel: str, caption: str, accent: bool = False) -> str:
        style = ("flex:1;min-width:0")
        cap_bg = "#1d3d1d" if accent else DARK
        return (f'<div style="{style}"><img src="{png_rel}" style="width:100%;display:block;'
                f'border:1px solid {"#6f6" if accent else "#333"}"/>'
                f'<div style="text-align:center;padding:5px 2px;background:{cap_bg};'
                f'font-weight:600;color:#fff">{caption}</div></div>')

    blocks = []
    for k in args.frames:
        if k not in byf:
            raise SystemExit(f"frame {k} not in transforms.json")
        f = byf[k]
        orig = cv2.imread(str(args.ns / f["file_path"]))[:, :, ::-1]
        gtf = args.gt_dir / f"{k}.png"
        if not gtf.exists():
            gtf = args.gt_dir / f"{k:06d}.png"
        gt = cv2.imread(str(gtf), -1)
        if gt is None:
            raise SystemExit(f"missing GT {gtf}")
        if gt.shape != (H, W):
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        gtcol = colorize_gt(gt, colors)

        cell_stat = {}
        preds = []
        for a in arms:
            z = a["npz"][k]
            arg, valid = z
            if arg.shape != (H, W):
                raise SystemExit(f"{a['key']} frame {k}: argmax {arg.shape} != {(H, W)}")
            preds.append(colorize_argmax(arg, valid, colors))
            # ceiling IoU exactly as eval_miou_room.py for class CEIL_IDX
            p = arg == CEIL_IDX
            g = gt == CEIL_GID
            void = gt == 0
            tp = int((p & g).sum()); fp = int((p & ~g & ~void).sum()); fn = int((g & ~p).sum())
            cell_stat[a["key"]] = (tp, fp, fn)

        # delta = injCeil vs geo oracle argmax (geo = the no-flood reference)
        za = arms[3]["npz"][k]; zb = arms[2]["npz"][k]
        delta, changed = delta_overlay(za[0], zb[0], colors)

        def iou(t):
            tp, fp, fn = t
            return tp / max(tp + fp + fn, 1)
        badge = (f"frame {k} — 天花板 ceiling IoU：reg {iou(cell_stat['reg']):.3f} · "
                 f"geo {iou(cell_stat['geo oracle']):.3f} · hinge A′ {iou(cell_stat['hinge A′']):.3f} · "
                 f"<b style='color:#8f8'>injCeil {iou(cell_stat['injCeil']):.3f}</b>　|　"
                 f"injCeil 溢出 FP（pred=天花板 但 GT 非天花板）{cell_stat['injCeil'][1]:,} px　|　"
                 f"inj↔geo 不一致 {changed:.1f}%")

        imgs = [orig, gtcol]
        tags = ["orig", "gt"]
        names = ["原图", "语义真值"]
        for a, im in zip(arms, preds):
            imgs.append(im); tags.append(a["key"].replace(" ", "_"))
            names.append(a["label"])
        imgs.append(delta); tags.append("delta"); names.append("差异 injCeil↔geo（灰=一致）")

        cells = []
        for im, tag, nm, acc in zip(imgs, tags, names,
                                    [False, False, False, False, True, False, False]):
            rel = f"panels/{k:04d}_{tag}.png"
            cv2.imwrite(str(out / rel), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cells.append(imgcell(rel, nm, accent=acc))

        strip = np.concatenate([cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                                   value=(30, 30, 30)) for im in imgs], axis=1)
        cv2.imwrite(str(out / f"{k:04d}_seven.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        blocks.append(
            f'<div style="margin:26px auto;max-width:2100px">'
            f'<h3 style="color:#eee;margin:6px 2px;font-size:15px">{badge}</h3>'
            f'<div style="display:flex;gap:6px">{"".join(cells)}</div></div>')
        print(f"frame {k}: ceiling IoU reg {iou(cell_stat['reg']):.3f} geo "
              f"{iou(cell_stat['geo oracle']):.3f} A′ {iou(cell_stat['hinge A′']):.3f} "
              f"injCeil {iou(cell_stat['injCeil']):.3f} · injCeil FP {cell_stat['injCeil'][1]:,}", flush=True)

    (out / "index.html").write_text(
        "<!doctype html><html><body style='background:#111;color:#ddd;font-family:sans-serif'>"
        "<h2 style='margin:10px'>office0 · 几何目标灌注 Arm B′（injCeil）效果对比 —— "
        "天花板修好了，代价是墙被灌进天花板（视觉版）</h2>"
        f'<p style="color:#999;margin:2px 10px">四个预测臂是不同 ckpt：'
        'reg/geo oracle 同 ckpt（geo = 查询期把朝下天花板像素重标 ceiling、墙不动），'
        'hinge A′ / injCeil 为重训场。纯 argmax 50 帧 {0,6,…,294} 评估。'
        '天花板 IoU 0.005→0.280 是<a href="#floodnote" style="color:#79c">靠 2.6M 墙-FP 撑起来</a>的：'
        '对照 geo oracle 列可看「不灌墙的天花板修复」长什么样。</p>'
        f"<div style='max-width:2100px'>{legend}</div>"
        f"<div style='background:#15161a;padding:10px;margin:10px;border-radius:8px;max-width:900px'>{table}"
        "<p style='color:#888;font-size:12px;margin:6px 0 0' id='floodnote'>▲ injCeil 天花板色溢出的去向："
        "pred=天花板 落 GT 墙 2,596k px（A′ 仅 13k）；落 ceiling 真阳 1,519k。"
        "即 injCeil 的 0.280 = 真阳 1.5M + 假阳 ~3.5M（对比 geo oracle 0.249 = 真阳 1.2M、假阳 ~0.2M，墙面几乎零泄漏）。"
        "hinge A′ 是上一代负排斥方案，0.236。</p></div>"
        + "".join(blocks) + "</body></html>")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
