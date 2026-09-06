#!/usr/bin/env python
"""gallery_lerf_grid.py — LERF per-object 可视化: 每帧一行 原图/重建RGB/语义真值/语义预测.

语义真值 = GT json 的 COCO 多边形按类别上色(未标注区域灰);
语义预测 = 两种模式:
  默认 (--argmax-pred 未开)  对每个 GT query 物体按"该帧最优 IoU 阈值"(同 eval_lerf_iou)
                            点亮其预测区域, 其余深灰 —— 只覆盖 GT 物体, 墙面等无 query 即无颜色.
  --argmax-pred             对 全部 GT 物体 + 附加场景 query(--extra-queries, 如 wall/floor)
                            做全图 argmax 上色(每有效像素取余弦最高的类别, 低于 floor 的置灰),
                            墙面/地面等也能被涂上颜色, 用于"模型到底认不认墙"的定性观察.

每帧并存: <out>/panels/<stem>_{orig,recon,gt,pred}.png 和整行 <out>/<stem>_row.png,
再写 <out>/index.html 供浏览器纵览(图下标题, 颜色图例固定).

Usage:
    # 逐物体最优阈值(默认, 与评测同口径)
    gallery_lerf_grid.py --config <run>/config.yml --ns data/waldo_kitchen_ns \
        --gt-dir data/lerf_ovs/label/waldo_kitchen --out results/lerf/grid_waldo
    # 全场景 argmax(墙/地板也被涂色)
    gallery_lerf_grid.py --config <run>/config.yml --ns data/waldo_kitchen_ns \
        --gt-dir data/lerf_ovs/label/waldo_kitchen --out results/lerf/grid_waldo_scene \
        --extra-queries "wall,floor,countertop,ceiling,stove,window" --argmax-pred
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))        # for eval_lerf_iou
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup
from semsplat.inference import scoring
from semsplat.teachers.clip_local_teacher import encode_text_prompts, resolve_text_model_name
from eval_lerf_iou import rasterize_polys

GRAY = np.array([32, 32, 34], np.uint8)   # 未标注/未点亮底
DARK = "#0b0b0f"
FLOOR = 0.12                                # argmax 模式下最低余弦, 低于则置灰


def palette(n: int):
    cols = []
    for i in range(n):
        h = (i * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.6, 0.95)
        cols.append((np.array([r, g, b]) * 255).round().astype(np.uint8))
    return cols


def best_thr_iou(m: np.ndarray, valid: np.ndarray, gt: np.ndarray):
    """m=该帧该 query 余弦图; 返回 (best_threshold, best_IoU)。同 eval_lerf_iou 口径。"""
    mm = m[valid]
    if mm.size == 0 or not gt[valid].any():
        return None, None
    lo, hi = float(np.min(mm)), float(np.max(mm))
    if hi - lo < 1e-9:
        return None, float((gt & valid).sum()) / max(float(gt.sum()), 1)
    th = np.linspace(lo, hi, 200)
    best_t, best_i = th[0], -1.0
    gtv = gt[valid]
    for t in th:
        p = m[valid] > t
        tp = int((p & gtv).sum()); fn = int((gtv & ~p).sum()); fp = int((p & ~gtv).sum())
        i = tp / max(tp + fp + fn, 1)
        if i > best_i:
            best_i, best_t = i, t
    return best_t, float(best_i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ns", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/lerf/grid_waldo"))
    ap.add_argument("--frames", nargs="*", default=None, help="缺省=全部 GT 帧")
    ap.add_argument("--extra-queries", default=None, help="逗号分隔场景 query(仅可视化, 无 GT)")
    ap.add_argument("--argmax-pred", action="store_true", help="预测用全图 argmax(含 extra), 非逐物体阈值")
    args = ap.parse_args()
    extras = [q.strip() for q in args.extra_queries.split(",") if q.strip()] if args.extra_queries else []

    meta = json.loads((args.ns / "transforms.json").read_text())
    frames = meta["frames"]
    idx_by_stem = {Path(f["file_path"]).stem: i for i, f in enumerate(frames)}
    W, H = int(meta["w"]), int(meta["h"])

    entries = []
    for jp in sorted(args.gt_dir.glob("*.json")):
        stem = jp.stem
        if stem not in idx_by_stem:
            continue
        d = json.loads(jp.read_text())
        entries.append((idx_by_stem[stem], stem, d.get("objects", [])))
    if args.frames:
        keep = set(args.frames)
        entries = [e for e in entries if e[1] in keep]

    all_cats = sorted({o["category"] for _, _, objs in entries for o in objs})
    legend_cats = all_cats + [e for e in extras if e not in all_cats]
    colors = dict(zip(legend_cats, palette(len(legend_cats))))
    print(f"frames={len(entries)} cats={len(all_cats)} extras={extras} argmax={args.argmax_pred}")

    _, pipeline, _, _ = eval_setup(args.config)
    model = pipeline.model
    model.eval()
    text = resolve_text_model_name(model.config)

    out = args.out
    panels = out / "panels"; panels.mkdir(parents=True, exist_ok=True)
    legend = "<div style='margin:8px'>" + " &nbsp; ".join(
        f'<span style="background:rgb{tuple(c)};color:#000;padding:1px 7px;border-radius:4px">{q}</span>'
        for q, c in colors.items()) + "</div>"

    def imgcell(rel, caption, note=""):
        n = f'<div style="color:#99a;font-size:12px">{note}</div>' if note else ""
        return (f'<div style="flex:1;min-width:0">'
                f'<img src="{rel}" style="width:100%;display:block;border:1px solid #333"/>'
                f'<div style="text-align:center;padding:4px;background:{DARK};font-weight:600;color:#fff">{caption}</div>{n}</div>')

    blocks = []
    for k, stem, objs in entries:
        f = frames[k]
        M = np.asarray(f["transform_matrix"], float)[:3, :4]
        cam = Cameras(camera_to_worlds=torch.tensor(M)[None].float().to(model.device),
                      fx=meta["fl_x"], fy=meta.get("fl_y", meta["fl_x"]),
                      cx=meta["cx"], cy=meta["cy"], width=W, height=H).to(model.device)

        orig = cv2.imread(str(args.ns / "images" / Path(f["file_path"]).name))[:, :, ::-1]
        recon = model.get_outputs_for_camera(cam)["rgb"].clamp(0, 1)
        recon = (recon.detach().cpu().numpy() * 255).round().astype(np.uint8)

        present = sorted({o["category"] for o in objs})
        # query 顺序: present 在前, extras 补后(去重)
        union = present + [e for e in extras if e not in present]
        emb = torch.stack([encode_text_prompts(union, text)[q] for q in union], dim=0).to(model.device)
        maps, alpha = scoring.semantic_maps_from_prompts(model, cam, emb)
        maps = maps.detach().cpu().numpy(); alpha = alpha.detach().cpu().numpy()
        valid = alpha[..., 0] > 0.5

        # GT 面板
        gtc = np.zeros((H, W, 3), np.uint8); gtc[:] = GRAY
        for o in objs:
            q = o["category"]
            gtc[rasterize_polys([o], W, H) > 0] = colors[q]

        notes, rows = [], []
        if not args.argmax_pred:
            # 逐物体最优阈值上色 + IoU 注记(同评测口径)
            pred_regions = {}
            for q in present:
                gtm = rasterize_polys([o for o in objs if o["category"] == q], W, H) > 0
                t, iou = best_thr_iou(maps[..., union.index(q)], valid, gtm)
                if t is not None:
                    pred_regions[q] = (maps[..., union.index(q)] > t) & valid
                    notes.append(f"{q}: IoU={iou:.2f} t={t:.2f}")
                else:
                    notes.append(f"{q}: IoU={iou:.2f} (uniform/empty)")
            def iou_of(q):
                _, i = best_thr_iou(maps[..., union.index(q)], valid,
                                    rasterize_polys([o for o in objs if o["category"] == q], W, H) > 0)
                return i if i is not None else -1.0
            order = sorted(pred_regions, key=iou_of, reverse=True)
            pdc = np.zeros((H, W, 3), np.uint8); pdc[:] = GRAY
            for q in order:
                pdc[pred_regions[q]] = colors[q]
        else:
            # 全图 argmax: 每像素取 union 里余弦最高的类别; 最高仍低于 FLOOR 则置灰
            best = maps.argmax(axis=-1)                       # [H,W]
            mx = maps.max(axis=-1)                            # [H,W]
            pdc = np.zeros((H, W, 3), np.uint8); pdc[:] = GRAY
            show = valid & (mx > FLOOR)
            for qi, q in enumerate(union):
                mask = show & (best == qi)
                if mask.any():
                    pdc[mask] = colors[q]
                    notes.append(f"{q}: {int(mask.sum())}px")
            pct_show = float(show.mean()) * 100
            notes.insert(0, f"argmax 点亮 {pct_show:.0f}% 有效像素 (floor={FLOOR})")

        imgs = [orig, recon, gtc, pdc]
        tags = ["orig", "recon", "gt", "pred"]
        names = ["原图", "重建 RGB", "语义真值 (GT)", "语义预测" + (" (全场景 argmax)" if args.argmax_pred else " (最优阈值)")]
        cells = []
        for im, tag, nm in zip(imgs, tags, names):
            rel = f"panels/{stem}_{tag}.png"
            cv2.imwrite(str(out / rel), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cells.append(imgcell(rel, nm))
        row = np.concatenate([cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                                 value=(30, 30, 30)) for im in imgs], axis=1)
        cv2.imwrite(str(out / f"{stem}_row.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        blocks.append(
            f'<div style="margin:26px auto;max-width:2000px">'
            f'<h3 style="color:#eee;margin:6px 2px">{stem} &nbsp;'
            f'<span style="color:#99a;font-size:13px">{" | ".join(notes)}</span></h3>'
            f'<div style="display:flex;gap:6px">{"".join(cells)}</div></div>')
        print(f"{stem}: " + "; ".join(notes), flush=True)

    # 标题从 run 的 config 推断(教师骨干/迭代/分辨率), 免误导
    cfg = model.config
    be = "OpenCLIP-laion" if getattr(cfg, "teacher_clip_backend", "openai") == "openclip" else "OpenAI-CLIP"
    res = "全分辨率" if getattr(cfg, "num_downscales", 1) == 0 else "半分辨率"
    tier = f"{getattr(cfg, 'max_num_iterations', 0)}it · {res} · {be}"
    title = (f"waldo_kitchen · {tier} · 全场景 argmax(含 wall/floor 等附加 query)"
             if args.argmax_pred else
             f"waldo_kitchen · {tier} · 原图 / 重建 / 语义真值 / 语义预测(最优阈值)")
    (out / "index.html").write_text(
        "<!doctype html><html><body style='background:#111;color:#ddd;font-family:sans-serif'>"
        f"<h2>{title}</h2>{legend}"
        + "".join(blocks) + "</body></html>")
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
