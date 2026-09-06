#!/usr/bin/env python
"""eval_lerf_langsplat.py — 与 LangSplat 论文评估协议逐字同构的 per-object IoU(单一尺度版).

把 LangSplat `eval/evaluate_iou_loc.py` 的 activate_stream/lerf 度量流程搬到我们的模型上,
唯一自由度: 我们的模型是单尺度(semantic 特征渲染一档), 而 LangSplat 是多尺度金字塔选层,
故"选层"这一环固定为 level 0(其余步骤含 softmax 背景/平滑/截断/阈值全部照搬)。
文本编码用**我们模型的 OpenAI-CLIP**(学生特征所在空间); LangSplat 原版用 OpenCLIP-laion,
因此本数 = "我们的方法在 LangSplat 尺子上"的分数, 与论文 44.5(waldo) 仅文本空间不同源。

协议复刻自其代码 (2026-09-06 抓取 minghanqin/LangSplat main 分支):
  openclip_encoder.get_relevancy: positives[该帧 GT category 原词] vs 固定 negatives
      ("object","things","stuff","texture"), softmax(10*sim) 取"最弱负类"那一对的 positive 概率
  1) act_j (H,W) in [0,1]   (无效几何像素置 0)
  2) 30x30 平均核 cv2.filter2D, 再 act = 0.5*(avg+act)
  3) 归一化: out=(act-min)/(max+1e-9); out=out*2-1; clip(0,1)
  4) mask = out > mask_thresh(=0.4), 再经 utils.smooth(7x7 众数滤波, 逐字照搬)
  5) 与该帧该 category 的 GT union 掩码算 IoU
最终均值 = 所有 (帧, 该帧 present 物体) 的 IoU 平均(同他们 mean_iou_chosen)。

Usage:
    eval_lerf_langsplat.py --config <run>/config.yml --ns data/waldo_kitchen_ns \
        --gt-dir data/lerf_ovs/label/waldo_kitchen \
        --out results/lerf/waldo_langsplat_proto.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))        # sibling imports
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup
from semsplat.inference import scoring
from semsplat.teachers.clip_local_teacher import encode_text_prompts, resolve_text_model_name

DEFAULT_NEG = ["object", "things", "stuff", "texture"]


# ---------- 逐字复刻 LangSplat eval/utils.py ----------
def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def stack_mask(mask_base, mask_add):
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask


def smooth(mask):
    h, w = mask.shape[:2]
    im_smooth = mask.copy()
    scale = 3
    for i in range(h):
        for j in range(w):
            square = mask[max(0, i - scale): min(i + scale + 1, h - 1),
                          max(0, j - scale): min(j + scale + 1, w - 1)]
            im_smooth[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return im_smooth


def smooth_fast(mask):
    """smooth 的矢量化近似: 7x7 众数 = box 均值>=0.5 (边界 BORDER_REPLICATE≈clamp)。
    仅用于阈值扫描; 固定 0.4 档仍用逐字 smooth。"""
    a = cv2.boxFilter(mask.astype(np.float32), -1, (7, 7), borderType=cv2.BORDER_REPLICATE)
    return (a >= 0.5).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ns", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/lerf/waldo_langsplat_proto.txt"))
    ap.add_argument("--mask-thresh", type=float, default=0.4, help="LangSplat mask_thresh")
    ap.add_argument("--dump-masks", type=Path, default=None, help="每帧 GT/pred 掩码 png 目录(可选)")
    args = ap.parse_args()

    meta = json.loads((args.ns / "transforms.json").read_text())
    frames = meta["frames"]
    idx_by_stem = {Path(f["file_path"]).stem: i for i, f in enumerate(frames)}
    W, H = int(meta["w"]), int(meta["h"])
    cam_xy = dict(fx=meta["fl_x"], fy=meta.get("fl_y", meta["fl_x"]),
                  cx=meta["cx"], cy=meta["cy"])

    # 每帧 category -> GT union 掩码
    entries = []
    for jp in sorted(args.gt_dir.glob("*.json")):
        stem = jp.stem
        if stem not in idx_by_stem:
            continue
        d = json.loads(jp.read_text())
        objs = d.get("objects", [])
        per_cat = {}
        for o in objs:
            m = polygon_to_mask((H, W), o["segmentation"])
            per_cat[o["category"]] = stack_mask(per_cat[o["category"]], m) if o["category"] in per_cat else m
        entries.append((idx_by_stem[stem], stem, per_cat))
    print(f"GT frames={len(entries)}")

    _, pipeline, _, _ = eval_setup(args.config)
    model = pipeline.model
    model.eval()
    text = resolve_text_model_name(model.config)
    neg = torch.stack([encode_text_prompts(DEFAULT_NEG, text)[q] for q in DEFAULT_NEG], dim=0).to(model.device)
    neg = torch.nn.functional.normalize(neg, dim=-1)

    Fg = scoring.per_gaussian_semantics(model)           # [N,512] 归一化
    if args.dump_masks is not None:
        args.dump_masks.mkdir(parents=True, exist_ok=True)

    all_iou = []                                         # [(stem, cat, iou)]
    HW = H * W
    for k, stem, per_cat in entries:
        f = frames[k]
        M = np.asarray(f["transform_matrix"], float)[:3, :4]
        cam = Cameras(camera_to_worlds=torch.tensor(M)[None].float().to(model.device),
                      width=W, height=H, **cam_xy).to(model.device)
        # 图像域像素特征: 512 维按通道分块栅格化(线性 alpha 混合), 流式汇总到 CPU 避免 OOM
        Fcpu = np.zeros((HW, Fg.shape[1]), np.float32)
        valid = None
        for b in range(0, Fg.shape[1], 64):
            maps, alpha = scoring.rasterize_class_scores(model, cam, Fg[:, b:b + 64])  # [H,W,B]
            if valid is None:
                valid = (alpha[..., 0] > 0.5).cpu().numpy()
            Fcpu[:, b:b + 64] = maps.reshape(HW, -1).cpu().numpy()
        Fcpu[~valid.reshape(-1)] = 0
        # 归一化有效像素行(无效行保持 0)
        norms = np.linalg.norm(Fcpu, axis=1, keepdims=True)
        nz = norms[:, 0] > 1e-8
        Fcpu[nz] /= norms[nz]

        cats = sorted(per_cat.keys())
        pos = torch.stack([encode_text_prompts(cats, text)[q] for q in cats], dim=0).to(model.device)
        pos = torch.nn.functional.normalize(pos, dim=-1)
        P = len(cats)
        act = np.zeros((HW, P), np.float32)
        posN = pos.cpu().numpy(); negN = neg.cpu().numpy()
        tile = 65536
        for s in range(0, HW, tile):
            Fb = Fcpu[s:s + tile]
            S = Fb @ np.concatenate([posN, negN], axis=0).T          # [t, P+N] (cos)
            posS = S[:, :P]
            negS = S[:, P:]
            pair = np.stack([np.broadcast_to(posS[..., None], (*posS.shape, negS.shape[1])),
                             np.broadcast_to(negS[:, None, :], (posS.shape[0], P, negS.shape[1]))],
                            axis=-1).astype(np.float64)               # [t,P,N,2]
            sm = np.exp(10 * pair)
            sm = sm / sm.sum(axis=-1, keepdims=True)
            act[s:s + tile] = sm[..., 0].min(axis=-1).astype(np.float32)
        act = act.reshape(H, W, P)                                    # [H,W,P]

        notes = []
        kernel = np.ones((30, 30), np.float32) / 900.0
        for j, q in enumerate(cats):
            rel = act[..., j].astype(np.float32)
            rel[~valid] = 0.0
            avg = cv2.filter2D(rel, -1, kernel)
            rel = 0.5 * (avg + rel)
            out = rel - rel.min()
            out = out / (rel.max() + 1e-9)
            out = out * 2.0 - 1.0
            out = np.clip(out, 0.0, 1.0)
            mask_gt = per_cat[q].astype(bool)
            # 固定 0.4(论文操作点)用逐字 smooth
            mp = smooth((out > args.mask_thresh).astype(np.uint8))
            inter = int(np.logical_and(mask_gt, mp).sum())
            union = int(np.logical_or(mask_gt, mp).sum())
            iou_fixed = inter / max(union, 1)
            # 阈值自由上界(同一 renormalize 空间, fast smooth): 扫阈值取最大 IoU
            best, bthr = 0.0, 0.0
            for th in np.linspace(0.0, 1.0, 201):
                mf = smooth_fast((out > th).astype(np.uint8))
                inter = int(np.logical_and(mask_gt, mf).sum())
                union = int(np.logical_or(mask_gt, mf).sum())
                i = inter / max(union, 1)
                if i > best:
                    best, bthr = i, float(th)
            all_iou.append((stem, q, iou_fixed, best, bthr))
            notes.append(f"{q}: @0.4={iou_fixed:.3f} best={best:.3f}@{bthr:.2f}")
            if args.dump_masks is not None:
                vis = np.zeros((H, W, 3), np.uint8)
                vis[mask_gt] = (0, 200, 255)            # GT
                vis[mp > 0] = (255, 60, 60)             # pred@0.4
                cv2.imwrite(str(args.dump_masks / f"{stem}_{q}.png"), vis)
        print(f"{stem}: " + "; ".join(notes), flush=True)

    backend = "OpenCLIP-laion(同协议文本)" if str(text).startswith("openclip:") else "our OpenAI-CLIP"
    lines = [f"LangSplat-protocol per-object IoU — text={backend}, single-scale (relevancy 几何照搬)",
             f"  iou@0.4 = 论文固定操作点(按 OpenCLIP-laion 激活分布校准); best = 同几何下阈值自由上界",
             f"{'category':22s} n    iou@0.4   best@thr"]
    per_cat = {}
    for stem, q, iou_fixed, best, bthr in all_iou:
        per_cat.setdefault(q, []).append((iou_fixed, best, bthr))
    for q in sorted(per_cat):
        a = np.mean([x[0] for x in per_cat[q]])
        b = np.mean([x[1] for x in per_cat[q]])
        lines.append(f"{q:22s} {len(per_cat[q]):3d}  {a:.4f}    {b:.4f}")
    mf = float(np.mean([s[2] for s in all_iou])); mb = float(np.mean([s[3] for s in all_iou]))
    lines.append("-" * 60)
    lines.append(f"mean over (frame, present-object):  iou@0.4 = {mf:.4f}    best = {mb:.4f}   (n={len(all_iou)})")
    txt = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
