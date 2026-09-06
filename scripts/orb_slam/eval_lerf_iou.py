#!/usr/bin/env python
"""eval_lerf_iou.py — LERF/3D-OVS 式 per-object 开放词汇分割评估（阈值扫描取最优 IoU）。

对 GT 目录（label/<scene>/ 下的 COCO 式 json：每帧含 objects[{category, group, segmentation:[[x,y],...]}]）
逐 query 物体：在该 query 出现的每帧上，渲染该帧所有 query 的余弦图，对目标类别多边形 GT
扫阈值取最大 IoU；最后按 query 平均并汇总（未标注像素不参与，无主观背景词）。

Usage:
    eval_lerf_iou.py --config <run>/config.yml --ns data/waldo_kitchen_ns \
        --gt-dir data/lerf_ovs/label/waldo_kitchen \
        --scene-name waldo_kitchen [--queries "sink,cabinet"] \
        --out results/lerf/waldo_kitchen_iou.txt [--dump-overlays results/lerf/overlay_waldo]
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
from semsplat.inference import scoring
from semsplat.teachers.clip_local_teacher import encode_text_prompts, resolve_text_model_name


def rasterize_polys(obj_list, W, H):
    """Union of COCO polygon segmentations -> binary uint8 mask (H,W)."""
    mask = np.zeros((H, W), np.uint8)
    for o in obj_list:
        poly = np.asarray(o["segmentation"], np.float32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ns", type=Path, required=True, help="ns 数据目录(transforms.json, images)")
    ap.add_argument("--gt-dir", type=Path, required=True, help="label/<scene> json 目录")
    ap.add_argument("--queries", default=None, help="逗号分隔类别子集；缺省=全部")
    ap.add_argument("--out", type=Path, default=Path("results/lerf/iou.txt"))
    ap.add_argument("--dump-overlays", type=Path, default=None, help="输出 GT/预测叠加 png")
    args = ap.parse_args()

    meta = json.loads((args.ns / "transforms.json").read_text())
    frames = meta["frames"]
    # frame file stem(frame_XXXXX) -> dataset index & pose
    idx_by_stem = {Path(f["file_path"]).stem: i for i, f in enumerate(frames)}
    W, H = int(meta["w"]), int(meta["h"])

    jsons = sorted((args.gt_dir).glob("*.json"))
    per_frame_objs = []  # (dataset_k, stem, [objects])
    for jp in jsons:
        stem = jp.stem
        if stem not in idx_by_stem:
            print(f"[warn] GT frame {stem} not registered in transforms, skip")
            continue
        d = json.loads(jp.read_text())
        per_frame_objs.append((idx_by_stem[stem], stem, d.get("objects", [])))

    # queries = union of categories; if restricted, only those appearing
    all_cats = sorted({o["category"] for _, _, objs in per_frame_objs for o in objs})
    queries = [q.strip() for q in args.queries.split(",")] if args.queries else all_cats
    missing = [q for q in queries if q not in all_cats]
    if missing:
        print(f"[warn] query not in GT: {missing}")
        queries = [q for q in queries if q in all_cats]
    print(f"queries({len(queries)}): {queries}   GT frames={len(per_frame_objs)}")

    _, pipeline, _, _ = eval_setup(args.config)
    model = pipeline.model
    model.eval()
    text = resolve_text_model_name(model.config)
    emb = torch.stack([encode_text_prompts(queries, text)[q] for q in queries], dim=0).to(model.device)  # [C,M]

    frame_of_q = {q: [] for q in queries}          # k indices where q present
    for k, stem, objs in per_frame_objs:
        present = {o["category"] for o in objs}
        for q in present:
            if q in frame_of_q:
                frame_of_q[q].append(k)

    # per-query accumulation over frames: store per-frame best IoU list
    best_iou = {q: [] for q in queries}
    gt_cnt = {q: 0 for q in queries}
    if args.dump_overlays is not None:
        args.dump_overlays.mkdir(parents=True, exist_ok=True)

    for k, stem, objs in per_frame_objs:
        f = frames[k]
        M = np.asarray(f["transform_matrix"], float)[:3, :4]
        cam = Cameras(camera_to_worlds=torch.tensor(M)[None].float().to(model.device),
                      fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                      width=W, height=H).to(model.device)
        maps, alpha = scoring.semantic_maps_from_prompts(model, cam, emb)  # [H,W,C], [H,W,1]
        maps = maps.detach().cpu().numpy(); alpha = alpha.detach().cpu().numpy()
        valid = alpha[..., 0] > 0.5
        present = {o["category"] for o in objs}
        for q in queries:
            if q not in present:
                continue
            gt = rasterize_polys([o for o in objs if o["category"] == q], W, H) > 0
            gt_cnt[q] += 1
            m = maps[..., queries.index(q)]
            mm = m[valid]
            if mm.size == 0 or not gt[valid].any():
                continue
            lo, hi = float(np.min(mm)), float(np.max(mm))
            if hi - lo < 1e-9:
                best = float((gt & valid).sum()) / max(float(gt.sum()), 1)  # constant map: IoU=recall
                best_iou[q].append(best)
                continue
            th = np.linspace(lo, hi, 200)
            ious = np.zeros_like(th)
            gtv = gt[valid]
            for i, t in enumerate(th):
                p = m[valid] > t
                tp = int((p & gtv).sum())
                fn = int((gtv & ~p).sum())
                fp = int((p & ~gtv).sum())
                ious[i] = tp / max(tp + fp + fn, 1)
            best_iou[q].append(float(ious.max()))
        # overlay dump (per GT frame): GT masks blue outline? -> write combined png
        if args.dump_overlays is not None:
            img = cv2.imread(str((args.ns / "images" / Path(f["file_path"]).name)))
            for o in objs:
                c = queries.index(o["category"]) if o["category"] in queries else -1
                if c >= 0:
                    msk = rasterize_polys([o], W, H) > 0
                    img[msk] = (0, 200, 255) if c == -1 else (255, 60, 60)  # red-ish GT poly
            cv2.imwrite(str(args.dump_overlays / f"{stem}_gt.png"), img)

    # report
    lines = ["object-query  nFrames  mean_best_IoU  per_frame"]
    results = {}
    for q in queries:
        b = best_iou[q]
        if not b:
            continue
        results[q] = float(np.mean(b))
        lines.append(f"{q:24s} {gt_cnt[q]:5d}  {results[q]:.3f}        "
                     f"[{', '.join(f'{x:.2f}' for x in b)}]")
    m = float(np.mean(list(results.values()))) if results else 0.0
    lines.append("-" * 70)
    lines.append(f"{'mean over queries':24s}  {m:.3f}   (n_queries={len(results)})")
    txt = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
