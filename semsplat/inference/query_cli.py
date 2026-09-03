"""``semsplat-query``: open-vocabulary text -> semantic maps / colored PLY.

Usage:
    semsplat-query --config outputs/<run>/config.yml \
        --prompts "sofa,chair,coffee table,rug,tv monitor,door,wall,window" \
        --camera-ids 0 50 100 --out-dir results/<scene>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# torch>=2.6 defaults to weights_only=True, which nerfstudio checkpoints (with
# numpy scalars) do not satisfy. We trust our own locally-trained checkpoints.
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

from nerfstudio.utils.eval_utils import eval_setup

from semsplat.inference import scoring
from semsplat.inference.ply_writer import write_semantic_ply
from semsplat.teachers.clip_local_teacher import encode_text_prompts


def _palette(n: int) -> np.ndarray:
    import colorsys

    cols = []
    for i in range(n):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        cols.append([int(255 * r), int(255 * g), int(255 * b)])
    return np.asarray(cols, dtype=np.uint8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Open-vocabulary semantic querying of a semsplat scene")
    ap.add_argument("--config", required=True, type=Path, help="path to the saved ns-train config.yml")
    ap.add_argument("--prompts", required=True, help="comma-separated text prompts (e.g. 'chair,sofa,table')")
    ap.add_argument("--camera-ids", nargs="*", type=int, default=None, help="eval camera indices; default all")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--export-ply", action="store_true", help="also write a per-Gaussian colored PLY")
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise SystemExit("no prompts given")

    print(">> loading pipeline/checkpoint from", args.config)
    _, pipeline, _, _ = eval_setup(args.config)
    model = pipeline.model
    model.eval()
    if not (hasattr(model, "gauss_params") and "sem_features" in model.gauss_params):
        raise SystemExit("model has no semantic features; train with `ns-train semsplat` first")
    device = model.device

    print(">> encoding text prompts:", prompts)
    emb_dict = encode_text_prompts(prompts, model.config.teacher_model_name, device=str(device))
    emb = torch.stack([emb_dict[p] for p in prompts], dim=0).to(device)  # [C,M]

    # per-Gaussian best class (for the PLY export) -----------------------------
    sem = scoring.per_gaussian_semantics(model)  # [N,M]
    gauss_scores = sem @ emb.T  # [N,C]
    best_label = gauss_scores.argmax(dim=-1).cpu().numpy()
    conf = gauss_scores.max(dim=-1).values.cpu().numpy()
    palette = _palette(len(prompts))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.export_ply:
        means = model.means.detach().cpu().numpy()
        ply_path = out / "scene_semantic.ply"
        write_semantic_ply(str(ply_path), means, best_label, conf, palette)
        print(">> wrote", ply_path)

    # rasterize score maps on requested cameras ---------------------------------
    try:
        cameras = pipeline.datamanager.eval_dataset.cameras
    except AttributeError:
        cameras = pipeline.datamanager.train_dataset.cameras
    ids = args.camera_ids if args.camera_ids is not None else list(range(len(cameras)))

    meta = {"prompts": prompts, "camera_ids": ids, "colors": palette.tolist()}
    for cid in ids:
        camera = cameras[cid : cid + 1].to(device)
        maps, alpha = scoring.semantic_maps_from_prompts(model, camera, emb)  # [H,W,C]
        maps = maps.detach().cpu().numpy()
        alpha = alpha.detach().cpu().numpy()
        valid = alpha[..., 0] > 0.5
        H, W = maps.shape[:2]

        argmax = np.full((H, W), len(prompts), dtype=np.uint16)  # "unclassified" = len(prompts)
        argmax[valid] = maps.argmax(axis=-1)[valid]
        seg_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(len(prompts)):
            seg_rgb[argmax == c] = palette[c]
        seg_rgb[~valid] = 40

        stem = out / f"cam_{cid:04d}"
        _save_png(seg_rgb, stem.with_name(stem.name + "_semantic.png"))
        np.savez_compressed(
            stem.with_name(stem.name + "_scores.npz"), argmax=argmax, scores=maps, prompts=prompts
        )
        # per-prompt heatmap (up to 8): class-c score minus best other class
        for c in range(min(len(prompts), 8)):
            if len(prompts) > 1:
                idx = np.arange(len(prompts)) != c
                heat = maps[..., c] - maps[..., idx].max(axis=-1)
            else:
                heat = maps[..., c]
            heat = np.clip(heat, 0, None)
            heat = heat / (heat.max() + 1e-6)
            hm = (np.clip(heat[..., None], 0, 1) * palette[c][None, None, :]).astype(np.uint8)
            hm[~valid] = 40
            _save_png(hm, stem.with_name(stem.name + f"_heat_{c:02d}.png"))

    with open(out / "query_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(">> results in", out)


if __name__ == "__main__":
    main()
