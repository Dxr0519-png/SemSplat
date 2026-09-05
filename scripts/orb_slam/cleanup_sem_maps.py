#!/usr/bin/env python
"""Offline cleanup of rendered semantic maps: merge fragmented masks & drop
high-frequency small specks on walls/ceilings (no retrain).

Consumes the per-frame artifacts written by ``semantic_bg.py`` (or ``semsplat
-query``): ``<idx>_scores.npz`` (keys ``scores``/``argmax``/``top_score``/
``margin``/``valid``, ``argmax`` in 0..C-1 with 255 = background) plus the
``prompts.json`` sidecar that names the class columns. Pixels classified as
small wrong-class islands *fully enclosed inside a wall/ceiling component* are
relabelled to the enclosing host class (wall speck -> wall, ceiling speck ->
ceiling); an optional global tiny-region merge pass can additionally fuse thin
wrong-colour seams. Output goes to ``--out-dir`` (never written back into
``--in-dir``) as ``<stem>_clean_scores.npz`` (same keys, only ``argmax``
replaced) + ``<stem>_sem_clean.png`` + ``<stem>_report.json`` + a summed
``cleanup_report.json``.

Optional no-model IoU quantification (mirrors ``eval_miou_room.py``'s GT
convention): pass ``--gt-dir`` + ``--gt-ids`` (prompt idx -> Replica class id)
to print per-class IoU before vs after over classified (non-255) valid pixels.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from semsplat.seg_cleanup import merge_tiny_regions, remove_enclosed_small_islands

BG = np.array([28, 28, 28], np.uint8)  # background marker colour (matches semantic_bg.py)
UNLIT = 40  # fixed "unlit / no alpha" grey used repo-wide


def _frame_id(p: Path) -> int | None:
    """Frame id from a ``<id>_scores.npz`` file name (the digit run before ``_scores``)."""
    m = re.search(r"(\d+)_scores\.npz$", p.name)
    return int(m.group(1)) if m else None


def _resolve_classes(tokens, prompts):
    """Map each token to prompt indices: an int is taken literally, a string is a
    case-insensitive substring that must match exactly one prompt."""
    idxs = set()
    for tok in tokens:
        try:
            idxs.add(int(tok))
            continue
        except ValueError:
            pass
        hits = [i for i, p in enumerate(prompts) if tok.lower() in p.lower()]
        if len(hits) != 1:
            raise SystemExit(f"class token {tok!r} matched {len(hits)} prompts: {prompts}")
        idxs.add(hits[0])
    return sorted(idxs)


def _load_gt(gt_dir: Path, frame: int) -> np.ndarray | None:
    for name in (f"{frame}.png", f"{frame:06d}.png"):
        f = gt_dir / name
        if f.exists():
            return cv2.imread(str(f), -1)
    return None


def _colorize(arg, cols, valid):
    H, W = arg.shape
    rgb = np.zeros((H, W, 3), np.uint8)
    for c, col in enumerate(cols):
        rgb[arg == c] = np.asarray(col, np.uint8)
    rgb[arg == 255] = BG
    rgb[~valid] = UNLIT
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True, help="dir of <stem>_scores.npz + prompts.json (semantic_bg output)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <in-dir>_clean (never writes into --in-dir)")
    ap.add_argument("--wall-classes", nargs="+", default=None,
                    help="prompt index or unique substring; default auto 'wall'")
    ap.add_argument("--ceiling-classes", nargs="+", default=None,
                    help="prompt index or unique substring; default auto 'ceiling'")
    ap.add_argument("--extra-host-classes", nargs="+", default=None,
                    help="extra host classes beyond wall/ceiling (e.g. floor) — index or unique substring")
    ap.add_argument("--min-area", type=int, default=None, help="absolute island-area threshold in px")
    ap.add_argument("--min-area-frac", type=float, default=0.0008,
                    help="island-area threshold as a fraction of the frame (H*W)")
    ap.add_argument("--merge-tiny", action="store_true", help="also run a global tiny-region merge pass")
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="restrict to these stems; default all")
    ap.add_argument("--gt-dir", type=Path, default=None, help="optional Replica GT class-id maps for IoU before/after")
    ap.add_argument("--gt-ids", type=int, nargs="+", default=None, help="per-prompt Replica class ids (len == n prompts)")
    args = ap.parse_args()

    in_dir = args.in_dir.resolve()
    if not (in_dir / "prompts.json").exists():
        raise SystemExit(f"no prompts.json in {in_dir} (expected semantic_bg.py output)")
    meta = json.loads((in_dir / "prompts.json").read_text())
    prompts = meta["prompts"]
    colors = meta["colors"]
    C = len(prompts)
    if args.gt_ids is not None and len(args.gt_ids) != C:
        raise SystemExit(f"--gt-ids has {len(args.gt_ids)} entries but there are {C} prompts")

    hosts = _resolve_classes(args.wall_classes or ["wall"], prompts)
    hosts += _resolve_classes(args.ceiling_classes or ["ceiling"], prompts)
    if args.extra_host_classes:
        hosts += _resolve_classes(args.extra_host_classes, prompts)
    if not hosts:
        raise SystemExit("no host (wall/ceiling) classes resolved")

    out_dir = (args.out_dir or Path(str(in_dir) + "_clean")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*_scores.npz") if _frame_id(p) is not None],
                   key=lambda p: _frame_id(p))
    if args.frames:
        want = set(args.frames)
        files = [f for f in files if _frame_id(f) in want]

    gt_targets = args.gt_ids if args.gt_ids is not None else list(range(C))
    # before/after confusion tallies (only filled when --gt-dir is present)
    tp_b = np.zeros(C); fp_b = np.zeros(C); fn_b = np.zeros(C)
    tp_a = np.zeros(C); fp_a = np.zeros(C); fn_a = np.zeros(C)

    total = {"n_islands": 0}
    for npz_path in files:
        stem = int(_frame_id(npz_path))
        z = np.load(str(npz_path))
        scores = z["scores"]
        argmax = z["argmax"].astype(np.int64)
        valid = z["valid"].astype(bool)
        H, W = argmax.shape
        min_area = args.min_area if args.min_area is not None else max(1, int(round(args.min_area_frac * H * W)))

        cleaned = argmax.copy()
        cleaned, n_islands = remove_enclosed_small_islands(cleaned, hosts, min_area=min_area)
        if args.merge_tiny:
            cleaned = merge_tiny_regions(cleaned, min_area=min_area)

        # per-class pixel delta for the report
        before = {str(c): int((argmax == c).sum()) for c in range(C)}
        after = {str(c): int((cleaned == c).sum()) for c in range(C)}
        host_changes = {}
        for h in hosts:
            host_changes[str(h)] = int((cleaned == h).sum() - (argmax == h).sum())
        report = {"frame": stem, "min_area": min_area, "n_islands": int(n_islands),
                  "host_classes": hosts, "before": before, "after": after, "host_delta": host_changes}
        total["n_islands"] += int(n_islands)

        if args.gt_dir is not None:
            gt = _load_gt(args.gt_dir, stem)
            if gt is None:
                print(f"frame {stem}: no GT at {args.gt_dir}/{stem}.png -- skipping IoU")
            else:
                gt = np.broadcast_to(gt, (H, W)).copy()
                sel = valid & (argmax != 255) & (gt != 0) & (gt != -1)
                # Only the previously-classified pixels are comparable; cleanup can only
                # move a classified pixel between classes, never out to bg.
                arg_v = argmax[sel]
                cl_v = cleaned[sel]
                gv = gt[sel]
                for i, gid in enumerate(gt_targets):
                    p = arg_v == i
                    g = gv == gid
                    tp_b[i] += int((p & g).sum())
                    fp_b[i] += int((p & (gv != gid)).sum())
                    fn_b[i] += int((g & ~p).sum())
                    p = cl_v == i
                    tp_a[i] += int((p & g).sum())
                    fp_a[i] += int((p & (gv != gid)).sum())
                    fn_a[i] += int((g & ~p).sum())

        # write cleaned artifacts
        np.savez_compressed(
            out_dir / f"{stem:04d}_clean_scores.npz",
            scores=scores, argmax=cleaned, top_score=z["top_score"], margin=z["margin"], valid=valid,
        )
        rgb = _colorize(cleaned, colors, valid)
        cv2.imwrite(str(out_dir / f"{stem:04d}_sem_clean.png"), rgb[:, :, ::-1])
        (out_dir / f"{stem:04d}_clean_report.json").write_text(json.dumps(report, indent=1))
        print(f"frame {stem}: islands removed {n_islands}  hosts {hosts}  "
              f"host delta { {prompts[h]: host_changes[str(h)] for h in hosts} }")

    if args.gt_dir is not None:
        print("\nIoU (before -> after)  [only pixels that stayed classified]")
        for i in range(C):
            iou_b = tp_b[i] / max(tp_b[i] + fp_b[i] + fn_b[i], 1)
            iou_a = tp_a[i] / max(tp_a[i] + fp_a[i] + fn_a[i], 1)
            print(f"{prompts[i]:24s} {iou_b:.3f} -> {iou_a:.3f}")
    (out_dir / "cleanup_report.json").write_text(json.dumps({"prompts": prompts, "colors": colors,
                                                             "wall_classes": hosts, **total}, indent=1))


if __name__ == "__main__":
    main()
