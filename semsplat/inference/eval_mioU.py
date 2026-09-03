"""[Reserved] Per-class mIoU vs Replica ground-truth semantics (M4, optional).

Interface only. To enable, feed Replica GT semantic label maps (rendered from the
scene's ``mesh_semantic.ply`` + ``info_semantic.json`` at the eval camera poses)
together with the argmax maps saved by ``semsplat-query`` (``*_scores.npz``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_miou(pred_argmax: np.ndarray, gt_ids: np.ndarray, n_classes: int) -> dict:
    """Mean IoU over classes present in gt_ids (pred in [0, n_classes))."""
    ious = {}
    for c in range(n_classes):
        inter = np.logical_and(pred_argmax == c, gt_ids == c).sum()
        union = np.logical_or(pred_argmax == c, gt_ids == c).sum()
        if union > 0:
            ious[c] = float(inter) / float(union)
    miou = float(np.mean(list(ious.values()))) if ious else 0.0
    return {"per_class": ious, "miou": miou}


def evaluate_replica(pred_dir: Path, gt_dir: Path) -> None:
    """Stub: not implemented in this milestone. See module docstring."""
    raise NotImplementedError(
        "Replica mIoU evaluation is an optional later milestone. "
        "Provide Replica semantic GT images aligned to the eval cameras."
    )


if __name__ == "__main__":
    raise NotImplementedError("run via semsplat-query and the M4 eval routine")
