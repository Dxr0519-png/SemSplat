"""Write a colorized per-point semantic PLY (means + best class id + confidence)."""
from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement


def write_semantic_ply(
    path: str,
    means: np.ndarray,          # [N, 3]
    best_labels: np.ndarray,    # [N] int
    confidences: np.ndarray,    # [N] float in [0,1]
    palette: np.ndarray,        # [num_classes, 3] uint8
) -> None:
    means = np.asarray(means, dtype=np.float32)
    colors = palette[np.asarray(best_labels, dtype=np.int64).clip(0, len(palette) - 1)]
    verts = np.zeros(means.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                            ("semantic_id", "u2"), ("confidence", "f4")])
    verts["x"], verts["y"], verts["z"] = means[:, 0], means[:, 1], means[:, 2]
    verts["red"], verts["green"], verts["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    verts["semantic_id"] = np.asarray(best_labels, dtype=np.uint16)
    verts["confidence"] = np.asarray(confidences, dtype=np.float32)
    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False).write(path)
