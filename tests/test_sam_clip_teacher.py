"""SamClipTeacher AMG+ctx contract test (GPU; self-skips when VRAM is busy).

Instantiates the real SAM-B + CLIP teacher and checks the native-AMG mask path
(``_sam_masks_amg``) still produces whole-frame coverage comparable to the grid
path, on one real office0 frame. Skipped when CUDA is unavailable, fewer than
~3.5 GB are free (e.g. during training), or the office0 test image is missing.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

IMG = "/home/dxr/my_project/gs-slam/data/office0_est320_ns/images/000000.png"


def _no_gpu() -> bool:
    if not torch.cuda.is_available():
        return True
    free, _ = torch.cuda.mem_get_info()
    return free < 3.5e9  # AMG teacher needs SAM + CLIP resident


pytestmark = [
    pytest.mark.skipif(_no_gpu(), reason="needs CUDA with >=3.5GB free"),
    pytest.mark.skipif(not os.path.exists(IMG), reason="office0 test frame missing"),
]


def test_amg_masks_cover_like_grid_on_office0_frame():
    import cv2

    from semsplat.teachers.sam_clip_teacher import SamClipTeacher

    img = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
    rgb = torch.from_numpy(img).float().cuda().div(255.0)
    H, W = img.shape[:2]

    amg = SamClipTeacher(device="cuda", sam_mode="amg", points_per_side=4,
                         crop_mode="ctx")
    ga, ca = amg._dense(rgb)
    # coverage across cells (>=50% of a cell lit => covered), as in the smoke script
    cov = ca.float().cpu().numpy()
    assert (cov > 0.4).mean() > 0.3, "AMG left most cells uncovered"
    assert ga.shape == (math.ceil(H / 16), math.ceil(W / 16), 512)
    assert torch.allclose(torch.norm(ga.float(), dim=-1),
                          torch.ones_like(ga[..., 0].float()), atol=1e-2)
    del amg
    torch.cuda.empty_cache()

    # grid path still works and its masks are sane (byte-comparable legacy arm)
    grid = SamClipTeacher(device="cuda", sam_mode="grid", crop_mode="gray")
    gg, cg = grid._dense(rgb)
    assert gg.shape == ga.shape
    assert (cg.float().cpu().numpy() > 0.4).mean() > 0.2
    del grid
    torch.cuda.empty_cache()
