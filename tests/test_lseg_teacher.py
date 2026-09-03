"""LSegDenseTeacher contract tests (GPU; skip unless ckpt + CUDA present)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

CKPT = Path("data/checkpoints/lseg/demo_e200.ckpt")
_HAS_CKPT = CKPT.exists()

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _HAS_CKPT, reason=f"missing {CKPT}"),
]


def test_dim_and_compute_shape():
    from semsplat.teachers.lseg_local_teacher import LSegDenseTeacher

    t = LSegDenseTeacher(ckpt_path=str(CKPT), input_long_side=384, cell=8, device="cuda", fp16=True)
    assert t.dim == 512
    img = torch.rand(300, 200, 3).cuda()
    grid = t.compute(img, key=7)  # cached, fp16
    assert grid.dtype == torch.float16
    gH = math.ceil(int(round(300 * 384 / 300)) / 8)  # hh = 300*(384/300)=384 -> /8
    assert grid.shape[-1] == 512
    row_l2 = torch.norm(grid.float(), dim=-1)
    assert (row_l2 - 1).abs().max().item() < 1e-2


def test_lru_cache_hit():
    from semsplat.teachers.lseg_local_teacher import LSegDenseTeacher

    t = LSegDenseTeacher(ckpt_path=str(CKPT), input_long_side=256, cell=16, device="cuda")
    img = torch.rand(128, 160, 3).cuda()
    a = t.compute(img, key=1)
    b = t.compute(img, key=1)  # cache hit
    assert a is b
    t.clear_cache()
    assert len(t._cache) == 0
