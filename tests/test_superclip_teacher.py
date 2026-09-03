"""SuperClipTeacher contract tests (GPU; skip unless CUDA + skimage present)."""
from __future__ import annotations

import math

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(pytest.importorskip("skimage") is None, reason="needs skimage"),
]


def test_dim_shape_and_cache():
    from semsplat.teachers.slic_clip_teacher import SuperClipTeacher

    t = SuperClipTeacher(n_segments=40, cell=16, device="cuda", fp16=True)
    assert t.dim == 512
    img = torch.rand(200, 160, 3).cuda()
    grid = t.compute(img, key=3)
    assert grid.dtype == torch.float16
    gH, gW = math.ceil(200 / 16), math.ceil(160 / 16)
    assert grid.shape == (gH, gW, 512)
    ref = torch.ones_like(grid[..., 0].float())
    assert torch.allclose(torch.norm(grid.float(), dim=-1), ref, atol=1e-2)
    assert t.compute(img, key=3) is grid  # cache hit
    t.clear_cache()
    assert len(t._cache) == 0
