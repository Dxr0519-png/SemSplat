"""Pure-math guard: avg_pool(linear(x)) == linear(avg_pool(x)).

The semantic loss now area-pools the K-dim feature map *before* the shared
``feature_head`` linear decoder (to avoid materialising [H,W,512]). A linear
layer commutes with average pooling (bias included), so this must hold.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_avgpool_linear_commute():
    torch.manual_seed(0)
    K, M, H, W, gH, gW = 32, 512, 64, 80, 7, 9
    x = torch.randn(H, W, K)
    lin = nn.Linear(K, M)

    head_first = lin(x)  # [H,W,M]
    a = F.adaptive_avg_pool2d(head_first.permute(2, 0, 1)[None], (gH, gW))[0].permute(1, 2, 0)

    pool_first = F.adaptive_avg_pool2d(x.permute(2, 0, 1)[None], (gH, gW))[0].permute(1, 2, 0)
    b = lin(pool_first)

    assert torch.allclose(a, b, atol=1e-5), "avg-pool and linear must commute"
