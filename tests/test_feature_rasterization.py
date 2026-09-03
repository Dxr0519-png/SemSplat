"""Verify gsplat's N-D (feature) rasterization semantics that the model relies on:
rendering per-Gaussian features with sh_degree=None is alpha-premultiplied, so
feat_render/alpha equals the expected feature where alpha > 0. Requires a GPU."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)

from gsplat.rendering import rasterization  # noqa: E402


def _render(feats: torch.Tensor, N: int = 2000):
    dev = "cuda"
    means = (torch.rand(N, 3, device=dev) - 0.5) * 2.0
    quats = torch.randn(N, 4, device=dev)
    scales = (torch.rand(N, 3, device=dev) + 0.2) * 0.03
    opac = torch.rand(N, device=dev) * 0.9 + 0.1
    view = torch.eye(4, device=dev)[None]
    K = torch.tensor([[160., 0, 80], [0, 160, 64], [0, 0, 1]], device=dev)[None]
    render, alpha, _ = rasterization(
        means, quats, scales, opac, feats, view, K, 160, 128,
        sh_degree=None, render_mode="RGB", backgrounds=None,
    )
    return render[0], alpha[0]  # [H,W,D], [H,W,1]


def test_const_feature_yields_const_expected():
    N = 2000
    D = 8
    const = torch.randn(D, device="cuda") * 0.5 + 1.0
    feats = const.repeat(N, 1)
    render, alpha = _render(feats, N=N)
    expected = render / alpha.clamp_min(1e-6)
    mask = (alpha > 0.5)[..., 0]
    if mask.sum() == 0:
        pytest.skip("no covered pixels in synthetic scene")
    # expected ~ const wherever covered
    err = (expected[mask] - const).abs().mean()
    assert err.item() < 1e-2, f"expected feature deviates: {err.item()}"


def test_zero_background_premultiplied():
    """Uncovered pixels (alpha ~ 0) must not leak feature energy."""
    N = 1
    # one tiny gaussian in the corner, far outside frustum center
    means = torch.tensor([[50.0, 50.0, 50.0]], device="cuda")
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    scales = torch.tensor([[0.01, 0.01, 0.01]], device="cuda")
    opac = torch.tensor([0.9], device="cuda")
    feats = torch.ones((1, 4), device="cuda")
    view = torch.eye(4, device="cuda")[None]
    K = torch.tensor([[160., 0, 80], [0, 160, 64], [0, 0, 1]], device="cuda")[None]
    render, alpha, _ = rasterization(
        means, quats, scales, opac, feats, view, K, 160, 128,
        sh_degree=None, render_mode="RGB", backgrounds=None,
    )
    # outside gaussian footprint premultiplied render must be ~0
    assert render[0, 10, 10].abs().max().item() < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
