"""Unit tests for the spatial regularizers in semsplat.losses (opacity entropy +
depth-aware TV), plus the default-OFF config guards. Loss math is pure CPU torch;
the RGB+ED parity check needs a GPU."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from semsplat.losses import depth_aware_tv_loss, opacity_entropy_loss  # noqa: E402

LN2 = 0.6931471805599453


def _adam_minimize(param: torch.Tensor, loss_fn, steps: int = 400, lr: float = 0.5):
    opt = torch.optim.Adam([param], lr=lr)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(param)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    return first, loss_fn(param).item()


# ------------------------------------------------------------------ opacity entropy
def test_entropy_zero_at_saturated_logits():
    z = torch.tensor([[-20.0], [20.0]], requires_grad=True)
    h = opacity_entropy_loss(z)
    assert h.item() < 1e-3


def test_entropy_ln2_at_logit_zero():
    z = torch.zeros(4, 1, requires_grad=True)
    h = opacity_entropy_loss(z)
    assert h.item() == pytest.approx(LN2, abs=1e-5)


def test_entropy_gradient_nonzero_in_logit_space():
    # dH/dz = a(1-a) ln((1-a)/a) must be nonzero away from z=0 and finite.
    z = torch.tensor([[1.0]], requires_grad=True)
    opacity_entropy_loss(z).backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().item() > 1e-3


def test_entropy_minimized_by_adam_toward_poles():
    # noise seeds the a=0.5 saddle (a local max of H, so Adam escapes it)
    z = torch.nn.Parameter(torch.randn(64, 1) * 0.1)
    first, last = _adam_minimize(z, opacity_entropy_loss, steps=500)
    assert last < first / 2.0
    a = torch.sigmoid(z).detach()
    # population should be pushed to both poles (bimodal), not collapsed to one side
    assert (a < 0.1).float().mean().item() > 0.2
    assert (a > 0.9).float().mean().item() > 0.2


# ------------------------------------------------------------------ depth-aware TV
def _flat_scene(feat_const=1.0):
    """Covered 8x8 constant-feature scene (uniform depth, alpha 1)."""
    H = W = 8
    K = 4
    f = torch.full((H, W, K), feat_const)
    alpha = torch.ones(H, W, 1)
    depth = torch.full((H, W, 1), 2.0)
    return f, alpha, depth


def test_tv_zero_on_constant_features():
    f, a, d = _flat_scene()
    assert depth_aware_tv_loss(f, a, d, sigma=0.1).item() == pytest.approx(0.0, abs=1e-6)


def test_tv_positive_on_internal_speckle():
    f, a, d = _flat_scene()
    f[4, 4, 0] = -1.0  # one salt-and-pepper pixel inside a continuous surface
    assert depth_aware_tv_loss(f, a, d, sigma=0.1).item() > 1e-3


def test_tv_zero_across_depth_step():
    """A feature boundary aligned with a hard depth step must NOT be smoothed."""
    K = 4
    W = 6
    f = torch.zeros(2, W, K)
    f[:, :3] = 1.0  # left half = A
    f[:, 3:] = 0.0  # right half = B != A (boundary sits on the depth step)
    alpha = torch.ones(2, W, 1)
    depth = torch.ones(2, W, 1)
    depth[:, 3:] = 5.0  # step between col 2 and 3, delta = 4 m >> sigma
    tv = depth_aware_tv_loss(f, alpha, depth, sigma=0.1)
    assert tv.item() < 1e-3

    # same features but no depth step -> the boundary now lies on a continuous
    # surface and must be penalized (proves the gate, not the feature layout, fired)
    depth_flat = torch.ones(2, W, 1)
    tv_flat = depth_aware_tv_loss(f, alpha, depth_flat, sigma=0.1)
    assert tv_flat.item() > 1e-3


def test_tv_zero_when_all_uncovered():
    f, a, d = _flat_scene()
    f[4, 4, 0] = -1.0
    a = torch.zeros_like(a)
    tv = depth_aware_tv_loss(f, a, d, sigma=0.1)
    assert tv.item() == pytest.approx(0.0, abs=1e-9)


def test_tv_no_nan_on_tiny_sigma():
    f, a, d = _flat_scene()
    f[4, 4, 0] = -1.0
    tv = depth_aware_tv_loss(f, a, d, sigma=1e-6)
    assert torch.isfinite(tv).all()


# ------------------------------------------------------------------ config guards
def test_regularizer_defaults_off():
    from semsplat.semsplat_model import SemsplatSplatfactoModelConfig

    cfg = SemsplatSplatfactoModelConfig()
    assert cfg.sem_tv_mult == 0.0
    assert cfg.opacity_entropy_mult == 0.0
    # entropy must not fight the pre-warmup densification seed population
    assert cfg.opacity_entropy_start_iter >= 500


# ------------------------------------------------------------------ GPU RGB+ED parity
pytestmark_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rgb_plus_ed_feature_channels_unperturbed():
    """RGB+ED must append the ED depth as the last channel and leave the K feature
    channels' premultiplied values identical to a plain RGB feature pass."""
    from gsplat.rendering import rasterization

    dev = "cuda"
    N, K = 2000, 8
    feats = torch.randn(N, K, device=dev) + 1.0
    means = (torch.rand(N, 3, device=dev) - 0.5) * 2.0
    quats = torch.randn(N, 4, device=dev)
    scales = (torch.rand(N, 3, device=dev) + 0.2) * 0.03
    opac = torch.rand(N, device=dev) * 0.9 + 0.1
    view = torch.eye(4, device=dev)[None]
    Kmat = torch.tensor([[160.0, 0, 80], [0, 160, 64], [0, 0, 1]], device=dev)[None]
    kw = dict(means=means, quats=quats, scales=scales, opacities=opac, viewmats=view,
              Ks=Kmat, width=160, height=128, sh_degree=None,
              packed=False, near_plane=0.01, far_plane=1e10)
    r_rgb, a_rgb, _ = rasterization(colors=feats, render_mode="RGB", **kw)
    r_ed, a_ed, _ = rasterization(colors=feats, render_mode="RGB+ED", channel_chunk=K + 1, **kw)
    assert r_ed.shape[-1] == K + 1
    torch.testing.assert_close(a_ed, a_rgb, atol=1e-5, rtol=1e-4)
    # premultiplied feature channels identical across the two modes
    torch.testing.assert_close(r_ed[..., :K], r_rgb, atol=1e-5, rtol=1e-4)
    # gsplat normalizes ONLY the last channel of RGB+ED => expected depth.
    # Check it equals the premultiplied D-mode render divided by alpha on covered px.
    r_d, a_d, _ = rasterization(colors=feats, render_mode="D", **kw)
    torch.testing.assert_close(a_d, a_ed, atol=1e-5, rtol=1e-4)
    cov = (a_ed > 0.5)[..., 0]
    if cov.any():
        torch.testing.assert_close(
            r_ed[..., K:][cov], (r_d / a_d.clamp_min(1e-6))[cov],
            atol=1e-3, rtol=1e-2,
        )
