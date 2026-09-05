"""Regularization losses for SemsplatModel (feature smoothness, alpha polarization).

Kept free of model imports so the math is unit-testable on CPU without building a
full splatfacto model (mirrors the primitive-level style of tests/test_regularizers.py).

Depth-aware TV loss
-------------------
Rendered semantic features are only supervised by the distillation loss at the
teacher's *coarse pooled* resolution (see ``SemsplatModel.get_loss_dict``), so
high-frequency per-pixel "salt and pepper" mislabelling inside an object is never
penalised. This term adds a total-variation smoothness prior on the **full-res**
rendered K-dim feature map, gated so it only acts *within a depth-continuous
surface* (``exp(-(Delta d)^2 / 2 sigma^2)`` on the gsplat expected-depth map), i.e.
it smooths intra-object speckle but never blurs across a physical edge.

Opacity entropy loss
--------------------
``alpha = sigmoid(logits)``; minimizing the binary entropy pushes every Gaussian's
alpha toward 0 or 1. Semi-transparent "foggy" Gaussians at object edges are forced
toward 0 and then culled by the base Splatfacto prune (``cull_alpha_thresh=0.1``),
which keeps semantics locked to physical boundaries.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def opacity_entropy_loss(opacity_logits: torch.Tensor) -> torch.Tensor:
    """Mean binary entropy (nats) of per-Gaussian alpha from raw logits ``[N,1]``.

    H(a) = -[a log a + (1-a) log(1-a)], a = sigmoid(z). Computed in logit space via
    softplus for numerical stability at a->0 / a->1 with no clamping bias:

        log a    = -softplus(-z)
        log(1-a) = -softplus(z)

    Gradient wrt logit z is ``a(1-a) ln((1-a)/a)``: it vanishes at the two minima
    a->{0,1} (where it is not needed) and peaks in the fog band a ~ 0.1-0.25; at the
    a=0.5 saddle H has a *local maximum* (unstable), so Adam escapes it rather than
    stalling. Do NOT implement via ``F.binary_cross_entropy_with_logits(z,
    sigmoid(z).detach())`` -- detaching the target kills the gradient.
    """
    a = torch.sigmoid(opacity_logits)
    log_a = -F.softplus(-opacity_logits)
    log_1ma = -F.softplus(opacity_logits)
    h = -(a * log_a + (1.0 - a) * log_1ma)
    return h.mean()


def depth_aware_tv_loss(
    feat_map: torch.Tensor,  # [H,W,K] rendered K-dim expected features (alpha-normalized)
    alpha: torch.Tensor,  # [H,W,1] coverage
    depth: torch.Tensor,  # [H,W,1] expected (ED) view-space depth, metres, no grad needed
    sigma: float,  # depth-continuity scale (metres)
    alpha_thresh: float = 0.5,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Depth-gated total variation on the per-pixel K-dim feature map.

    Penalises squared L2 difference between *adjacent* feature pixels only where
    both endpoints are covered (``alpha > alpha_thresh``) and lie on a
    depth-continuous surface. The neighbour weight ``exp(-Delta_d^2 / 2 sigma^2)``
    suppresses smoothing across physical edges while smoothing intra-object speckle.

    Features are L2-normalized per pixel first (same normalizing convention as the
    teacher-space cosine distillation), making the term scale-invariant and bounded
    O(0..4), so ``sem_tv_mult`` is directly comparable to ``semantic_loss_mult``.
    Gradient flows only into ``feat_map`` (the sem_features raster): ``depth`` is
    detached and ``alpha`` carries no gradient through the semantic feature pass
    (geometry/opacity are detached there).
    """
    a = alpha[..., 0]  # [H,W]
    cov = a > alpha_thresh  # [H,W] bool
    d = depth[..., 0].detach()  # [H,W] metres

    # per-pixel unit vectors (scale-invariant features)
    fn = feat_map / feat_map.norm(dim=-1, keepdim=True).clamp_min(eps)  # [H,W,K]

    # neighbour differences (drop last row / last col)
    fv = fn[1:] - fn[:-1]  # [H-1,W,K]  vertical (rows)
    fh = fn[:, 1:] - fn[:, :-1]  # [H,W-1,K] horizontal (cols)
    dv = (d[1:] - d[:-1]).abs()  # [H-1,W]
    dh = (d[:, 1:] - d[:, :-1]).abs()  # [H,W-1]

    cov_v = cov[1:] & cov[:-1]  # both endpoints covered
    cov_h = cov[:, 1:] & cov[:, :-1]

    gv = torch.exp(-(dv * dv) / (2.0 * sigma * sigma))
    gh = torch.exp(-(dh * dh) / (2.0 * sigma * sigma))
    wv = torch.where(cov_v, gv, torch.zeros_like(gv))
    wh = torch.where(cov_h, gh, torch.zeros_like(gh))

    num = ((fv.square().sum(-1)) * wv).sum() + ((fh.square().sum(-1)) * wh).sum()
    den = wv.sum() + wh.sum()
    if not torch.isfinite(num):
        return torch.zeros((), device=feat_map.device, dtype=feat_map.dtype)
    return num / den.clamp_min(eps)
