"""Rasterize per-class open-vocabulary score maps from a trained SemsplatModel."""
from __future__ import annotations

from typing import Tuple

import torch
from gsplat.rendering import rasterization

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import get_viewmat

EPS = 1e-6


@torch.no_grad()
def per_gaussian_semantics(model) -> torch.Tensor:
    """L2-normalized per-Gaussian semantic vectors in CLIP teacher space, [N, M]."""
    assert hasattr(model, "gauss_params") and "sem_features" in model.gauss_params
    sem = model.gauss_params["sem_features"]
    out = model.feature_head(sem)
    return torch.nn.functional.normalize(out, dim=-1)


@torch.no_grad()
def rasterize_class_scores(
    model, camera: Cameras, class_scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render per-Gaussian ``[N, C]`` class_scores to ``[C, H, W]`` maps + ``[H, W, 1]`` alpha.

    Mirrors the training-time feature pass: ``sh_degree=None`` alpha-blends the
    score channels, premultiplied output is divided by alpha.
    """
    assert class_scores.shape[0] == model.num_points
    device = model.device
    camera = camera.to(device)
    optimized_camera_to_world = camera.camera_to_worlds
    viewmat = get_viewmat(optimized_camera_to_world)
    K = camera.get_intrinsics_matrices().to(device)
    W, H = int(camera.width.item()), int(camera.height.item())

    means = model.means
    quats = model.quats
    scales = torch.exp(model.scales)
    opacities = torch.sigmoid(model.opacities).squeeze(-1)

    render, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=class_scores.float().to(device),  # [N, C]
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        sh_degree=None,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode=model.config.rasterize_mode,
    )
    maps = render[0] / alpha[0].clamp_min(EPS)  # [H,W,C]
    return maps, alpha[0]


@torch.no_grad()
def semantic_maps_from_prompts(
    model, camera: Cameras, prompt_embs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given normalized text embeddings ``[C, M]`` return score maps [H,W,C] + alpha."""
    sem = per_gaussian_semantics(model)  # [N, M]
    scores = sem @ prompt_embs.T  # [N, C]
    return rasterize_class_scores(model, camera, scores)
