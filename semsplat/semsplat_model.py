"""SemsplatModel: Splatfacto + per-Gaussian open-vocabulary semantic features.

Design (Feature-3DGS-style distillation, nerfstudio-native):
  * Each Gaussian stores an extra low-dim semantic vector ``sem_features`` in
    ``self.gauss_params`` (rows stay in lockstep with ``means`` through gsplat's
    DefaultStrategy densification, since it splats/duplicates/prunes every key of
    the params dict).
  * A second ``gsplat.rasterization`` call with ``sh_degree=None`` alpha-blends
    the ``[N, K]`` features into a per-pixel map (gsplat N-D feature pass). The
    premultiplied output is divided by alpha to give the expected feature map.
  * ``self.feature_head`` (``nn.Linear(K, M)``, shared decoder, the analogue of
    Feature-3DGS's 1x1-conv speed-up module) maps K -> teacher dim M (512 for
    OpenAI CLIP ViT-B/16).
  * Loss: rendered map is area-pooled to the teacher's patch grid and matched to
    a frozen OpenAI-CLIP dense (patch-token) teacher of the same training image
    with a normalized cosine / L1 term, gated on ``semantic_start_iter``.

NOTE on ``get_outputs``: it is a verbatim copy of Splatfacto v1.1.5's
``get_outputs`` (nerfstudio/models/splatfacto.py) with an added feature pass and
``# SEMSPLAT`` markers, because nerfstudio is pinned to exactly 1.1.5 and we need
the exact local viewmats/K/downscale used by the RGB pass. Keep in sync only if
the pin changes.
"""
# ruff: noqa: E741

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch.nn import Parameter

# Keep the same gsplat path nerfstudio v1.1.5 uses.
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
)
from nerfstudio.utils.rich_utils import CONSOLE
from gsplat.rendering import rasterization

EPS = 1e-6


@dataclass
class SemsplatSplatfactoModelConfig(SplatfactoModelConfig):
    """Splatfacto + per-Gaussian distilled semantic features."""

    _target: Type = field(default_factory=lambda: SemsplatModel)

    # ---- semantic feature field ----
    semantic_feature_dim: int = 32
    """Per-Gaussian semantic latent dimension K (K <= 32 keeps gsplat N-D fast)."""
    semantic_head_dim: int = 512
    """Dimension M of the teacher space the K-dim vectors are decoded to."""
    semantic_start_iter: int = 3000
    """Iteration at which the semantic distillation loss (and feature pass) turn on."""
    semantic_loss_type: str = "cos"
    """Feature distillation loss: 'cos' | 'l1' | 'mse' (on L2-normalized vectors)."""
    semantic_loss_mult: float = 1.0
    """Weight of the semantic distillation loss."""
    semantic_detach_geometry: bool = True
    """Detach means/quats/scales in the feature pass (feature loss doesn't move geometry)."""
    semantic_detach_opacity: bool = True
    """Detach opacities in the feature pass (feature loss doesn't move opacity/alpha)."""
    semantic_head_lr_mult: float = 1.0
    """Learning-rate multiplier applied when registering the semantic_head group."""
    semantic_feature_lr_mult: float = 1.0
    """Learning-rate multiplier applied to the sem_features group's optimizer LR."""

    # ---- teacher ----
    teacher_model_name: str = "openai/clip-vit-base-patch16"
    """HuggingFace OpenAI CLIP checkpoint used as dense local-feature teacher."""
    teacher_cache_fp16: bool = True
    """Cache teacher grids in fp16 to save VRAM."""
    teacher_cache_max_entries: int = 2000
    """Max cached (image_idx -> grid) entries on GPU (LRU eviction)."""


class SemsplatModel(SplatfactoModel):
    """Splatfacto with a distilled per-Gaussian open-vocabulary semantic field."""

    config: SemsplatSplatfactoModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clip_teacher = None  # lazy; NOT a submodule so it is never serialized
        self._last_cam_idx: Optional[int] = None

    # ------------------------------------------------------------------ setup
    def populate_modules(self):
        super().populate_modules()
        if self.config.semantic_feature_dim > 0:
            sem = torch.zeros((self.num_points, self.config.semantic_feature_dim))
            torch.nn.init.normal_(sem, std=0.01)
            self.gauss_params["sem_features"] = Parameter(sem)
            # Shared K -> M decoder. Kept OUT of gauss_params so densification never
            # touches it; it is a module and therefore saved in the state_dict.
            self.feature_head = torch.nn.Linear(
                self.config.semantic_feature_dim, self.config.semantic_head_dim
            )

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Include the new gaussian field so it gets its own optimizer group and is
        splatted/pruned in lockstep by gsplat's DefaultStrategy."""
        gps = super().get_gaussian_param_groups()
        if "sem_features" in self.gauss_params:
            gps["sem_features"] = [self.gauss_params["sem_features"]]
        return gps

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        gps = super().get_param_groups()
        if hasattr(self, "feature_head"):
            gps["semantic_head"] = list(self.feature_head.parameters())
        return gps

    # ------------------------------------------------------------------ render
    def get_outputs(self, camera) -> Dict[str, Union[torch.Tensor, List]]:
        """Copy of SplatfactoModel.get_outputs (v1.1.5) + a semantic feature pass."""
        if not isinstance(camera, Cameras):
            CONSOLE.log("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
            if camera.metadata is not None:
                self._last_cam_idx = camera.metadata.get("cam_idx")
            else:
                self._last_cam_idx = None
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization normalizes internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        # ============================ SEMSPLAT feature pass ====================
        # (only during the semantic phase of training; inference querying re-rasterizes
        #  class scores separately in semsplat.inference)
        if (
            self.config.semantic_feature_dim > 0
            and "sem_features" in self.gauss_params
            and self.training
            and self.step >= self.config.semantic_start_iter
        ):
            if crop_ids is not None:
                sem_crop = self.gauss_params["sem_features"][crop_ids]
            else:
                sem_crop = self.gauss_params["sem_features"]

            if self.config.semantic_detach_geometry:
                f_means, f_scales, f_quats = means_crop.detach(), scales_crop.detach(), quats_crop.detach()
            else:
                f_means, f_scales, f_quats = means_crop, scales_crop, quats_crop
            if self.config.semantic_detach_opacity:
                f_opac = opacities_crop.detach()
            else:
                f_opac = opacities_crop

            feat_render, feat_alpha, _ = rasterization(
                means=f_means,
                quats=f_quats,
                scales=torch.exp(f_scales),
                opacities=torch.sigmoid(f_opac).squeeze(-1),
                colors=sem_crop,  # [N, K] with sh_degree=None => N-D feature pass
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
                rasterize_mode=self.config.rasterize_mode,
            )
            # premultiplied output; divide by alpha => alpha-weighted expected feature
            feat_map = feat_render / feat_alpha.clamp_min(EPS)  # [1,H,W,K]
            outputs = {
                "sem_feature_map": feat_map.squeeze(0),  # [H,W,K]
                "sem_feature_alpha": feat_alpha.squeeze(0),  # [H,W,1]
            }
        else:
            outputs = {}
        # =======================================================================

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        outputs.update(
            {
                "rgb": rgb.squeeze(0),  # type: ignore
                "depth": depth_im,  # type: ignore
                "accumulation": alpha.squeeze(0),  # type: ignore
                "background": background,  # type: ignore
            }
        )
        return outputs  # type: ignore

    # ------------------------------------------------------------------ losses
    def _get_teacher(self):
        """Lazily build the frozen CLIP teacher (not part of the model graph)."""
        if self._clip_teacher is None:
            from semsplat.teachers.clip_local_teacher import ClipLocalTeacher

            self._clip_teacher = ClipLocalTeacher(
                model_name=self.config.teacher_model_name,
                fp16=self.config.teacher_cache_fp16,
                max_entries=self.config.teacher_cache_max_entries,
                device=self.device,
            )
            if self._clip_teacher.dim != self.config.semantic_head_dim:
                raise ValueError(
                    f"teacher dim {self._clip_teacher.dim} != semantic_head_dim "
                    f"{self.config.semantic_head_dim}; set semantic_head_dim accordingly."
                )
        return self._clip_teacher

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if not self.training:
            return loss_dict
        if "sem_feature_map" not in outputs:
            return loss_dict  # semantic phase not active yet

        feat_map = outputs["sem_feature_map"]  # [H,W,K]
        feat_alpha = outputs["sem_feature_alpha"]  # [H,W,1]

        # The GT image actually used for the RGB loss (same downscale as the render).
        gt_img = self.get_gt_img(batch["image"])  # [H,W,3] in [0,1]
        teacher = self._get_teacher()
        grid = teacher.compute(gt_img, key=self._last_cam_idx)  # [gH,gW,M]

        gH, gW, M = grid.shape
        pred = self.feature_head(feat_map)  # [H,W,M]
        # area-pool predicted features + alpha to the teacher's patch grid
        pred_pooled = F.adaptive_avg_pool2d(
            pred.permute(2, 0, 1)[None, ...], (gH, gW)
        )[0].permute(1, 2, 0)  # [gH,gW,M]
        alpha_pooled = F.adaptive_avg_pool2d(
            feat_alpha.permute(2, 0, 1)[None, ...], (gH, gW)
        )[0].permute(1, 2, 0)  # [gH,gW,1]

        pred_n = self._l2norm(pred_pooled)
        tgt_n = self._l2norm(grid)
        mask = alpha_pooled > 0.5  # valid (covered) patch cells

        loss_type = self.config.semantic_loss_type
        if loss_type == "cos":
            sim = (pred_n * tgt_n).sum(dim=-1)
            loss = (1.0 - sim)
        elif loss_type == "l1":
            loss = (pred_n - tgt_n).abs().sum(dim=-1)
        elif loss_type == "mse":
            loss = ((pred_n - tgt_n) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"unknown semantic_loss_type {loss_type}")

        if mask.any():
            sem_loss = loss[mask.squeeze(-1)].mean()
        else:
            sem_loss = torch.zeros((), device=self.device)

        loss_dict["semantic_loss"] = sem_loss * self.config.semantic_loss_mult
        return loss_dict
