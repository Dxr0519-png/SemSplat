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
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

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

from semsplat.losses import (
    depth_aware_tv_loss,
    opacity_entropy_loss,
)

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

    # ---- spatial regularizers (polish; both default OFF so old configs/checkpoints
    #      stay byte-comparable) ----
    sem_tv_mult: float = 0.0
    """Weight of the full-res depth-aware TV smoothness on the rendered K-dim feature
    map (0 = off). When >0 the semantic feature pass renders RGB+ED to also emit depth."""
    sem_tv_depth_sigma: float = 0.1
    """Depth-continuity sigma (metres) for the TV edge gate exp(-Delta_d^2 / 2 sigma^2)."""
    sem_tv_eps: float = 1e-3
    """Small constant: per-pixel feature-norm clamp + weighted-mean denominator."""
    opacity_entropy_mult: float = 0.0
    """Weight of the mean per-Gaussian binary-entropy (alpha polarization) loss (0 = off)."""
    opacity_entropy_start_iter: int = 1000
    """Iteration at which the opacity-entropy term turns on (>= refine warmup 500 so it
    does not collapse the initial alpha~0.1 seed population before geometry is built)."""

    # ---- teacher ----
    teacher_kind: str = "clip"
    """'clip' = OpenAI-CLIP patch | 'lseg' | 'superclip' = SLIC+CLIP | 'samclip' = SAM+CLIP."""
    teacher_sam_model_name: str = "facebook/sam-vit-base"
    """HF SAM checkpoint used for whole-object masks (samclip)."""
    teacher_sam_grid: int = 6
    """grid x grid SAM point prompts (samclip)."""
    teacher_sam_iou_thr: float = 0.86
    """Keep SAM masks whose predicted IoU >= thr (samclip)."""
    teacher_sam_mask_bg: float = 0.5
    """Crop background value 0..1 for CLIP crops (0 black, 0.5 neutral gray)."""
    # samclip context guidance + mask hygiene + global-first wall folding.
    # Every toggle below defaults to the pre-change behaviour so existing runs /
    # checkpoints stay byte-comparable with the baseline.
    teacher_sam_context_ratio: float = 0.6
    """Context pad (= ratio of the object bbox side, per side) around CLIP crops."""
    teacher_sam_context_weight: float = 0.0
    """Dual-branch CLIP fusion weight; 0 = disabled (baseline gray-bg object-only path)."""
    teacher_sam_mask_min_comp_px: int = 0
    """Drop <min_area speck islands inside a SAM mask before encoding (0 = off)."""
    teacher_sam_depth_dir: Optional[Path] = None
    """Dir of {cam_idx:06d}.png uint16-mm depth + parent transforms.json (K); None = folding off."""
    teacher_sam_fold_enabled: bool = False
    """Master switch for depth-aided plane detection + wall-fragment folding."""
    teacher_sam_fold_tol_m: float = 0.03
    """Plane-support residual (m) for RANSAC plane fitting."""
    teacher_sam_fold_min_plane_frac: float = 0.15
    """A dominant plane must cover >= this fraction of the valid-depth sample."""
    teacher_sam_fold_max_plane_n: int = 2
    """Fit up to this many dominant planes (back wall + side wall)."""
    teacher_sam_fold_max_area_frac: float = 0.03
    """Only fold independent fragments <= this frame-area fraction (protects tv/clock)."""
    teacher_sam_fold_rgb_maxdiff: Optional[float] = 0.15
    """Albedo veto: fold only if the fragment colour ~ its wall ring (mean |diff|); None = depth only."""
    # ---- samclip native AMG (automatic mask generator) + contextual CLIP crops ----
    # Defaults are the NEW behaviour; set teacher-sam-mode grid + teacher-sam-crop-mode gray
    # to reproduce the legacy 6x6-point/gray-background baseline exactly.
    teacher_sam_mode: str = "amg"
    """samclip mask generator: 'amg' = native auto-mask grid | 'grid' = legacy 6x6 per-point loop."""
    teacher_sam_points_per_side: int = 8
    """amg: points_per_side^2 grid of single-point SAM prompts (full-image coverage)."""
    teacher_sam_stability_thr: float = 0.9
    """amg: keep candidates whose logit stability >= thr."""
    teacher_sam_stability_offset: float = 1.0
    """amg: logit offset used by the stability score."""
    teacher_sam_stability_big_area_frac: float = 0.08
    """amg: masks >= this frame area fraction are exempt from the stability gate (soft low-texture planes)."""
    teacher_sam_max_masks: int = 40
    """amg: cap on kept masks after greedy IoU dedup (bounds the CLIP crop batch)."""
    teacher_sam_mask_overlap: float = 0.85
    """amg: greedy IoU dedup overlap threshold."""
    teacher_sam_amg_long_side: int = 1024
    """amg: SAM working-res long side (legacy grid path stays at 512)."""
    teacher_sam_crop_mode: str = "ctx"
    """samclip CLIP crop: 'ctx' = adaptive blur/darken context window | 'gray' = legacy mask_bg crop."""
    teacher_sam_ctx_margin: float = 0.25
    """ctx crop: window expand = margin * mask bbox side, per side."""
    teacher_sam_ctx_blur_sigma: float = 24.0
    """ctx crop: gaussian sigma at a 512px window (auto-scaled to window size)."""
    teacher_sam_ctx_darken: float = 0.0
    """ctx crop: >0 darkens the non-mask background by this factor instead of blurring."""
    teacher_slic_n_segments: int = 250
    """SLIC superpixel count for the superclip teacher."""
    teacher_slic_compactness: float = 10.0
    """SLIC compactness for the superclip teacher."""
    teacher_model_name: str = "openai/clip-vit-base-patch16"
    """HuggingFace OpenAI CLIP checkpoint used as dense local-feature teacher (clip kind)."""
    teacher_text_model_name: Optional[str] = None
    """Optional override for the query text encoder; None resolves from teacher_kind."""
    teacher_lseg_ckpt: str = "data/checkpoints/lseg/demo_e200.ckpt"
    """Feature-3DGS/isl-org LSeg checkpoint (clip_vitl16_384 + demo_e200)."""
    teacher_lseg_input_long_side: int = 384
    """Long side (px) the LSeg teacher resizes training images to."""
    teacher_lseg_cell: int = 8
    """Average-pool teacher pixels into ~input_long_side/cell supervision grid."""
    teacher_lseg_l2norm: bool = True
    """Channel-L2-normalize LSeg per-pixel features before pooling."""
    teacher_cache_fp16: bool = True
    """Cache teacher grids in fp16 to save VRAM."""
    teacher_cache_max_entries: int = 2000
    """Max cached (image_idx -> grid) entries on GPU (LRU eviction)."""

    # ---- (geometric target injection removed 2026-09-05: ceiling-only arm B′ fixed
    #      ceiling 0.005→0.280 but flooded walls with "flat white ceiling" text
    #      (2.6M GT-wall px), mIoU 0.196 < reg 0.240. Rolled back to plain reg;
    #      mechanism + results recorded in docs/2026-09-05_几何目标灌注.md.) ----


class SemsplatModel(SplatfactoModel):
    """Splatfacto with a distilled per-Gaussian open-vocabulary semantic field."""

    config: SemsplatSplatfactoModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher = None  # lazy; NOT a submodule so it is never serialized
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

            # depth-aware TV needs a per-pixel depth aligned to the feature map; gsplat
            # "RGB+ED" appends expected z-depth (already alpha-normalized) as the last
            # channel, so the K feature channels below are untouched by this switch.
            need_depth = self.config.sem_tv_mult > 0
            feat_render_mode = "RGB+ED" if need_depth else "RGB"
            sem_K = sem_crop.shape[-1]
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
                render_mode=feat_render_mode,
                sh_degree=None,
                # keep the [N,K+1] blended buffer in a single kernel at the default
                # K=32 (channel_chunk would otherwise split 33 channels into two)
                channel_chunk=sem_K + (1 if feat_render_mode == "RGB+ED" else 0),
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.config.rasterize_mode,
            )
            # premultiplied output; divide by alpha => alpha-weighted expected feature
            feat_map = feat_render[..., :sem_K] / feat_alpha.clamp_min(EPS)  # [1,H,W,K]
            outputs = {
                "sem_feature_map": feat_map.squeeze(0),  # [H,W,K]
                "sem_feature_alpha": feat_alpha.squeeze(0),  # [H,W,1]
            }
            if feat_render_mode == "RGB+ED":
                outputs["sem_feature_depth"] = feat_render[..., sem_K:].squeeze(0)  # [H,W,1]
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
        """Lazily build the frozen teacher (clip | lseg), not part of the model graph."""
        if self._teacher is None:
            cfg = self.config
            if cfg.teacher_kind == "lseg":
                from semsplat.teachers.lseg_local_teacher import LSegDenseTeacher

                self._teacher = LSegDenseTeacher(
                    ckpt_path=cfg.teacher_lseg_ckpt,
                    input_long_side=cfg.teacher_lseg_input_long_side,
                    cell=cfg.teacher_lseg_cell,
                    l2norm=cfg.teacher_lseg_l2norm,
                    fp16=cfg.teacher_cache_fp16,
                    max_entries=cfg.teacher_cache_max_entries,
                    device=str(self.device),
                )
            elif cfg.teacher_kind == "superclip":
                from semsplat.teachers.slic_clip_teacher import SuperClipTeacher

                self._teacher = SuperClipTeacher(
                    image_model_name=cfg.teacher_model_name,
                    n_segments=cfg.teacher_slic_n_segments,
                    compactness=cfg.teacher_slic_compactness,
                    fp16=cfg.teacher_cache_fp16,
                    max_entries=cfg.teacher_cache_max_entries,
                    device=str(self.device),
                )
            elif cfg.teacher_kind == "samclip":
                from semsplat.teachers.sam_clip_teacher import SamClipTeacher

                self._teacher = SamClipTeacher(
                    image_model_name=cfg.teacher_model_name,
                    sam_model_name=cfg.teacher_sam_model_name,
                    grid=cfg.teacher_sam_grid,
                    iou_thr=cfg.teacher_sam_iou_thr,
                    mask_bg=cfg.teacher_sam_mask_bg,
                    context_ratio=cfg.teacher_sam_context_ratio,
                    context_weight=cfg.teacher_sam_context_weight,
                    mask_min_comp_px=cfg.teacher_sam_mask_min_comp_px,
                    depth_dir=str(cfg.teacher_sam_depth_dir) if cfg.teacher_sam_depth_dir else None,
                    fold_enabled=cfg.teacher_sam_fold_enabled,
                    fold_tol_m=cfg.teacher_sam_fold_tol_m,
                    fold_min_plane_frac=cfg.teacher_sam_fold_min_plane_frac,
                    fold_max_plane_n=cfg.teacher_sam_fold_max_plane_n,
                    fold_max_area_frac=cfg.teacher_sam_fold_max_area_frac,
                    fold_rgb_maxdiff=cfg.teacher_sam_fold_rgb_maxdiff,
                    sam_mode=cfg.teacher_sam_mode,
                    points_per_side=cfg.teacher_sam_points_per_side,
                    stability_thr=cfg.teacher_sam_stability_thr,
                    stability_offset=cfg.teacher_sam_stability_offset,
                    stability_big_area_frac=cfg.teacher_sam_stability_big_area_frac,
                    max_masks=cfg.teacher_sam_max_masks,
                    mask_overlap=cfg.teacher_sam_mask_overlap,
                    amg_long_side=cfg.teacher_sam_amg_long_side,
                    crop_mode=cfg.teacher_sam_crop_mode,
                    ctx_margin=cfg.teacher_sam_ctx_margin,
                    ctx_blur_sigma=cfg.teacher_sam_ctx_blur_sigma,
                    ctx_darken=cfg.teacher_sam_ctx_darken,
                    fp16=cfg.teacher_cache_fp16,
                    max_entries=cfg.teacher_cache_max_entries,
                    device=str(self.device),
                )
            else:
                from semsplat.teachers.clip_local_teacher import ClipLocalTeacher

                self._teacher = ClipLocalTeacher(
                    model_name=cfg.teacher_model_name,
                    fp16=cfg.teacher_cache_fp16,
                    max_entries=cfg.teacher_cache_max_entries,
                    device=str(self.device),
                )
            if self._teacher.dim != cfg.semantic_head_dim:
                raise ValueError(
                    f"teacher dim {self._teacher.dim} != semantic_head_dim "
                    f"{cfg.semantic_head_dim}; set semantic_head_dim accordingly."
                )
        return self._teacher

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if not self.training:
            return loss_dict

        # ---- alpha polarization: push each Gaussian's opacity toward 0 or 1 so the
        #      semi-transparent "foggy" edge Gaussians get culled (polish) ----
        if self.config.opacity_entropy_mult > 0 and self.step >= self.config.opacity_entropy_start_iter:
            loss_dict["opacity_entropy"] = (
                opacity_entropy_loss(self.opacities) * self.config.opacity_entropy_mult
            )

        if "sem_feature_map" not in outputs:
            return loss_dict  # semantic phase not active yet

        feat_map = outputs["sem_feature_map"]  # [H,W,K]
        feat_alpha = outputs["sem_feature_alpha"]  # [H,W,1]

        # The GT image actually used for the RGB loss (same downscale as the render).
        gt_img = self.get_gt_img(batch["image"])  # [H,W,3] in [0,1]
        teacher = self._get_teacher()
        grid = teacher.compute(gt_img, key=self._last_cam_idx)  # [gH,gW,M]

        gH, gW, M = grid.shape
        # area-pool the K-dim map, then decode to M (linear + avg-pool commute, so
        # this avoids materializing the [H,W,M] tensor before pooling)
        pred_pooled = F.adaptive_avg_pool2d(
            feat_map.permute(2, 0, 1)[None, ...], (gH, gW)
        )[0].permute(1, 2, 0)  # [gH,gW,K]
        pred_pooled = self.feature_head(pred_pooled)  # [gH,gW,M]
        alpha_pooled = F.adaptive_avg_pool2d(
            feat_alpha.permute(2, 0, 1)[None, ...], (gH, gW)
        )[0].permute(1, 2, 0)  # [gH,gW,1]

        pred_n = self._l2norm(pred_pooled)
        tgt_n = self._l2norm(grid)
        mask = alpha_pooled > 0.5  # geometry coverage
        cov_fn = getattr(teacher, "coverage", None)
        if cov_fn is not None:
            cov = cov_fn(gt_img, key=self._last_cam_idx)  # [gH,gW]
            if cov.ndim == 2:
                cov = cov[..., None]
            mask = mask & (cov > 0.4)

        loss_type = self.config.semantic_loss_type

        # per-cell distillation against the L2-normalized teacher target. (This is the
        # plain "standard reg" loss; the geometric target-injection mechanism that
        # temporarily overrode ``tgt_n`` was rolled back 2026-09-05 after arm B′ fixed
        # ceiling 0.005->0.280 but flooded 2.6M GT-wall px with "flat white ceiling"
        # text (mIoU 0.196 < reg 0.240). See docs/2026-09-05_几何目标灌注.md.)
        if loss_type == "cos":
            loss = (1.0 - (pred_n * tgt_n).sum(dim=-1))
        elif loss_type == "l1":
            loss = (pred_n - tgt_n).abs().sum(dim=-1)
        elif loss_type == "mse":
            loss = ((pred_n - tgt_n) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"unknown semantic_loss_type {loss_type}")
        loss_dict["semantic_loss"] = loss[mask.squeeze(-1)].mean() * self.config.semantic_loss_mult

        # ---- full-res depth-aware TV on the K-dim feature map (speckle removal).
        #      Gated by rendered-depth continuity so it never smooths across a
        #      physical edge; runs before the teacher-grid pooling above. ----
        if self.config.sem_tv_mult > 0:
            loss_dict["sem_tv"] = depth_aware_tv_loss(
                feat_map=outputs["sem_feature_map"],  # [H,W,K]
                alpha=outputs["sem_feature_alpha"],  # [H,W,1]
                depth=outputs["sem_feature_depth"],  # [H,W,1] (present because RGB+ED)
                sigma=self.config.sem_tv_depth_sigma,
                eps=self.config.sem_tv_eps,
            ) * self.config.sem_tv_mult

        return loss_dict
