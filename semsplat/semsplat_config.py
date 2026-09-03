"""MethodSpecification(s) registering ``ns-train semsplat`` / ``semsplat-replica``."""
from __future__ import annotations

from typing import Optional

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from semsplat.semsplat_model import SemsplatSplatfactoModelConfig


def _base_optimizers(feature_lr: float, head_lr: float) -> dict:
    """Stock splatfacto optimizer/scheduler groups + the two new semantic groups."""
    opts = {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000),
        },
        "features_dc": {"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15), "scheduler": None},
        "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15), "scheduler": None},
        "opacities": {"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15), "scheduler": None},
        "scales": {"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15), "scheduler": None},
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "bilateral_grid": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        # ---- semsplat additions ----
        "sem_features": {
            "optimizer": AdamOptimizerConfig(lr=feature_lr, eps=1e-15),
            "scheduler": None,
        },
        "semantic_head": {
            "optimizer": AdamOptimizerConfig(lr=head_lr, eps=1e-15),
            "scheduler": None,
        },
    }
    return opts


def _make_trainer(method_name: str, model: SemsplatSplatfactoModelConfig, dataparser) -> TrainerConfig:
    return TrainerConfig(
        method_name=method_name,
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(dataparser=dataparser, cache_images_type="uint8"),
            model=model,
        ),
        optimizers=_base_optimizers(model.semantic_feature_lr_mult * 1e-3, model.semantic_head_lr_mult * 1e-3),
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    )


# Default: flexible (auto orient/center/scale) like stock splatfacto.
semsplat_method = MethodSpecification(
    config=_make_trainer(
        "semsplat",
        SemsplatSplatfactoModelConfig(),
        NerfstudioDataParserConfig(load_3D_points=True),
    ),
    description=(
        "Splatfacto + per-Gaussian open-vocabulary semantic features distilled from frozen "
        "OpenAI-CLIP local features (Feature-3DGS-style)."
    ),
)

# Replica-specific: keep metric ground-truth poses as-is (no auto orient/center/scale),
# bigger random init to cover the metric room scale, denser gaussian cap relaxed.
semsplat_replica_method = MethodSpecification(
    config=_make_trainer(
        "semsplat-replica",
        SemsplatSplatfactoModelConfig(num_random=50000, random_scale=5.0),
        NerfstudioDataParserConfig(
            orientation_method="none",
            center_method="none",
            auto_scale_poses=False,
            load_3D_points=True,
        ),
    ),
    description=(
        "semsplat preset for Replica processed RGB-D sequences (ground-truth poses kept "
        "metric; expects data already converted to nerfstudio transforms.json)."
    ),
)
