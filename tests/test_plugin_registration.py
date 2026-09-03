"""Method registration + config sanity (no GPU needed)."""
from __future__ import annotations

import pytest

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

import semsplat.semsplat_config as scfg
from semsplat.semsplat_model import SemsplatModel, SemsplatSplatfactoModelConfig


def _assert_method(method: MethodSpecification, name: str):
    cfg = method.config
    assert isinstance(cfg, TrainerConfig)
    assert cfg.method_name == name
    assert set(cfg.optimizers) >= {"sem_features", "semantic_head"}
    model: SemsplatSplatfactoModelConfig = cfg.pipeline.model  # type: ignore[attr-defined]
    assert model._target is SemsplatModel
    assert model.semantic_feature_dim > 0


def test_semsplat_method_registered():
    _assert_method(scfg.semsplat_method, "semsplat")


def test_semsplat_replica_method_registered():
    _assert_method(scfg.semsplat_replica_method, "semsplat-replica")


def test_optimizer_groups_match_param_groups():
    """Every optimizer group the model will declare must exist in the TrainerConfig."""
    for method in (scfg.semsplat_method, scfg.semsplat_replica_method):
        cfg = method.config
        model = cfg.pipeline.model  # type: ignore[attr-defined]
        assert set(cfg.optimizers) >= {
            "means", "scales", "quats", "features_dc", "features_rest", "opacities",
            "sem_features", "semantic_head",
        }
        # semantic head dim matches the CLIP teacher projection dim (512)
        assert model.semantic_head_dim == 512
        assert model.teacher_model_name == "openai/clip-vit-base-patch16"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
