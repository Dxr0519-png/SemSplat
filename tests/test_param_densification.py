"""Lockstep densification: gsplat's split/remove ops must keep the new
``sem_features`` gaussian field row-aligned with ``means`` (the mechanism the
model relies on). Requires a GPU."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)

from gsplat.strategy import ops  # noqa: E402


def _params(n: int, kdim: int = 8, dev: str = "cuda"):
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.randn(n, 3, device=dev)),
            "scales": torch.nn.Parameter(torch.log(torch.rand(n, 3, device=dev) * 0.1 + 0.01)),
            "quats": torch.nn.Parameter(torch.randn(n, 4, device=dev)),
            "opacities": torch.nn.Parameter(torch.logit(0.5 * torch.ones(n, 1, device=dev))),
            "sem_features": torch.nn.Parameter(torch.randn(n, kdim, device=dev) * 0.01),
        }
    )
    optimizers = {k: torch.optim.Adam([v]) for k, v in params.items()}
    state = {"counts": torch.ones(n, device=dev)}
    return params, optimizers, state


def test_split_keeps_sem_features_aligned():
    n = 64
    params, optimizers, state = _params(n)
    sel = torch.rand(n, device="cuda") > 0.6  # ~40% split
    ops.split(params, optimizers, state, sel)
    assert params["means"].shape[0] == params["sem_features"].shape[0]
    assert params["sem_features"].shape[0] > n
    # optimizer params updated to the new tensor
    assert optimizers["sem_features"].param_groups[0]["params"][0].shape == params["sem_features"].shape
    assert optimizers["means"].param_groups[0]["params"][0].shape[0] == params["means"].shape[0]


def test_remove_keeps_sem_features_aligned():
    n = 64
    params, optimizers, state = _params(n)
    rm = torch.rand(n, device="cuda") < 0.3
    ops.remove(params, optimizers, state, rm)
    assert params["means"].shape[0] == params["sem_features"].shape[0]
    assert params["sem_features"].shape[0] < n


def test_split_then_remove_roundtrip_counts():
    n = 100
    params, optimizers, state = _params(n)
    sel = torch.rand(n, device="cuda") > 0.5
    ops.split(params, optimizers, state, sel)
    n1 = params["means"].shape[0]
    rm = torch.rand(n1, device="cuda") < 0.4
    ops.remove(params, optimizers, state, rm)
    assert params["means"].shape[0] == params["sem_features"].shape[0]
    assert 0 < params["means"].shape[0] <= n1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
