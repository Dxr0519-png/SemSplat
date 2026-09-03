"""Load the vendored Feature-3DGS / isl-org LSeg dense image encoder.

The shipped ``demo_e200.ckpt`` is a pytorch-lightning checkpoint whose
``state_dict`` keys are prefixed ``net.*``; the part we need is the *dense image
encoder* (timm ViT-L/16@384 backbone + DPT-style refinenet + head1/output_conv).
The CLIP text tower baked into the original module (``net.clip_pretrained.*``)
is not needed for the dense-image path and is dropped here, so no ``clip``
package is required.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

CKPT_DEFAULT = "data/checkpoints/lseg/demo_e200.ckpt"
_BACKBONE = "clip_vitl16_384"


def build_lseg_net(
    ckpt_path: Optional[str] = None,
    device: str = "cpu",
    fp16: bool = True,
    strict_report: bool = True,
) -> torch.nn.Module:
    """Build the LSegNet dense encoder and (optionally) load pretrained weights."""
    from .lseg_net import LSegNet  # local import keeps heavy module lazy

    net = LSegNet(
        labels=[],
        backbone=_BACKBONE,
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Mirror LSegModule: pretend the input is 480x480 before the ckpt pos-embed loads.
    net.pretrained.model.patch_embed.img_size = (480, 480)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
        own = {}
        for k, v in sd.items():
            if k.startswith("net.clip_pretrained."):
                continue  # CLIP text tower baked into the original checkpoint
            if k.startswith("net."):
                k = k[4:]
            own[k] = v
        missing, unexpected = net.load_state_dict(own, strict=False)
        missing = sorted(m for m in missing if not m.startswith("clip_pretrained"))
        unexpected = sorted(u for u in unexpected if not u.startswith("clip_pretrained"))
        if strict_report:
            print(f"[lseg] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
            if missing:
                print("  e.g.", missing[:8])
            if unexpected:
                print("  unexpected e.g.", unexpected[:8])
        elif missing or unexpected:
            print(f"[lseg] WARNING: missing {len(missing)}, unexpected {len(unexpected)}")
    else:
        print(f"[lseg] WARNING: ckpt not found at {ckpt_path!r}; using random weights")

    net = net.to(device)
    if fp16:
        net = net.half()
    net.eval()
    return net
