"""Frozen OpenAI-CLIP dense (patch-token) teacher + CLIP text encoder.

Why HF ``transformers`` and not raw OpenCLIP: we need per-patch embeddings from a
ViT, which the reference implementation does not expose directly, and we need
positional-embedding interpolation for non-224x224 inputs. HF's
``CLIPVisionModel`` supports ``interpolate_pos_encoding=True`` and returns
``last_hidden_state`` with an explicit patch grid, which is the cleanest way to
get a deterministic ``[gH, gW, D]`` grid at any input resolution. The weights are
OpenAI's (``openai/clip-vit-base-patch16``) so the feature space is exactly the
"pure CLIP local features" chosen for the teacher; each patch token is pushed
through ``CLIPModel.visual_projection`` (768 -> 512) to align it with the shared
CLIP text space used for open-vocabulary queries.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

_PATCH = 16  # ViT patch size used for the default checkpoint


def _to_patch_grid(image: torch.Tensor) -> torch.Tensor:
    """Resize a [H, W, 3] float image (0..1) so H and W are multiples of 16."""
    H, W = image.shape[:2]
    gH, gW = math.ceil(H / _PATCH), math.ceil(W / _PATCH)
    hh, ww = gH * _PATCH, gW * _PATCH
    if (hh, ww) != (H, W):
        img = image.permute(2, 0, 1)[None, ...]  # [1,3,H,W]
        img = F.interpolate(img, size=(hh, ww), mode="bilinear", align_corners=False)
        image = img[0].permute(1, 2, 0)
    return image


class ClipLocalTeacher:
    """Frozen OpenAI-CLIP image encoder returning L2-normalized patch-token grids."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        fp16: bool = True,
        max_entries: int = 2000,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.fp16 = fp16
        self.device = torch.device(device)
        self.dim: int = None  # type: ignore[assignment]  # set below
        self._cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._cache_max = int(max_entries)

        dtype = torch.float16 if self.fp16 else torch.float32
        self._model = CLIPModel.from_pretrained(model_name).to(self.device).eval().to(dtype)
        # frozen
        for p in self._model.parameters():
            p.requires_grad_(False)
        hidden = self._model.vision_model.config.hidden_size
        proj = self._model.visual_projection.out_features if hasattr(self._model.visual_projection, "out_features") else None
        self.dim = int(proj if proj is not None else hidden)
        self._dtype = dtype

    @torch.no_grad()
    def dense_features(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a [H, W, 3] float image (0..1) -> L2-normalized [gH, gW, self.dim]."""
        image = image.detach().to(self.device)
        image = _to_patch_grid(image)
        gH, gW = image.shape[0] // _PATCH, image.shape[1] // _PATCH

        mean = CLIP_MEAN.to(image.device)
        std = CLIP_STD.to(image.device)
        img = (image - mean) / std
        px = img.permute(2, 0, 1)[None, ...].to(self._dtype)  # [1,3,H,W]

        out = self._model.vision_model(pixel_values=px, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state[0]  # [1+gH*gW, hidden]
        patch = tokens[1:]  # drop [CLS]
        patch = patch.reshape(gH, gW, -1)
        feats = self._model.visual_projection(patch)  # [gH,gW,dim] (half model -> half)
        feats = feats.float()
        return F.normalize(feats, dim=-1)

    def compute(self, image: torch.Tensor, key: Optional[int] = None) -> torch.Tensor:
        """Cached dense_features. ``key`` is an image identity (cam_idx) for the LRU."""
        if key is not None and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        grid = self.dense_features(image)
        if key is not None:
            if self.fp16:
                grid = grid.half()
            self._cache[key] = grid
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return grid

    def clear_cache(self) -> None:
        self._cache.clear()


def load_clip(model_name: str = "openai/clip-vit-base-patch16", fp16: bool = False, device: str = "cuda"):
    """Build a frozen CLIPModel (fp32/fp16) for text encoding / projection access."""
    dtype = torch.float16 if fp16 else torch.float32
    model = CLIPModel.from_pretrained(model_name).to(device).eval().to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_text_prompts(
    prompts, model_name: str = "openai/clip-vit-base-patch16", device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """L2-normalized text embeddings (in CLIP's projected space) for each prompt."""
    from transformers import CLIPTokenizer  # local import kept near usage

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = load_clip(model_name, fp16=False, device=device)

    out = {}
    for p in prompts:
        tokens = tokenizer(p, return_tensors="pt", padding=True, truncation=True).to(device)
        pooled = model.text_model(**tokens).pooler_output  # [1, hidden]
        emb = model.text_projection(pooled)  # [1, proj]
        out[p] = F.normalize(emb.float(), dim=-1)[0]
    return out
