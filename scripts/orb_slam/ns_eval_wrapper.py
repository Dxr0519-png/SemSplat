#!/usr/bin/env python
"""ns-eval entrypoint with torch>=2.6 weights_only monkeypatch (same quirk the
query_cli handles). Usage: ns_eval_wrapper.py --load-config <cfg> --output-path <m.json>
"""
from __future__ import annotations

import sys
import torch as _torch

_orig = _torch.load


def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig(*a, **k)


_torch.load = _load

from nerfstudio.scripts.eval import entrypoint  # noqa: E402

sys.exit(entrypoint())
