#!/usr/bin/env python
"""ns-viewer entrypoint with torch>=2.6 weights_only monkeypatch.

Usage: ns_viewer_wrapper.py --load-config <cfg.yml> [--viewer.websocket-port 7007]
"""
from __future__ import annotations

import sys
import torch as _torch

_orig = _torch.load
_torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

from nerfstudio.scripts.viewer.run_viewer import entrypoint  # noqa: E402

sys.exit(entrypoint())
