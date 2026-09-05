"""CPU-only tests for the AMG/context-crop pure helpers in sam_clip_teacher.py.

Covers the official-AMG stability score, the adaptive context window, the
mask-sharp on blurred/darkened background composite, and the IoU dedup used by
both mask paths. No SAM/CLIP weights are loaded (importing the module is cheap;
weights load only in SamClipTeacher.__init__).
"""
from __future__ import annotations

import numpy as np

from semsplat.teachers.sam_clip_teacher import (
    SamClipTeacher,
    _ctx_composite,
    _ctx_window,
    logit_stability,
)


# ------------------------------------------------------------- stability score
def test_stability_sharp_binary_mask_is_one():
    log = np.full((1, 1, 8, 8), -5.0)   # hard negative
    log[0, 0, :4] = 5.0                 # hard positive half
    # every >+1 pixel is also > -1  -> inter/union = 1
    assert logit_stability(log, offset=1.0) == 1.0


def test_stability_punishes_soft_decision_band():
    log = np.zeros((1, 1, 12, 1))
    log[0, 0, :3] = 5.0     # 3 confident positives
    log[0, 0, 3:9] = 0.0    # 6 in the ambiguous (-1, 1) band -> soft boundary
    log[0, 0, 9:] = -5.0    # 3 confident negatives (not counted in union)
    s = logit_stability(log, offset=1.0)
    assert s == 3.0 / 9.0


def test_stability_all_ambiguous_is_zero():
    log = np.zeros((2, 3, 4, 4))  # everything inside (-1, 1)
    assert (logit_stability(log, offset=1.0) == 0.0).all()


# ------------------------------------------------------------- context window
def test_ctx_window_small_mask_expands_by_margin():
    ys = np.array([50, 60]); xs = np.array([40, 52])
    h, w = 200, 200
    Y0, Y1, X0, X1 = _ctx_window(ys, xs, h, w, ctx_margin=0.25)
    side = max(60 - 50 + 1, 52 - 40 + 1)  # 13
    mg = int(round(13 * 0.25))            # 3
    assert (Y0, Y1, X0, X1) == (50 - 3, 61 + 3, 40 - 3, 53 + 3)


def test_ctx_window_floor_mask_keeps_bottom_layout_and_real_context():
    # floor reaches the frame bottom and spans the width: window must reach the
    # bottom edge (top/bottom layout preserved) and keep real context above it --
    # the anti "floor-becomes-ceiling" property of the gray-background baseline.
    ys = np.arange(150, 200); xs = np.arange(0, 200)
    h, w = 200, 200
    Y0, Y1, X0, X1 = _ctx_window(ys, xs, h, w, ctx_margin=0.25)
    assert Y1 == h                       # window pinned to the ground edge
    assert X0 == 0 and X1 == w           # full horizontal span
    assert Y0 == 100                     # 50 rows of context above the mask kept
    assert Y1 - Y0 > ys.size             # window is strictly taller than the mask


def test_ctx_window_clips_to_frame_edges():
    ys = np.array([0, 5]); xs = np.array([0, 5])
    Y0, Y1, X0, X1 = _ctx_window(ys, xs, 200, 200, ctx_margin=0.5)
    assert Y0 == 0 and X0 == 0            # never negative
    assert Y1 > 5 and X1 > 5              # expansion still applies on free sides


# ------------------------------------------------------------- bg composite
def _img(h, w):
    # smooth vertical colour ramp so blur genuinely mixes neighbouring values
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (xx / max(w - 1, 1))[..., None]
    g = (yy / max(h - 1, 1))[..., None]
    b = (np.sin(xx / 7.0) * 0.5 + 0.5)[..., None]
    return np.concatenate([r, g, b], axis=-1)  # 0..1 float


def test_ctx_composite_mask_pixels_stay_sharp_and_true_colour():
    img = _img(80, 96)
    mask = np.zeros((80, 96), bool); mask[20:60, 10:50] = True
    Y0, Y1, X0, X1 = _ctx_window(*np.where(mask), 80, 96, 0.25)
    comp = _ctx_composite(img, mask, Y0, Y1, X0, X1, blur_sigma=24.0, darken=0.0)
    mm = mask[Y0:Y1, X0:X1]
    expect = np.clip(img[Y0:Y1, X0:X1] * 255, 0, 255).round().astype(np.uint8)
    assert comp.dtype == np.uint8 and comp.shape == expect.shape
    assert (comp[mm] == expect[mm]).all()       # in-mask untouched
    assert (comp[~mm] != expect[~mm]).any()     # background actually blurred


def test_ctx_composite_darken_scales_background_only():
    img = _img(60, 60)
    mask = np.zeros((60, 60), bool); mask[10:50, 10:50] = True
    Y0, Y1, X0, X1 = _ctx_window(*np.where(mask), 60, 60, 0.25)
    comp = _ctx_composite(img, mask, Y0, Y1, X0, X1, blur_sigma=24.0, darken=0.3)
    win = np.clip(img[Y0:Y1, X0:X1] * 255, 0, 255).round().astype(np.uint8)
    mm = mask[Y0:Y1, X0:X1]
    assert (comp[mm] == win[mm]).all()
    # same float32 scale+round pipeline the function uses (avoids f32/f64 boundary diff)
    want = np.clip(win[~mm].astype(np.float32) * 0.3, 0, 255).round().astype(np.uint8)
    assert (comp[~mm] == want).all()


def test_ctx_composite_whole_window_mask_is_identity():
    img = _img(40, 40)
    mask = np.ones((40, 40), bool)
    win = np.clip(img * 255, 0, 255).round().astype(np.uint8)
    comp = _ctx_composite(img, mask, 0, 40, 0, 40, blur_sigma=24.0, darken=0.0)
    assert (comp == win).all()


# ------------------------------------------------------------- IoU dedup
def test_iou_static():
    a = np.zeros((20, 20), bool); a[0:10, 0:20] = True
    b = np.zeros((20, 20), bool); b[5:15, 0:20] = True
    assert abs(SamClipTeacher._iou(a, a) - 1.0) < 1e-9
    assert abs(SamClipTeacher._iou(a, b) - (100 / 300.0)) < 1e-9
    empty = np.zeros((20, 20), bool)
    assert SamClipTeacher._iou(a, empty) == 0.0
