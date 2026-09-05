"""Shared connected-component / area-threshold cleanup for semantic masks & maps.

Single home for the "merge fragmented masks / drop high-frequency small specks"
primitives so both sides reuse the same code:
  * the SAM+CLIP teacher (drop speckle islands inside a candidate SAM mask before
    it is CLIP-encoded; wall-fragment folding in the depth-aided path), and
  * the offline scripts (remove small wrong-class islands enclosed inside a big
    wall/ceiling component from rendered argmax maps, no retrain).

Pure numpy + cv2 -- no torch / nerfstudio dependency, safe to import from either
the teacher package or a repo-root sys.path script.

Conventions
  * label maps are int arrays; value ``255`` is the reserved "background/void"
    marker used by the offline renderers (kept frozen by default).
  * ``min_area`` is in pixels at the map's resolution.
"""
from __future__ import annotations

from typing import FrozenSet, Sequence

import cv2
import numpy as np

__all__ = [
    "component_stats",
    "remove_small_components",
    "merge_tiny_regions",
    "remove_enclosed_small_islands",
]


def component_stats(mask: np.ndarray, connectivity: int = 8):
    """connectedComponentsWithStats on a bool mask -> ``(labels, stats)``.

    Label 0 is background; foreground components are labels 1..N. Stats rows are
    cv2 CC_STAT_* (x, y, w, h, area). 8-connectivity is the default (diagonal
    neighbours count as one region), matching SAM's object notion better.
    """
    m = np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=connectivity)
    return labels, stats


def remove_small_components(binary: np.ndarray, min_area: int, keep_largest: bool = False) -> np.ndarray:
    """Drop every 8-connected foreground component with area < ``min_area``.

    Used by the teacher to excise speckle islands *inside* a SAM mask so a stray
    hole can never be encoded as its own object. Returns an all-False mask when
    nothing survives (the caller then drops that candidate mask); with
    ``keep_largest`` the single largest component is kept instead.
    """
    m = np.asarray(binary) > 0
    if min_area <= 1 or not m.any():
        return m
    labels, stats = component_stats(m, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return np.zeros_like(m, bool)
    keep = np.flatnonzero(areas >= min_area) + 1
    if keep.size:
        return np.isin(labels, keep)
    if keep_largest:
        big = int(np.argmax(areas)) + 1
        return labels == big
    return np.zeros_like(m, bool)


def merge_tiny_regions(
    label_map: np.ndarray,
    min_area: int,
    keep: FrozenSet[int] = frozenset({255}),
    connectivity: int = 4,
) -> np.ndarray:
    """Merge every component smaller than ``min_area`` into its dominant neighbour.

    Generalisation of ``SuperClipTeacher._merge_tiny`` from integer *labels* to
    *components of a class map*: a small isolated region (e.g. a thin seam between
    two wall masses that got a wrong label, or a leftover speck) is relabelled to
    the most frequent class among its 4-neighbours, restricted to classes that
    themselves occupy >= ``min_area`` pixels overall (fallback: the most populous
    non-keep class). Classes in ``keep`` (background/void) are never scanned and
    never assigned. Optional global denoise pass in the offline script.
    """
    out = np.array(label_map, copy=True)
    keep = set(keep)
    classes, counts = np.unique(out, return_counts=True)
    cand = [(int(c), int(n)) for c, n in zip(classes.tolist(), counts.tolist()) if int(c) not in keep]
    if not cand:
        return out
    cand.sort(key=lambda t: -t[1])
    big = {c for c, n in cand if n >= min_area}
    big_label = cand[0][0]  # most populous non-keep class
    kernel = np.ones((3, 3), np.uint8)
    for c, _n in cand:
        cmask = out == c
        if not cmask.any():
            continue
        clabels, cstats = component_stats(cmask, connectivity=connectivity)
        areas = cstats[1:, cv2.CC_STAT_AREA]
        small = np.flatnonzero(areas < min_area) + 1
        for comp in small.tolist():
            comp_mask = clabels == comp
            ring = cv2.dilate(comp_mask.astype(np.uint8), kernel) > 0
            ring[comp_mask] = False
            ys, xs = np.where(ring)
            if ys.size == 0:  # isolated blob with no neighbours at all
                target = big_label
            else:
                if ys.size > 400:  # sampling cap keeps big components cheap
                    sel = np.linspace(0, ys.size - 1, 400).astype(int)
                    ys, xs = ys[sel], xs[sel]
                vals, vc = np.unique(out[ys, xs], return_counts=True)
                order = np.argsort(-vc)
                target = None
                for vi in order:
                    v = int(vals[vi])
                    if v == c or v in keep:
                        continue
                    target = v
                    if v in big:
                        break  # largest qualifying neighbour is good enough
                if target is None:
                    target = big_label
            out[comp_mask] = target
    return out


def remove_enclosed_small_islands(
    argmax: np.ndarray,
    hosts: Sequence[int],
    min_area: int,
    bg: int = 255,
):
    """Relabel small foreign-class islands enclosed inside ``hosts`` to the host.

    The precise "high-frequency small specks on walls/ceilings" operator: a
    foreign (non-host, non-``bg``) 8-connected component with area < ``min_area``
    is treated as noise when the 3x3 ring around it lies entirely inside the host
    mask -- i.e. it is a speck floating on a big wall/ceiling region. It is then
    relabelled to the dominant host class on that ring (a speck on the ceiling
    becomes ceiling, on a wall becomes wall). Components touching the image
    border, or with any ring pixel that is another foreign class / background,
    are kept (partially visible objects and objects adjacent to clutter are not
    "enclosed islands").

    Returns ``(cleaned_argmax, n_islands)``.
    """
    out = np.array(argmax, copy=True)
    host = np.isin(out, hosts)
    foreign = (~host) & (out != bg)
    if min_area <= 1 or not foreign.any():
        return out, 0
    H, W = out.shape
    labels, stats = component_stats(foreign, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    small = np.flatnonzero(areas < min_area) + 1
    kernel = np.ones((3, 3), np.uint8)
    n_islands = 0
    for comp in small.tolist():
        comp_mask = labels == comp
        ys, xs = np.where(comp_mask)
        if ys.size == 0:
            continue
        # touches the frame border => partially visible, not an enclosed island
        if ys.min() == 0 or ys.max() == H - 1 or xs.min() == 0 or xs.max() == W - 1:
            continue
        ring = cv2.dilate(comp_mask.astype(np.uint8), kernel) > 0
        ring[comp_mask] = False
        if not ring.any() or not np.all(host[ring]):
            continue  # isolated or abuts another foreign class / bg / border
        vals, vc = np.unique(out[ring], return_counts=True)
        target = int(vals[np.argmax(vc)])
        out[comp_mask] = target
        n_islands += 1
    return out, n_islands
