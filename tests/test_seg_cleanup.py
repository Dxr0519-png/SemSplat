"""CPU-only unit tests for the shared connected-component cleanup utilities."""
from __future__ import annotations

import numpy as np

from semsplat.seg_cleanup import (
    merge_tiny_regions,
    remove_enclosed_small_islands,
    remove_small_components,
)


def test_remove_small_components_drops_speck_inside_big_mask():
    m = np.zeros((60, 60), bool)
    m[10:50, 10:50] = True  # big blob (still one component once the hole is cut)
    m[30:35, 30:35] = False  # interior hole
    m[2:6, 2:6] = True  # small 16-px speck, disconnected
    out = remove_small_components(m, min_area=20)
    # big blob survives with its interior hole intact; 16-px speck is gone
    expected = m.copy()
    expected[2:6, 2:6] = False
    assert (out == expected).all()
    assert not out[2:6, 2:6].any()
    assert out.sum() == 40 * 40 - 5 * 5


def test_remove_small_components_all_small_returns_empty_by_default():
    m = np.zeros((30, 30), bool)
    m[5:9, 5:9] = True  # 16 px
    assert not remove_small_components(m, min_area=20).any()
    # keep_largest restores the single biggest component instead
    out = remove_small_components(m, min_area=20, keep_largest=True)
    assert out.sum() == 16


def test_remove_small_components_noop_with_small_threshold():
    m = np.zeros((20, 20), bool)
    m[5:8, 5:8] = True
    out = remove_small_components(m, min_area=1)
    assert (out == m).all()


def test_merge_tiny_island_into_dominant_neighbour():
    a = np.full((200, 200), 5, np.int64)  # a big "wall" region (class 5)
    a[95:100, 95:103] = 1  # 40-px speck of a foreign class
    a[60:62, 60:75] = 2  # 30-px class-2 sliver, also below min_area
    out = merge_tiny_regions(a, min_area=60)
    # both tiny islands merge into the surrounding wall class
    assert (out[95:100, 95:103] == 5).all()
    assert (out[60:62, 60:75] == 5).all()
    assert (out == 5).all()  # nothing else left


def test_merge_tiny_regions_keeps_bg_untouched():
    a = np.full((200, 200), 5, np.int64)
    a[0:40, 0:200] = 255  # a big void/background strip
    a[100:105, 100:110] = 3  # small island on the wall
    out = merge_tiny_regions(a, min_area=60)
    assert (out[0:40, :] == 255).all()  # keep class never rewritten
    assert (out[100:105, 100:110] == 5).all()  # island merged into wall, not void
    assert (out[100:105, 100:110] != 255).all()


def test_remove_enclosed_small_islands_on_wall():
    a = np.full((200, 200), 5, np.int64)  # everything is wall
    a[95:100, 95:103] = 1  # 40-px chair-coloured speck fully inside wall
    out, n = remove_enclosed_small_islands(a, hosts=[5], min_area=60)
    assert n == 1
    assert (out == 5).all()


def test_remove_enclosed_small_islands_picks_ceiling_host():
    a = np.full((200, 200), 7, np.int64)  # ceiling
    a[95:100, 95:103] = 1
    out, n = remove_enclosed_small_islands(a, hosts=[5, 7], min_area=60)
    assert n == 1
    assert (out[95:100, 95:103] == 7).all()


def test_border_touching_island_is_kept():
    a = np.full((200, 200), 5, np.int64)
    a[0:2, 90:100] = 1  # touches top border
    out, n = remove_enclosed_small_islands(a, hosts=[5], min_area=60)
    assert n == 0
    assert (out[0:2, 90:100] == 1).all()


def test_island_abutting_another_foreign_class_is_kept():
    a = np.full((200, 200), 5, np.int64)
    a[80:120, 80:120] = 4  # a big window region
    a[79:81, 90:98] = 1  # 16-px speck whose ring touches class 4
    out, n = remove_enclosed_small_islands(a, hosts=[5], min_area=60)
    assert n == 0
    assert (out[79:81, 90:98] == 1).all()


def test_island_inside_void_is_kept_and_void_untouched():
    a = np.full((200, 200), 5, np.int64)
    a[90:130, 90:130] = 255  # a void hole in the wall
    a[95:100, 95:103] = 1  # speck fully inside the void (not enclosed by a host)
    out, n = remove_enclosed_small_islands(a, hosts=[5], min_area=60)
    assert n == 0
    assert (out == a).all()  # void region, island and wall all untouched


def test_remove_islands_is_noop_when_min_area_is_tiny():
    a = np.full((100, 100), 5, np.int64)
    a[40:50, 40:50] = 1  # a 100-px island, well above a threshold of 2 px
    out, n = remove_enclosed_small_islands(a, hosts=[5], min_area=2)
    assert n == 0
    assert (out == a).all()
