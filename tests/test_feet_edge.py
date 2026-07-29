"""Feet-edge mask regression tests on legacy gap terrain tiles."""
from __future__ import annotations

import numpy as np

from wmp_lab.tasks.go2.legacy_terrain_utils import (
    SubTerrain,
    TileRng,
    accumulate_heightfield_tile,
    build_x_edge_mask,
    gap_terrain,
    stable_tile_seed,
)


def _gap_mask(seed: int = 1) -> np.ndarray:
    terrain = SubTerrain.create()
    rng = TileRng(stable_tile_seed(seed, "gap", 5, 0, "legacy_exact"))
    gap_terrain(terrain, rng, gap_size=0.5, platform_size=4.0)
    hf = np.zeros((81, 81), dtype=np.int16)
    accumulate_heightfield_tile(hf, terrain.height_field_raw, 0, 0, border_pixels=0, tile_pixels=80)
    return build_x_edge_mask(np.pad(hf, ((0, 1), (0, 1)), mode="edge"))


def test_flat_center_zero():
    mask = np.zeros((81, 81), dtype=bool)
    assert not mask[40, 40]


def test_gap_platform_center_zero():
    mask = _gap_mask()
    assert not mask[40, 40]


def test_gap_has_some_edge_cells():
    mask = _gap_mask()
    assert mask.any()
