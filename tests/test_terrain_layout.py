"""Terrain layout consistency between generator and environment."""
from __future__ import annotations

from wmp_lab.tasks.go2.legacy_terrain_layout import (
    column_category_names,
    LEGACY_TERRAIN_NAMES,
    normalize_terrain_name,
    ordered_row_difficulty,
    TERRAIN_CATEGORY_COLORS,
    tile_translation,
)


def test_play_single_column_slope_is_negative_in_legacy_mode():
    names = column_category_names(1)
    assert names == ["slope"]


def test_difficulty_independent_of_column():
    assert ordered_row_difficulty(3, 10) == 0.3


def test_historical_pit_alias_maps_to_climb():
    assert normalize_terrain_name("pit") == "climb"


def test_local_tiles_are_not_shifted_by_half_a_tile():
    assert tile_translation(0, 0, (8.0, 8.0)) == (0.0, 0.0)
    assert tile_translation(9, 19, (8.0, 8.0)) == (72.0, 152.0)


def test_viewer_palette_covers_every_category():
    assert set(TERRAIN_CATEGORY_COLORS) == set(LEGACY_TERRAIN_NAMES)
