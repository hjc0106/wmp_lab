"""Shared legacy WMP terrain category order and column mapping.

Both the terrain generator and environment category masks must use the same
column→category assignment as the original IsaacGym ``Terrain.evaluated_terrain``.
"""
from __future__ import annotations

from typing import Sequence

LEGACY_TERRAIN_NAMES: tuple[str, ...] = (
    "wave",
    "slope",
    "stairs_up",
    "stairs_down",
    "discrete",
    "gap",
    "climb",
    "tilt",
    "crawl",
    "rough_flat",
)

# Alias kept only for parsing external / historical configs.
TERRAIN_NAME_ALIASES: dict[str, str] = {"pit": "climb"}

DEFAULT_TERRAIN_PROPORTIONS: tuple[float, ...] = (
    0.0,
    0.05,
    0.15,
    0.15,
    0.0,
    0.25,
    0.25,
    0.05,
    0.05,
    0.05,
)

# Expected per-column counts for the default 20-column layout.
EXPECTED_COLUMN_COUNTS: dict[str, int] = {
    "slope": 1,
    "stairs_up": 3,
    "stairs_down": 3,
    "gap": 5,
    "climb": 5,
    "tilt": 1,
    "crawl": 1,
    "rough_flat": 1,
}


def normalize_terrain_name(name: str) -> str:
    """Map historical aliases (e.g. ``pit``) to canonical names."""
    return TERRAIN_NAME_ALIASES.get(name, name)


def cumulative_proportions(proportions: Sequence[float] | None = None) -> list[float]:
    props = list(proportions if proportions is not None else DEFAULT_TERRAIN_PROPORTIONS)
    return [sum(props[: i + 1]) for i in range(len(props))]


def legacy_choice(column: int, num_cols: int) -> float:
    """Old column selection value passed to ``make_terrain``."""
    return column / num_cols + 0.001


def category_index_from_choice(choice: float, proportions: Sequence[float] | None = None) -> int:
    """Return the terrain category index for a legacy ``choice`` value."""
    props = cumulative_proportions(proportions)
    for idx, bound in enumerate(props):
        if choice < bound:
            return idx
    return len(props) - 1


def category_name_from_choice(choice: float, proportions: Sequence[float] | None = None) -> str:
    return LEGACY_TERRAIN_NAMES[category_index_from_choice(choice, proportions)]


def column_category_indices(num_cols: int, proportions: Sequence[float] | None = None) -> list[int]:
    """Per-column category indices using the legacy ``choice = col/num_cols + 0.001`` rule."""
    return [category_index_from_choice(legacy_choice(col, num_cols), proportions) for col in range(num_cols)]


def column_category_names(num_cols: int, proportions: Sequence[float] | None = None) -> list[str]:
    return [LEGACY_TERRAIN_NAMES[i] for i in column_category_indices(num_cols, proportions)]


def column_category_counts(num_cols: int, proportions: Sequence[float] | None = None) -> dict[str, int]:
    counts = {name: 0 for name in LEGACY_TERRAIN_NAMES}
    for name in column_category_names(num_cols, proportions):
        counts[name] += 1
    return counts


def ordered_row_difficulty(row: int, num_rows: int) -> float:
    """Strict curriculum difficulty: 0.0, 0.1, …, 0.9 for 10 rows."""
    return row / num_rows


def tile_translation(row: int, column: int, size: tuple[float, float]) -> tuple[float, float]:
    """Translate a legacy tile whose local coordinates start at ``(0, 0)``."""
    return row * size[0], column * size[1]
