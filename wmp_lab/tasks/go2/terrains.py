# Copyright (c) 2026, WMP-Go2-Lab authors.
# SPDX-License-Identifier: BSD-3-Clause
"""Legacy-compatible WMP Go2 terrain generators.

Each ``legacy_*`` function builds one 8×8 m tile using the same formulas as
``legged_gym/utils/terrain.py`` in the IsaacGym WMP repo, but with isolated
per-tile RNG (see ``legacy_terrain_utils.TileRng``).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh

from .legacy_terrain_layout import (
    LEGACY_TERRAIN_NAMES,
    category_index_from_choice,
    cumulative_proportions,
    legacy_choice,
)
from .legacy_terrain_utils import (
    HORIZONTAL_SCALE,
    SLOPE_THRESHOLD,
    TILE_SIZE,
    VERTICAL_SCALE,
    CompatMode,
    SubTerrain,
    TileRng,
    box_trimesh,
    climb_terrain,
    combine_meshes,
    discrete_obstacles_terrain,
    gap_terrain,
    heightfield_to_trimesh_mesh,
    pyramid_sloped_terrain,
    pyramid_stairs_terrain,
    random_uniform_terrain,
    stable_tile_seed,
    terrain_origin_from_heightfield,
    validate_mesh,
    wave_terrain,
)


def _rough_noise(terrain: SubTerrain, rng: TileRng) -> SubTerrain:
    return random_uniform_terrain(
        terrain,
        rng,
        min_height=-0.05,
        max_height=0.05,
        step=0.005,
        downsampled_scale=0.2,
    )


def _hf_result(terrain: SubTerrain) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh = heightfield_to_trimesh_mesh(terrain.height_field_raw)
    origin = terrain_origin_from_heightfield(terrain.height_field_raw)
    validate_mesh(mesh)
    return [mesh], origin


def legacy_wave_terrain(difficulty: float, rng: TileRng) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain = SubTerrain.create()
    amplitude = 0.1 + 0.2 * difficulty
    wave_terrain(terrain, num_waves=5, amplitude=amplitude)
    _rough_noise(terrain, rng)
    return _hf_result(terrain)


def legacy_slope_terrain(
    difficulty: float,
    rng: TileRng,
    column: int,
    num_cols: int,
    proportions: list[float] | None = None,
    slope_direction: Literal["legacy", "up", "down"] = "legacy",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain = SubTerrain.create()
    slope = difficulty * 0.4
    if slope_direction == "up":
        pass
    elif slope_direction == "down":
        slope *= -1
    else:
        choice = legacy_choice(column, num_cols)
        props = cumulative_proportions(proportions)
        if choice < (props[0] + props[1]) / 2:
            slope *= -1
    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
    _rough_noise(terrain, rng)
    return _hf_result(terrain)


def legacy_stairs_terrain(
    difficulty: float,
    rng: TileRng,
    column: int,
    num_cols: int,
    proportions: list[float] | None = None,
    *,
    stairs_up: bool | None = None,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain = SubTerrain.create()
    step_height = 0.05 + 0.18 * difficulty
    if stairs_up is None:
        choice = legacy_choice(column, num_cols)
        props = cumulative_proportions(proportions)
        if choice < props[2]:
            step_height *= -1
    elif not stairs_up:
        step_height *= -1
    step_width = 0.30 + rng.py_rng.random() * 0.04
    pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=3.0)
    return _hf_result(terrain)


def legacy_discrete_terrain(difficulty: float, rng: TileRng) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain = SubTerrain.create()
    obstacle_h = 0.05 + difficulty * 0.2
    discrete_obstacles_terrain(terrain, rng, obstacle_h, 1.0, 2.0, 20, platform_size=3.0)
    _rough_noise(terrain, rng)
    return _hf_result(terrain)


def legacy_gap_terrain(difficulty: float, rng: TileRng) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain = SubTerrain.create()
    gap_terrain(terrain, rng, gap_size=difficulty, platform_size=4.0)
    _rough_noise(terrain, rng)
    return _hf_result(terrain)


def legacy_climb_terrain(difficulty: float, rng: TileRng) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain = SubTerrain.create()
    climb_terrain(terrain, rng, depth=0.6 * difficulty)
    _rough_noise(terrain, rng)
    return _hf_result(terrain)


def legacy_tilt_terrain(difficulty: float, rng: TileRng) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    tile_w = TILE_SIZE
    tilt_width = 0.32 - 0.04 * difficulty
    box_z = 1.0
    box_x = 0.4 + 0.4 * rng.py_rng.random()
    half_gap_y = (tile_w - tilt_width) / 2.0
    quarter_gap_y = (tile_w - tilt_width) / 4.0
    boxes = []
    specs = [
        (2.0 + box_x / 2.0, tile_w / 2.0 - tilt_width / 2.0 - quarter_gap_y),
        (2.0 + box_x / 2.0, tile_w / 2.0 + tilt_width / 2.0 + quarter_gap_y),
        (tile_w - 2.0 - box_x / 2.0, tile_w / 2.0 - tilt_width / 2.0 - quarter_gap_y),
        (tile_w - 2.0 - box_x / 2.0, tile_w / 2.0 + tilt_width / 2.0 + quarter_gap_y),
    ]
    for cx, cy in specs:
        boxes.append(
            box_trimesh(
                np.array([box_x, half_gap_y, box_z], dtype=np.float32),
                np.array([cx, cy, box_z / 2.0], dtype=np.float32),
            )
        )
    ground = heightfield_to_trimesh_mesh(np.zeros((int(tile_w / HORIZONTAL_SCALE),) * 2, dtype=np.int16))
    mesh = combine_meshes([ground, *boxes])
    origin = np.array([tile_w / 2.0, tile_w / 2.0, 0.0], dtype=np.float64)
    validate_mesh(mesh)
    return [mesh], origin


def legacy_crawl_terrain(
    difficulty: float,
    rng: TileRng,
    compat_mode: CompatMode = "legacy_exact",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    tile_w = TILE_SIZE
    crawl_height = 0.35 - 0.15 * difficulty
    box_x = 0.2 + 0.2 * rng.py_rng.random()
    box_z = 1.0
    front_x = 2.0 + box_x / 2.0
    if compat_mode == "legacy_fixed":
        back_x = tile_w - 2.0 - box_x / 2.0
    else:
        back_x = front_x
    bar_center_z = crawl_height + box_z / 2.0
    bars = [
        box_trimesh(
            np.array([box_x, tile_w, box_z], dtype=np.float32),
            np.array([front_x, tile_w / 2.0, bar_center_z], dtype=np.float32),
        ),
        box_trimesh(
            np.array([box_x, tile_w, box_z], dtype=np.float32),
            np.array([back_x, tile_w / 2.0, bar_center_z], dtype=np.float32),
        ),
    ]
    ground = heightfield_to_trimesh_mesh(np.zeros((int(tile_w / HORIZONTAL_SCALE),) * 2, dtype=np.int16))
    mesh = combine_meshes([ground, *bars])
    origin = np.array([tile_w / 2.0, tile_w / 2.0, 0.0], dtype=np.float64)
    validate_mesh(mesh)
    return [mesh], origin


def legacy_rough_flat_terrain(_difficulty: float, rng: TileRng) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    terrain = SubTerrain.create()
    random_uniform_terrain(terrain, rng, -0.05, 0.05, step=0.005, downsampled_scale=0.2)
    return _hf_result(terrain)


def make_legacy_tile(
    row: int,
    column: int,
    num_rows: int,
    num_cols: int,
    training_seed: int,
    compat_mode: CompatMode = "legacy_exact",
    proportions: list[float] | None = None,
    slope_direction: Literal["legacy", "up", "down"] = "legacy",
    difficulty_override: float | None = None,
) -> tuple[list[trimesh.Trimesh], np.ndarray, str, float]:
    """Generate one tile; returns meshes, origin, category name, difficulty."""
    from .legacy_terrain_layout import ordered_row_difficulty

    choice = legacy_choice(column, num_cols)
    cat_idx = category_index_from_choice(choice, proportions)
    cat_name = LEGACY_TERRAIN_NAMES[cat_idx]
    difficulty = ordered_row_difficulty(row, num_rows) if difficulty_override is None else difficulty_override
    tile_seed = stable_tile_seed(training_seed, cat_name, row, column, compat_mode)
    rng = TileRng(tile_seed)

    if cat_name == "wave":
        meshes, origin = legacy_wave_terrain(difficulty, rng)
    elif cat_name == "slope":
        meshes, origin = legacy_slope_terrain(difficulty, rng, column, num_cols, proportions, slope_direction)
    elif cat_name == "stairs_up":
        meshes, origin = legacy_stairs_terrain(difficulty, rng, column, num_cols, proportions, stairs_up=False)
    elif cat_name == "stairs_down":
        meshes, origin = legacy_stairs_terrain(difficulty, rng, column, num_cols, proportions, stairs_up=True)
    elif cat_name == "discrete":
        meshes, origin = legacy_discrete_terrain(difficulty, rng)
    elif cat_name == "gap":
        meshes, origin = legacy_gap_terrain(difficulty, rng)
    elif cat_name == "climb":
        meshes, origin = legacy_climb_terrain(difficulty, rng)
    elif cat_name == "tilt":
        meshes, origin = legacy_tilt_terrain(difficulty, rng)
    elif cat_name == "crawl":
        meshes, origin = legacy_crawl_terrain(difficulty, rng, compat_mode)
    else:
        meshes, origin = legacy_rough_flat_terrain(difficulty, rng)
    return meshes, origin, cat_name, difficulty
