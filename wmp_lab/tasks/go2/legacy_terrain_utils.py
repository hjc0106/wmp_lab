"""Ported IsaacGym terrain helpers with per-tile RNG isolation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import trimesh
from scipy.interpolate import RectBivariateSpline

HORIZONTAL_SCALE = 0.1
VERTICAL_SCALE = 0.005
SLOPE_THRESHOLD = 0.75
TILE_SIZE = 8.0

CompatMode = Literal["legacy_exact", "legacy_fixed"]


def stable_tile_seed(
    training_seed: int,
    terrain_type: str,
    row: int,
    column: int,
    compat_mode: CompatMode,
) -> int:
    """Derive a reproducible 32-bit seed; never use Python ``hash()``."""
    key = f"{training_seed}:{terrain_type}:{row}:{column}:{compat_mode}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@dataclass
class TileRng:
    """Independent NumPy and Python RNGs for one terrain tile."""

    seed: int
    np_rng: np.random.Generator = field(init=False)
    py_rng: np.random.RandomState = field(init=False)

    def __post_init__(self) -> None:
        self.np_rng = np.random.default_rng(self.seed)
        self.py_rng = np.random.RandomState(self.seed & 0xFFFFFFFF)


@dataclass
class SubTerrain:
    """Minimal IsaacGym ``SubTerrain`` stand-in (80×80 for 8 m @ 0.1 m)."""

    width: int
    length: int
    horizontal_scale: float = HORIZONTAL_SCALE
    vertical_scale: float = VERTICAL_SCALE
    height_field_raw: np.ndarray = field(init=False)

    @classmethod
    def create(cls, tile_size: float = TILE_SIZE, horizontal_scale: float = HORIZONTAL_SCALE) -> SubTerrain:
        pixels = int(tile_size / horizontal_scale)
        return cls(width=pixels, length=pixels, horizontal_scale=horizontal_scale)

    def __post_init__(self) -> None:
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)


def random_uniform_terrain(
    terrain: SubTerrain,
    rng: TileRng,
    min_height: float,
    max_height: float,
    step: float = 0.005,
    downsampled_scale: float | None = 0.2,
) -> SubTerrain:
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    min_h = int(min_height / terrain.vertical_scale)
    max_h = int(max_height / terrain.vertical_scale)
    step_u = int(step / terrain.vertical_scale)
    heights_range = np.arange(min_h, max_h + step_u, step_u)
    ds_x = int(terrain.width * terrain.horizontal_scale / downsampled_scale)
    ds_y = int(terrain.length * terrain.horizontal_scale / downsampled_scale)
    height_field_downsampled = rng.np_rng.choice(heights_range, (ds_x, ds_y))

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])
    # RectBivariateSpline replaces removed scipy.interp2d; kx=ky=1 for linear interpolation.
    func = RectBivariateSpline(x, y, height_field_downsampled, kx=1, ky=1)
    x_up = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_up = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_up = func(x_up, y_up)
    terrain.height_field_raw += np.rint(z_up).astype(np.int16)
    return terrain


def wave_terrain(terrain: SubTerrain, num_waves: int, amplitude: float) -> SubTerrain:
    amplitude_u = int(0.5 * amplitude / terrain.vertical_scale)
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        terrain.height_field_raw += (
            amplitude_u * np.cos(yy / div) + amplitude_u * np.sin(xx / div)
        ).astype(terrain.height_field_raw.dtype)
    return terrain


def pyramid_sloped_terrain(terrain: SubTerrain, slope: float, platform_size: float = 3.0) -> SubTerrain:
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_u = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_u
    x2 = terrain.width // 2 + platform_u
    y1 = terrain.length // 2 - platform_u
    y2 = terrain.length // 2 + platform_u
    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain


def pyramid_stairs_terrain(
    terrain: SubTerrain,
    step_width: float,
    step_height: float,
    platform_size: float = 3.0,
) -> SubTerrain:
    step_width_u = int(step_width / terrain.horizontal_scale)
    step_height_u = int(step_height / terrain.vertical_scale)
    platform_u = int(platform_size / terrain.horizontal_scale)
    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_u and (stop_y - start_y) > platform_u:
        start_x += step_width_u
        stop_x -= step_width_u
        start_y += step_width_u
        stop_y -= step_width_u
        height += step_height_u
        terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height
    return terrain


def discrete_obstacles_terrain(
    terrain: SubTerrain,
    rng: TileRng,
    max_height: float,
    min_size: float,
    max_size: float,
    num_rects: int,
    platform_size: float = 3.0,
) -> SubTerrain:
    max_height_u = int(max_height / terrain.vertical_scale)
    min_size_u = int(min_size / terrain.horizontal_scale)
    max_size_u = int(max_size / terrain.horizontal_scale)
    platform_u = int(platform_size / terrain.horizontal_scale)
    i, j = terrain.height_field_raw.shape
    height_range = [-max_height_u, -max_height_u // 2, max_height_u // 2, max_height_u]
    width_range = range(min_size_u, max_size_u, 4)
    length_range = range(min_size_u, max_size_u, 4)
    for _ in range(num_rects):
        width = rng.py_rng.choice(width_range)
        length = rng.py_rng.choice(length_range)
        start_i = rng.py_rng.choice(range(0, i - width, 4))
        start_j = rng.py_rng.choice(range(0, j - length, 4))
        terrain.height_field_raw[start_i : start_i + width, start_j : start_j + length] = rng.py_rng.choice(
            height_range
        )
    x1 = (terrain.width - platform_u) // 2
    x2 = (terrain.width + platform_u) // 2
    y1 = (terrain.length - platform_u) // 2
    y2 = (terrain.length + platform_u) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def gap_terrain(terrain: SubTerrain, rng: TileRng, gap_size: float, platform_size: float = 4.0) -> SubTerrain:
    gap_u = int(gap_size / terrain.horizontal_scale)
    platform_u = int(platform_size / terrain.horizontal_scale)
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = int(center_x - 1 / terrain.horizontal_scale)
    x2 = int(center_x + 2 / terrain.horizontal_scale)
    x3 = x1 - gap_u
    x4 = x2 + gap_u
    width = 1 + 1.0 * rng.py_rng.random()
    half_width = width / 2
    y1 = int(center_y - half_width / terrain.horizontal_scale)
    y2 = int(center_y + half_width / terrain.horizontal_scale)
    x5 = gap_u
    terrain.height_field_raw[:, :] = -1000
    terrain.height_field_raw[x5:x3, y1:y2] = 0
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    terrain.height_field_raw[x4:, y1:y2] = 0
    return terrain


def climb_terrain(terrain: SubTerrain, rng: TileRng, depth: float) -> SubTerrain:
    depth_u = int(depth / terrain.vertical_scale)
    x1 = int(1 / terrain.horizontal_scale)
    length = 1.0 + 0.2 * rng.py_rng.random()
    x2 = int((1 + length) / terrain.horizontal_scale)
    x3 = int(6 / terrain.horizontal_scale)
    length2 = 1.0 + 0.2 * rng.py_rng.random()
    x4 = int((6 + length2) / terrain.horizontal_scale)
    terrain.height_field_raw[x1:x2, :] = depth_u
    terrain.height_field_raw[x3:x4, :] = depth_u
    return terrain


def convert_heightfield_to_trimesh(
    height_field_raw: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hf = height_field_raw
    num_rows, num_cols = hf.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3
    return vertices, triangles


def box_trimesh(size: np.ndarray, center_position: np.ndarray) -> trimesh.Trimesh:
    size = np.asarray(size, dtype=np.float32)
    center_position = np.asarray(center_position, dtype=np.float32)
    vertices = np.empty((8, 3), dtype=np.float32)
    vertices[:] = center_position
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2
    faces = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
            [0, 4, 2],
            [2, 4, 6],
            [4, 5, 6],
            [5, 7, 6],
            [1, 3, 5],
            [3, 7, 5],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def combine_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if not meshes:
        raise ValueError("combine_meshes requires at least one mesh")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def terrain_origin_from_heightfield(
    height_field_raw: np.ndarray,
    tile_size: float = TILE_SIZE,
    horizontal_scale: float = HORIZONTAL_SCALE,
    vertical_scale: float = VERTICAL_SCALE,
) -> np.ndarray:
    """Origin z = max height in the center 2×2 m region (legacy rule)."""
    x1 = int((tile_size / 2.0 - 1.0) / horizontal_scale)
    x2 = int((tile_size / 2.0 + 1.0) / horizontal_scale)
    y1 = int((tile_size / 2.0 - 1.0) / horizontal_scale)
    y2 = int((tile_size / 2.0 + 1.0) / horizontal_scale)
    origin_z = float(np.max(height_field_raw[x1:x2, y1:y2]) * vertical_scale)
    return np.array([tile_size / 2.0, tile_size / 2.0, origin_z], dtype=np.float64)


def heightfield_to_trimesh_mesh(
    height_field_raw: np.ndarray,
    horizontal_scale: float = HORIZONTAL_SCALE,
    vertical_scale: float = VERTICAL_SCALE,
    slope_threshold: float | None = SLOPE_THRESHOLD,
) -> trimesh.Trimesh:
    # An 80-sample legacy height field spans 0..7.9 m. Repeat its terminal
    # samples so each independently generated tile has collision coverage up
    # to the full 8.0 m tile boundary.
    height_field_raw = np.pad(height_field_raw, ((0, 1), (0, 1)), mode="edge")
    vertices, triangles = convert_heightfield_to_trimesh(
        height_field_raw, horizontal_scale, vertical_scale, slope_threshold
    )
    return trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)


def validate_mesh(mesh: trimesh.Trimesh, tile_size: float = TILE_SIZE) -> None:
    verts = mesh.vertices
    faces = mesh.faces
    if not np.all(np.isfinite(verts)):
        raise ValueError("mesh vertices contain NaN or Inf")
    if faces.size == 0:
        raise ValueError("mesh has no faces")
    if np.any(faces < 0) or np.any(faces >= len(verts)):
        raise ValueError("mesh face indices out of bounds")
    if verts[:, 0].max() > tile_size + 0.2 or verts[:, 1].max() > tile_size + 0.2:
        raise ValueError("mesh extends beyond tile bounds")
    if len(np.unique(faces.ravel())) < 3:
        raise ValueError("mesh has degenerate topology")
