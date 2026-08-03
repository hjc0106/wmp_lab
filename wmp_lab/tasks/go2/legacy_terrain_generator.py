"""Legacy WMP terrain generator compatible with Isaac Lab ``TerrainImporter``."""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import trimesh
from isaaclab.utils.configclass import configclass
from isaaclab.utils.timer import Timer

from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_border

from .legacy_terrain_layout import DEFAULT_TERRAIN_PROPORTIONS, TERRAIN_CATEGORY_COLORS, tile_translation
from .legacy_terrain_utils import (
    CompatMode,
    accumulate_heightfield_tile,
    build_x_edge_mask,
    combine_meshes,
    stable_tile_seed,
)
from .terrains import make_legacy_tile

logger = logging.getLogger(__name__)


@configclass
class LegacyTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration for deterministic legacy WMP terrain generation."""

    # Resolved in __post_init__ because the generator class is declared below.
    class_type: type = None
    compat_mode: str = "legacy_exact"
    ordered_generation: bool = True
    slope_direction: str = "legacy"
    terrain_proportions: list[float] = list(DEFAULT_TERRAIN_PROPORTIONS)
    color_by_category: bool = False
    max_difficulty: float = 1.0

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = LegacyTerrainGenerator


class LegacyTerrainGenerator(TerrainGenerator):
    """Generate terrains using legacy column/difficulty rules with per-tile RNG."""

    cfg: LegacyTerrainGeneratorCfg

    def __init__(self, cfg: LegacyTerrainGeneratorCfg, device: str = "cpu"):
        if cfg.compat_mode not in ("legacy_exact", "legacy_fixed"):
            raise ValueError(f"Unsupported terrain compatibility mode: {cfg.compat_mode!r}")
        if cfg.slope_direction not in ("legacy", "up", "down"):
            raise ValueError(f"Unsupported slope direction: {cfg.slope_direction!r}")
        if not 0.0 <= cfg.max_difficulty <= 1.0:
            raise ValueError("Legacy terrain max_difficulty must be in [0, 1]")
        if tuple(cfg.size) != (8.0, 8.0) or cfg.horizontal_scale != 0.1 or cfg.vertical_scale != 0.005:
            raise ValueError("Legacy WMP terrains require size=8x8, horizontal_scale=0.1, vertical_scale=0.005")
        if len(cfg.terrain_proportions) != 10 or sum(cfg.terrain_proportions) <= 0:
            raise ValueError("Legacy WMP terrain proportions must contain 10 entries with a positive sum")
        if len(cfg.sub_terrains) == 0:
            # Isaac Lab requires at least one sub-terrain entry; legacy path ignores it.
            from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

            cfg.sub_terrains = {"legacy": SubTerrainBaseCfg(proportion=1.0, function=lambda d, c: ([], np.zeros(3)))}
        self.cfg = cfg
        self.device = device
        self.flat_patches = {}
        self.terrain_meshes: list[trimesh.Trimesh] = []
        self.terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.x_edge_mask: np.ndarray | None = None
        self.edge_mask_world_origin = np.zeros(2, dtype=np.float64)
        self.edge_mask_horizontal_scale = cfg.horizontal_scale

        if cfg.use_cache and cfg.seed is None:
            logger.warning("Legacy terrain cache enabled without seed; generation may not be reproducible.")

        with Timer("[INFO] Generating legacy terrains took"):
            self._generate_legacy_tiles()
        self._build_edge_mask()
        self._add_terrain_border()
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        transform = np.eye(4)
        transform[:2, -1] = -cfg.size[0] * cfg.num_rows * 0.5, -cfg.size[1] * cfg.num_cols * 0.5
        self.terrain_mesh.apply_transform(transform)
        self.terrain_origins += transform[:3, -1]

    def _generate_legacy_tiles(self) -> None:
        seed = 0 if self.cfg.seed is None else int(self.cfg.seed)
        compat: CompatMode = self.cfg.compat_mode  # type: ignore[assignment]
        slope_dir: Literal["legacy", "up", "down"] = self.cfg.slope_direction  # type: ignore[assignment]
        props = list(self.cfg.terrain_proportions)
        tile_pixels = int(self.cfg.size[0] / self.cfg.horizontal_scale)
        border_pixels = int(self.cfg.border_width / self.cfg.horizontal_scale)
        self._height_field_raw = np.zeros(
            (
                self.cfg.num_rows * tile_pixels + 2 * border_pixels,
                self.cfg.num_cols * tile_pixels + 2 * border_pixels,
            ),
            dtype=np.int16,
        )

        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = None
                if not self.cfg.ordered_generation:
                    difficulty_seed = stable_tile_seed(seed, "difficulty", row, col, compat)
                    difficulty = float(np.random.default_rng(difficulty_seed).uniform(0.0, 1.0))
                if difficulty is None:
                    difficulty = row / self.cfg.num_rows
                difficulty = min(difficulty, self.cfg.max_difficulty)
                meshes, origin, category, _diff, tile_hf = make_legacy_tile(
                    row=row,
                    column=col,
                    num_rows=self.cfg.num_rows,
                    num_cols=self.cfg.num_cols,
                    training_seed=seed,
                    compat_mode=compat,
                    proportions=props,
                    slope_direction=slope_dir,
                    difficulty_override=difficulty,
                )
                if tile_hf is not None:
                    accumulate_heightfield_tile(
                        self._height_field_raw,
                        tile_hf,
                        row,
                        col,
                        border_pixels=border_pixels,
                        tile_pixels=tile_pixels,
                    )
                mesh = combine_meshes(meshes)
                if self.cfg.color_by_category:
                    mesh.visual.vertex_colors = np.tile(
                        np.asarray(TERRAIN_CATEGORY_COLORS[category], dtype=np.uint8),
                        (len(mesh.vertices), 1),
                    )
                self._add_sub_terrain_legacy(mesh, origin, row, col)

    def _build_edge_mask(self) -> None:
        center_shift_x = self.cfg.size[0] * self.cfg.num_rows * 0.5
        center_shift_y = self.cfg.size[1] * self.cfg.num_cols * 0.5
        self.edge_mask_world_origin = np.array(
            [-center_shift_x - self.cfg.border_width, -center_shift_y - self.cfg.border_width],
            dtype=np.float64,
        )
        padded = np.pad(self._height_field_raw, ((0, 1), (0, 1)), mode="edge")
        self.x_edge_mask = build_x_edge_mask(
            padded,
            horizontal_scale=self.cfg.horizontal_scale,
            vertical_scale=self.cfg.vertical_scale,
            slope_threshold=self.cfg.slope_threshold,
        )

    def _add_sub_terrain_legacy(self, mesh: trimesh.Trimesh, origin: np.ndarray, row: int, col: int) -> None:
        transform = np.eye(4)
        # Legacy tile meshes use local coordinates in [0, size], not center-zero coordinates.
        transform[0:2, -1] = tile_translation(row, col, self.cfg.size)
        mesh = mesh.copy()
        mesh.apply_transform(transform)
        self.terrain_meshes.append(mesh)
        self.terrain_origins[row, col] = origin + transform[:3, -1]

    def _add_terrain_border(self) -> None:
        if self.cfg.border_width <= 0.0:
            return
        border_size = (
            self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
            self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
        )
        inner_size = (self.cfg.num_rows * self.cfg.size[0], self.cfg.num_cols * self.cfg.size[1])
        border_center = (
            self.cfg.num_rows * self.cfg.size[0] / 2,
            self.cfg.num_cols * self.cfg.size[1] / 2,
            -self.cfg.border_height / 2,
        )
        border_meshes = make_border(border_size, inner_size, height=abs(self.cfg.border_height), position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        if self.cfg.color_by_category:
            border.visual.vertex_colors = np.tile(
                np.asarray((105, 108, 103, 255), dtype=np.uint8),
                (len(border.vertices), 1),
            )
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        self.terrain_meshes.append(border)
